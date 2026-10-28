"""Experimental motion challenge. This is NOT a validated anti-spoofing model."""
import secrets
import time
import numpy as np
from verification import analyze_image, log_event

COMPLETED = 'Motion challenge completed. This does not prove liveness or reject replay attacks.'


def start_challenge():
    turns = ['left', 'right']
    if secrets.randbelow(2):
        turns.reverse()
    state = dict(steps=['center', *turns, 'center'], step=0, hits=0,
                 started=time.monotonic(), embedding=None)
    return state, 'Look straight at the camera.'


def advance_challenge(state, pose, now):
    """Require two consecutive frames per requested pose within 30 seconds."""
    state = dict(state)
    if now - state['started'] > 30:
        raise ValueError('Challenge expired. Start a new challenge.')
    if state['step'] >= len(state['steps']):
        return state, COMPLETED
    target = state['steps'][state['step']]
    correct = {'center': abs(pose) < 0.12, 'left': pose < -0.22, 'right': pose > 0.22}[target]
    state['hits'] = state['hits'] + 1 if correct else 0
    if state['hits'] >= 2:
        state['step'] += 1
        state['hits'] = 0
    if state['step'] == len(state['steps']):
        return state, COMPLETED
    target = state['steps'][state['step']]
    return state, ('Look straight at the camera.' if target == 'center' else f'Turn your nose toward the {target} edge of the preview.')


def track_frame(image, state):
    if image is None or not state:
        return image, state, 'Start a challenge, then enable the webcam.'
    try:
        view, face = analyze_image(image)
        embedding = face.normed_embedding
        if state['embedding'] is None:
            state = dict(state, embedding=embedding)
        elif float(np.dot(state['embedding'], embedding)) < 0.40:
            raise ValueError('Face continuity lost. Start again.')
        left, right, nose = face.kps[:3]
        axis = right - left
        pose = float(np.dot(nose - (left + right) / 2, axis) / max(np.dot(axis, axis), 1))
        state, message = advance_challenge(state, pose, time.monotonic())
        return view, state, message
    except ValueError as error:
        log_event('webcam_rejected', reason=str(error))
        return image, None, str(error)
    except Exception as error:
        log_event('webcam_failed', error_type=type(error).__name__)
        return image, None, 'Camera processing failed. Start again or check server logs.'
