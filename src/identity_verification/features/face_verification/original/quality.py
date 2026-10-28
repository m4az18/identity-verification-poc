"""Small, configurable image-quality checks on the detected face region."""
import cv2
from config import MIN_FACE_SIZE, MIN_BLUR_VARIANCE, MIN_BRIGHTNESS, MAX_BRIGHTNESS


def check_quality(face_bgr):
    if face_bgr is None or face_bgr.size == 0 or min(face_bgr.shape[:2]) < MIN_FACE_SIZE:
        raise ValueError('Face is too small; move closer or use a larger image.')
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if not MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS:
        raise ValueError('Face lighting is too dark or too bright.')
    if blur < MIN_BLUR_VARIANCE:
        raise ValueError('Face is too blurred; hold still and refocus.')
    return {'brightness': brightness, 'blur_variance': blur}
