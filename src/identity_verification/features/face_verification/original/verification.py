import cv2
import numpy as np
import json
import logging
import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from config import DETECTION_CONFIDENCE, DETECTION_SIZE, CROP_MARGIN
from quality import check_quality

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('verification')


def log_event(event, **fields):
    logger.info(json.dumps({'event': event, **fields}))

MODEL_REVISIONS = {
    'arnabdhar/YOLOv8-Face-Detection': '52fa54977207fa4f021de949b515fb19dcab4488',
    'fal/AuraFace-v1': 'af6d057c9b0ec4071d4c49c80e3539258798b609',
}

def initialize_models():
    yolo_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt",
        revision=MODEL_REVISIONS['arnabdhar/YOLOv8-Face-Detection'],
    )
    yolo_model = YOLO(yolo_path)

    snapshot_download(
        "fal/AuraFace-v1",
        revision=MODEL_REVISIONS['fal/AuraFace-v1'],
        local_dir="models/auraface",
        allow_patterns=["*.onnx"],
    )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    auraface_app = FaceAnalysis(name='auraface', providers=['CPUExecutionProvider'],
        root='.', allowed_modules=['detection', 'recognition'], sess_options=session_options)
    auraface_app.prepare(ctx_id=-1, det_size=(DETECTION_SIZE, DETECTION_SIZE))

    return yolo_model, auraface_app


yolo = auraface = None

def load_models():
    global yolo, auraface
    if yolo is None:
        log_event('model_loading')
        try:
            yolo, auraface = initialize_models()
        except Exception as error:
            log_event('model_loading_failed', error_type=type(error).__name__)
            raise
        log_event('models_loaded', revisions=MODEL_REVISIONS)


def detect_and_crop(image_rgb):
    """Detect a face, return a visualization and a crop for AuraFace."""
    if image_rgb is None:
        return None, None

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    load_models()
    results = yolo(img_bgr, verbose=False, conf=DETECTION_CONFIDENCE, imgsz=DETECTION_SIZE)

    if len(results[0].boxes) != 1:
        return image_rgb, None

    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    check_quality(img_bgr[max(0, y1):y2, max(0, x1):x2])
    margin_y = int((y2 - y1) * CROP_MARGIN)
    margin_x = int((x2 - x1) * CROP_MARGIN)
    crop_y1 = max(0, y1 - margin_y)
    crop_y2 = min(img_bgr.shape[0], y2 + margin_y)
    crop_x1 = max(0, x1 - margin_x)
    crop_x2 = min(img_bgr.shape[1], x2 + margin_x)
    cropped_face = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    visualization = img_bgr.copy()
    cv2.rectangle(visualization, (x1, y1), (x2, y2), (34, 197, 94), 3)
    visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)

    return visualization_rgb, cropped_face


def get_face_features(cropped_face_bgr):
    """Return AuraFace landmarks and embedding for one detected face."""
    padding = max(cropped_face_bgr.shape[:2]) // 2
    padded_face = cv2.copyMakeBorder(
        cropped_face_bgr,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
    )
    faces = auraface.get(padded_face)
    if len(faces) != 1:
        return None
    return faces[0]


def analyze_image(image_rgb):
    """Shared checks and feature extraction for the UI, webcam and evaluator."""
    view, crop = detect_and_crop(image_rgb)
    if crop is None:
        raise ValueError('Exactly one face is required in each image.')
    face = get_face_features(crop)
    if face is None or not np.isfinite(face.normed_embedding).all():
        raise ValueError('Could not extract facial features. Try a clearer photo.')
    return view, face

