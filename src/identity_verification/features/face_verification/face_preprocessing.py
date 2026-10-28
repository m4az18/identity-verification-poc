"""Face detection in upload orientation and five-point face alignment.

The preserved detector remains responsible for locating exactly one usable face and
for its quality gate.  This adapter standardizes geometry around that implementation so
ID portraits and live captures reach the recognizer in the same canonical form.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np


@dataclass(slots=True, frozen=True)
class StandardizedFace:
    face: object
    operations: tuple[str, ...]
    aligned_image_rgb: np.ndarray | None = None
    upright_score: float = 0.0
    preview_steps: tuple[tuple[str, str, np.ndarray], ...] = ()


def analyze_standardized_face(verification, image_rgb: np.ndarray) -> StandardizedFace:
    """Run preserved quality checks and return a canonical-alignment embedding.

    Unit-test doubles and older integrations keep using ``analyze_image``.  The loaded
    preserved module exposes the lower-level detector and recognizer functions required
    to materialize the aligned crop.
    """

    if not isinstance(verification, ModuleType):
        _, face = verification.analyze_image(image_rgb)
        return StandardizedFace(face=face, operations=(), upright_score=1.0)

    detector_rgb, preprocessing_operations, detector_steps = _prepare_detector_input(image_rgb)
    detection_rgb, crop_bgr = verification.detect_and_crop(detector_rgb)
    if crop_bgr is None:
        raise ValueError("Exactly one face is required in each image.")

    face = verification.get_face_features(crop_bgr)
    embedding = getattr(face, "normed_embedding", None) if face is not None else None
    if embedding is None or not np.isfinite(embedding).all():
        raise ValueError("Could not extract facial features. Try a clearer photo.")

    landmarks = getattr(face, "kps", None)
    if landmarks is None or np.asarray(landmarks).shape != (5, 2):
        # InsightFace normally supplies these five points. Its returned embedding is
        # already aligned internally, so retaining it is the safe compatibility path.
        return StandardizedFace(
            face=face,
            operations=(*preprocessing_operations, "recognizer_internal_alignment"),
            upright_score=0.5,
            preview_steps=(
                *detector_steps,
                ("Face detected", "YOLOv8 face localization", detection_rgb),
                ("Portrait crop", "Centered detected region", crop_bgr[:, :, ::-1].copy()),
            ),
        )

    landmarks = np.asarray(landmarks, dtype=np.float32)
    upright_score = _upright_landmark_score(landmarks)
    aligned_bgr, mirrored = align_face_crop(crop_bgr, landmarks)
    standardized = _embedding_from_aligned_crop(verification, aligned_bgr, face)
    operations = [
        *preprocessing_operations,
        "five_point_alignment",
        "canonical_112x112_crop",
        "horizontal_flip_embedding_ensemble",
    ]
    if mirrored:
        operations.insert(0, "horizontal_flip_corrected")
    aligned_rgb = aligned_bgr[:, :, ::-1].copy()
    return StandardizedFace(
        face=standardized,
        operations=tuple(operations),
        aligned_image_rgb=aligned_rgb,
        upright_score=upright_score,
        preview_steps=(
            *detector_steps,
            ("Face detected", "YOLOv8 face localization", detection_rgb),
            ("Portrait crop", "Centered detected region", crop_bgr[:, :, ::-1].copy()),
            ("Aligned face", "Five landmarks, centered and resized to 112 x 112", aligned_rgb),
        ),
    )


def _prepare_detector_input(
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[str, str, np.ndarray], ...]]:
    """Create the standardized image that YOLO receives.

    Every pixel-level transform happens before face detection. Random crops are training
    augmentation and are intentionally excluded from verification. Ultralytics performs
    the final uint8-to-float [0, 1] scaling internally when this image enters YOLO.
    """

    import cv2

    image = np.ascontiguousarray(image_rgb[:, :, :3], dtype=np.uint8)
    height, width = image.shape[:2]
    target = 640
    scale = min(target / max(width, 1), target / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    standardized = np.full((target, target, 3), 114, dtype=np.uint8)
    x = (target - resized_width) // 2
    y = (target - resized_height) // 2
    standardized[y : y + resized_height, x : x + resized_width] = resized

    grayscale = cv2.cvtColor(standardized, cv2.COLOR_RGB2GRAY)
    denoised = cv2.bilateralFilter(grayscale, 5, 20, 20)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    balanced = clahe.apply(denoised)
    enhanced_gray = cv2.addWeighted(denoised, 0.75, balanced, 0.25, 0)
    enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    operations = (
        "rgb_decode_and_exif_orientation",
        "letterbox_resize_640x640",
        "grayscale_three_channel_conversion_before_detection",
        "bilateral_noise_reduction_before_detection",
        "local_contrast_normalization_before_detection",
        "yolo_native_pixel_rescaling_0_1",
    )
    steps = (
        ("Detector resize", "Aspect-preserving 640 x 640 letterbox", standardized),
        (
            "Black and white",
            "Grayscale converted to three equal channels for model compatibility",
            cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB),
        ),
        (
            "Noise reduction",
            "Edge-preserving bilateral filter before YOLO",
            cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB),
        ),
        ("Contrast normalization", "CLAHE grayscale correction before YOLO", enhanced),
    )
    return enhanced, operations, steps


def _upright_landmark_score(landmarks: np.ndarray) -> float:
    """Score pose orientation from eyes, nose and mouth without identity matching."""

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_vector = right_eye - left_eye
    mouth_vector = right_mouth - left_mouth
    eye_width = max(float(np.linalg.norm(eye_vector)), 1e-6)
    mouth_width = max(float(np.linalg.norm(mouth_vector)), 1e-6)
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    face_height = max(float(np.linalg.norm(mouth_center - eye_center)), 1e-6)

    eye_horizontal = abs(float(eye_vector[0])) / eye_width
    mouth_horizontal = abs(float(mouth_vector[0])) / mouth_width
    mouth_below_eyes = float(np.clip((mouth_center[1] - eye_center[1]) / face_height, 0.0, 1.0))
    nose_below_eyes = 1.0 if nose[1] > eye_center[1] else 0.0
    nose_above_mouth = 1.0 if nose[1] < mouth_center[1] else 0.0
    score = (
        0.35 * eye_horizontal
        + 0.25 * mouth_horizontal
        + 0.2 * mouth_below_eyes
        + 0.1 * nose_below_eyes
        + 0.1 * nose_above_mouth
    )
    return float(np.clip(score, 0.0, 1.0))


def align_face_crop(
    crop_bgr: np.ndarray,
    padded_landmarks: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Deskew and crop a detected face using its five padded-crop landmarks."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - face extra always supplies OpenCV
        raise RuntimeError("OpenCV is required for face alignment.") from exc

    padding = max(crop_bgr.shape[:2]) // 2
    padded = cv2.copyMakeBorder(
        crop_bgr,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
    )
    landmarks = padded_landmarks.copy()
    mirrored = bool(landmarks[0, 0] > landmarks[1, 0])
    if mirrored:
        padded = cv2.flip(padded, 1)
        landmarks[:, 0] = padded.shape[1] - 1 - landmarks[:, 0]

    canonical = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, canonical, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("Facial landmarks could not be aligned reliably.")
    aligned = cv2.warpAffine(
        padded,
        matrix,
        (112, 112),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return aligned, mirrored


def _embedding_from_aligned_crop(verification, aligned_bgr: np.ndarray, fallback_face):
    models = getattr(getattr(verification, "auraface", None), "models", {})
    recognizer = models.get("recognition") if isinstance(models, dict) else None
    if recognizer is None or not hasattr(recognizer, "get_feat"):
        return fallback_face

    original = np.asarray(recognizer.get_feat(aligned_bgr), dtype=np.float32).reshape(-1)
    flipped = np.asarray(
        recognizer.get_feat(np.fliplr(aligned_bgr).copy()),
        dtype=np.float32,
    ).reshape(-1)
    if not np.isfinite(original).all() or not np.isfinite(flipped).all():
        raise ValueError("Could not extract facial features from the aligned face.")
    original_norm = float(np.linalg.norm(original))
    flipped_norm = float(np.linalg.norm(flipped))
    if original_norm <= 0 or flipped_norm <= 0:
        raise ValueError("Could not extract facial features from the aligned face.")
    features = original / original_norm + flipped / flipped_norm
    norm = float(np.linalg.norm(features))
    if not np.isfinite(features).all() or norm <= 0:
        raise ValueError("Could not extract facial features from the aligned face.")
    return SimpleNamespace(normed_embedding=features / norm)
