"""Deterministic document-image normalization shared by OCR and face verification.

The loader applies this module once.  Downstream OCR and biometric adapters therefore
receive the same pixels instead of independently rotating or enhancing an upload.
Transforms are intentionally conservative: when a reliable card boundary or skew line
cannot be found, the original geometry is retained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class DocumentNormalization:
    image_rgb: np.ndarray
    operations: tuple[str, ...] = ()


def normalize_document_image(image_rgb: np.ndarray) -> DocumentNormalization:
    """Isolate, straighten and gently normalize one rendered/uploaded ID page."""

    image = _as_rgb_uint8(image_rgb)
    operations: list[str] = []

    try:
        import cv2
    except ImportError:
        return DocumentNormalization(image_rgb=image)

    geometry, geometry_operation = _normalize_geometry(cv2, image)
    if geometry_operation:
        operations.append(geometry_operation)

    normalized = _normalize_illumination(cv2, geometry)
    if not np.array_equal(normalized, geometry):
        operations.append("illumination_contrast_normalized")

    return DocumentNormalization(
        image_rgb=np.ascontiguousarray(normalized),
        operations=tuple(operations),
    )


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Document image must be an RGB image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image[:, :, :3])


def _normalize_geometry(cv2, image: np.ndarray) -> tuple[np.ndarray, str | None]:
    scale = min(1.0, 1200.0 / max(image.shape[:2]))
    preview = image
    if scale < 1.0:
        preview = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    preview_area = preview.shape[0] * preview.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area_ratio = cv2.contourArea(contour) / max(1, preview_area)
        if area_ratio < 0.18:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon.reshape(4, 2).astype(np.float32) / scale
        warped = _four_point_warp(cv2, image, points)
        if warped is None:
            continue
        # Emirates IDs are landscape.  This also makes portrait phone captures and
        # sideways scans consistent before OCR and portrait detection.
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped, "card_perspective_corrected"

    angle = _dominant_skew_angle(cv2, edges)
    if angle is None or abs(angle) < 0.7:
        return image, None
    return _rotate_bound(cv2, image, angle), "document_deskewed"


def _four_point_warp(cv2, image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right
    ordered[1] = points[np.argmin(differences)]  # top-right
    ordered[3] = points[np.argmax(differences)]  # bottom-left

    top_left, top_right, bottom_right, bottom_left = ordered
    width = int(round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))))
    height = int(round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))))
    if width < 160 or height < 100:
        return None

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _dominant_skew_angle(cv2, edges: np.ndarray) -> float | None:
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(60, min(edges.shape) // 4),
        maxLineGap=25,
    )
    if lines is None:
        return None
    angles: list[float] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        while angle <= -45:
            angle += 90
        while angle > 45:
            angle -= 90
        if abs(angle) <= 15:
            angles.append(angle)
    return float(np.median(angles)) if angles else None


def _rotate_bound(cv2, image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    output_width = int(height * sine + width * cosine)
    output_height = int(height * cosine + width * sine)
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _normalize_illumination(cv2, image: np.ndarray) -> np.ndarray:
    """Reduce shadows/highlight dominance without fabricating missing glare pixels."""

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=19, sigmaY=19)
    corrected = cv2.divide(lightness, np.maximum(background, 1), scale=185)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    # Blend with the source to keep portrait colour/texture natural for face models.
    lightness = cv2.addWeighted(lightness, 0.45, enhanced, 0.55, 0)
    return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2RGB)
