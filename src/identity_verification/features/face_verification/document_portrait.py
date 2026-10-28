from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class PortraitCandidate:
    """One deterministic document crop submitted to the preserved face pipeline.

    This adapter does not perform face recognition and does not replace any of the
    teammate's detection/quality/AuraFace logic. It only removes page margins and
    presents a few deterministic ID-card regions at a larger effective scale when a
    full rendered PDF page is too large for the original detector to find the small
    printed portrait.
    """

    image_rgb: np.ndarray
    strategy: str


def document_portrait_candidates(image_rgb: np.ndarray) -> list[PortraitCandidate]:
    """Return deterministic fallback crops after full-page analysis has failed.

    Candidate order is fixed and independent of similarity. The caller stops at the
    first crop that produces one usable face, so this is *not* score shopping.
    """

    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        return []

    base = _trim_uniform_page_margin(image_rgb[:, :, :3])
    height, width = base.shape[:2]
    if height < 80 or width < 80:
        return []

    candidates: list[PortraitCandidate] = []
    if base.shape[:2] != image_rgb.shape[:2]:
        candidates.append(PortraitCandidate(base, "content_trim"))

    # Emirates-ID front designs place the portrait toward a side of the card. Cropping
    # a side region makes that portrait substantially larger at the preserved YOLO
    # detector's fixed input size. Both sides are supported so the adapter is not tied
    # to one card revision or scan orientation.
    side_width = max(1, int(width * 0.62))
    candidates.extend(
        [
            PortraitCandidate(base[:, :side_width], "left_62"),
            PortraitCandidate(base[:, width - side_width :], "right_62"),
        ]
    )

    # A second, tighter deterministic pass handles PDFs with a large amount of card
    # context while still keeping enough surrounding area for the teammate's crop and
    # AuraFace alignment logic.
    top = int(height * 0.05)
    bottom = max(top + 1, int(height * 0.95))
    tight_width = max(1, int(width * 0.52))
    candidates.extend(
        [
            PortraitCandidate(base[top:bottom, :tight_width], "left_52"),
            PortraitCandidate(base[top:bottom, width - tight_width :], "right_52"),
        ]
    )

    return [candidate for candidate in candidates if candidate.image_rgb.size]


def _trim_uniform_page_margin(image_rgb: np.ndarray) -> np.ndarray:
    """Trim near-uniform PDF/page background while retaining a small safety border."""

    height, width = image_rgb.shape[:2]
    if height < 20 or width < 20:
        return image_rgb

    corner = max(2, min(height, width) // 40)
    samples = np.concatenate(
        [
            image_rgb[:corner, :corner].reshape(-1, 3),
            image_rgb[:corner, -corner:].reshape(-1, 3),
            image_rgb[-corner:, :corner].reshape(-1, 3),
            image_rgb[-corner:, -corner:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.int16)
    background = np.median(samples, axis=0)
    difference = np.abs(image_rgb.astype(np.int16) - background).max(axis=2)
    mask = difference > 18

    ys, xs = np.where(mask)
    if not len(xs) or not len(ys):
        return image_rgb

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    # Ignore tiny trims; they only add work without changing detector scale.
    content_area = max(1, (x2 - x1) * (y2 - y1))
    page_area = height * width
    if content_area / page_area > 0.92:
        return image_rgb

    pad_x = max(4, int((x2 - x1) * 0.03))
    pad_y = max(4, int((y2 - y1) * 0.03))
    x1 = max(0, x1 - pad_x)
    x2 = min(width, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(height, y2 + pad_y)
    return image_rgb[y1:y2, x1:x2]
