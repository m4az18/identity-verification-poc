"""Resolution control and MRZ crop enhancement before OCR.

The pipeline previously handed OCR whatever pixels the loader produced. For an A4 page
rendered at 220 DPI that is roughly 1,870 x 2,640, of which an Emirates ID occupies a
small fraction. Detection cost in EasyOCR scales with pixel count, so most of the
measured per-document time was spent scanning blank page. Capping the long edge is the
single change with the largest effect on latency, and it costs no accuracy: the card's
printed text is already far above the resolution any recogniser needs at 1,600 px.

The MRZ is the opposite case and needs the resolution increased, not reduced. OCR-B at
crop scale sits near the lower bound of what a scene-text recogniser resolves, so the
crop is upscaled and contrast-normalised before its dedicated pass.

Both functions are pure and take no configuration beyond their arguments, so the same
transforms can be applied inside an evaluation harness without starting the API.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_MAX_EDGE = 1600
MRZ_TARGET_HEIGHT = 320
MRZ_MAX_UPSCALE = 4.0


def cap_resolution(image_rgb: np.ndarray, max_edge: int = DEFAULT_MAX_EDGE) -> np.ndarray:
    """Downscale so the longer edge is at most `max_edge`, preserving aspect ratio.

    Images already within the bound are returned unchanged rather than round-tripped
    through PIL, so this is free for camera captures that are already reasonable.
    """

    height, width = image_rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image_rgb

    scale = max_edge / longest
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    # LANCZOS preserves the thin strokes in printed ID text better than bilinear, which
    # matters because the downscale is the last chance to keep a stroke readable.
    return np.asarray(Image.fromarray(image_rgb).resize(target, Image.LANCZOS))


def enhance_mrz_crop(crop_rgb: np.ndarray) -> np.ndarray:
    """Upscale and contrast-normalise an MRZ crop for its dedicated OCR pass.

    Returned as three-channel RGB because the OCR adapters expect it, even though the
    content is grayscale after normalisation.
    """

    if crop_rgb.size == 0:
        return crop_rgb

    height = crop_rgb.shape[0]
    if height <= 0:
        return crop_rgb

    scale = min(MRZ_MAX_UPSCALE, max(1.0, MRZ_TARGET_HEIGHT / height))
    working = crop_rgb
    if scale > 1.0:
        target = (
            max(1, int(round(crop_rgb.shape[1] * scale))),
            max(1, int(round(height * scale))),
        )
        working = np.asarray(Image.fromarray(crop_rgb).resize(target, Image.LANCZOS))

    grayscale = np.dot(working[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    # Percentile stretch rather than min/max: a single dark speck or glare highlight
    # would otherwise anchor the range and leave the actual text low-contrast.
    low, high = np.percentile(grayscale, (2.0, 98.0))
    stretched = grayscale if high - low < 1e-3 else np.clip((grayscale - low) * (255.0 / (high - low)), 0, 255)

    channel = stretched.astype(np.uint8)
    return np.stack([channel] * 3, axis=-1)
