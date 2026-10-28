import numpy as np

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.domain.models import ImageDiagnostics


def analyze_image(image_rgb: np.ndarray, settings: Settings) -> ImageDiagnostics:
    """Return lightweight, non-blocking image-quality signals.

    The values are advisory in this POC. They help explain OCR failures and identify
    samples that may need recapture, but they do not automatically reject an ID until
    thresholds are calibrated on representative evaluation data.
    """

    height, width = image_rgb.shape[:2]
    grayscale = np.dot(image_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    brightness = float(grayscale.mean())

    # Laplacian variance is a cheap focus/sharpness proxy. A lower value usually means
    # less edge detail, but camera and document characteristics affect the scale, so the
    # warning threshold remains configurable rather than being treated as ground truth.
    laplacian = (
        -4 * grayscale[1:-1, 1:-1]
        + grayscale[:-2, 1:-1]
        + grayscale[2:, 1:-1]
        + grayscale[1:-1, :-2]
        + grayscale[1:-1, 2:]
    )
    blur_variance = float(laplacian.var()) if laplacian.size else 0.0

    warnings: list[str] = []
    if min(width, height) < settings.quality_min_dimension:
        warnings.append("Low image resolution may reduce OCR accuracy.")
    if brightness < settings.quality_dark_threshold:
        warnings.append("Image appears very dark.")
    elif brightness > settings.quality_bright_threshold:
        warnings.append("Image appears very bright; glare may affect OCR.")
    if blur_variance < settings.quality_blur_warning_threshold:
        warnings.append("Image may be blurred or lack sufficient edge detail.")

    return ImageDiagnostics(
        width=width,
        height=height,
        mean_brightness=round(brightness, 2),
        blur_variance=round(blur_variance, 2),
        warnings=warnings,
    )
