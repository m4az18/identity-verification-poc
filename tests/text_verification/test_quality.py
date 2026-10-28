import numpy as np

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.quality.metrics import analyze_image


def test_quality_metrics_are_advisory_and_include_blur_signal() -> None:
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    diagnostics = analyze_image(image, Settings())

    assert diagnostics.width == 400
    assert diagnostics.height == 300
    assert diagnostics.blur_variance == 0.0
    assert any("resolution" in warning.lower() for warning in diagnostics.warnings)
    assert any("blurred" in warning.lower() for warning in diagnostics.warnings)
