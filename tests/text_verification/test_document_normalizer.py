import numpy as np
import pytest

from identity_verification.features.text_verification.input.document_normalizer import (
    normalize_document_image,
)

cv2 = pytest.importorskip("cv2")


def test_document_normalizer_extracts_and_straightens_card_boundary():
    image = np.full((700, 1000, 3), 245, dtype=np.uint8)
    card = np.array([[150, 155], [865, 100], [900, 555], [105, 610]], dtype=np.int32)
    cv2.fillConvexPoly(image, card, (80, 120, 150))
    cv2.polylines(image, [card], True, (10, 10, 10), 10)

    result = normalize_document_image(image)

    assert result.image_rgb.shape[1] > result.image_rgb.shape[0]
    assert "card_perspective_corrected" in result.operations
    assert "illumination_contrast_normalized" in result.operations


def test_document_normalizer_keeps_rgb_uint8_contract():
    image = np.full((120, 200, 4), 128.0, dtype=np.float32)
    result = normalize_document_image(image)
    assert result.image_rgb.dtype == np.uint8
    assert result.image_rgb.shape[2] == 3
