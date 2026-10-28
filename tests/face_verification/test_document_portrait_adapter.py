import numpy as np

from identity_verification.features.face_verification.document_portrait import (
    document_portrait_candidates,
)


def test_portrait_adapter_returns_deterministic_side_crops():
    image = np.full((1000, 1600, 3), 255, dtype=np.uint8)
    image[200:800, 250:1350] = 120
    candidates = document_portrait_candidates(image)
    strategies = [item.strategy for item in candidates]
    assert "content_trim" in strategies
    assert "left_62" in strategies
    assert "right_62" in strategies
    assert strategies.index("left_62") < strategies.index("right_62")
