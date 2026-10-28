from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from identity_verification.features.face_verification.domain.models import FaceVerificationStatus
from identity_verification.features.face_verification.service import FaceVerificationService


def _face(values):
    return SimpleNamespace(normed_embedding=np.asarray(values, dtype=float))


def test_adapter_uses_preserved_similarity_and_threshold_logic():
    verification = Mock()
    verification.analyze_image.side_effect = [
        (None, _face([1.0, 0.0])),  # live
        (None, _face([0.8, 0.6])),  # current ID
    ]
    verification.log_event = Mock()
    config = SimpleNamespace(get_threshold=lambda: 0.75)

    with patch(
        "identity_verification.features.face_verification.service.load_original_face_modules",
        return_value=(verification, config),
    ):
        result = FaceVerificationService().verify_images(
            [np.zeros((100, 100, 3), dtype=np.uint8)],
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    assert result.status == FaceVerificationStatus.MATCH
    assert result.similarity == 0.8
    assert result.threshold == 0.75
    assert result.document_page_used == 1


def test_adapter_falls_back_to_second_document_page_without_score_shopping():
    verification = Mock()
    verification.analyze_image.side_effect = [
        (None, _face([1.0, 0.0])),
        ValueError("Exactly one face is required in each image."),
        (None, _face([0.9, 0.1])),
    ]
    verification.log_event = Mock()
    config = SimpleNamespace(get_threshold=lambda: 0.4)

    with patch(
        "identity_verification.features.face_verification.service.load_original_face_modules",
        return_value=(verification, config),
    ):
        result = FaceVerificationService().verify_images(
            [np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8)],
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    assert result.status == FaceVerificationStatus.MATCH
    assert result.document_page_used == 2
    assert verification.analyze_image.call_count == 3


def test_adapter_uses_document_crop_only_after_full_pages_fail():
    from identity_verification.features.face_verification.document_portrait import PortraitCandidate

    verification = Mock()
    verification.analyze_image.side_effect = [
        (None, _face([1.0, 0.0])),  # live
        ValueError("Exactly one face is required in each image."),
        (None, _face([0.8, 0.6])),  # deterministic crop
    ]
    verification.log_event = Mock()
    config = SimpleNamespace(get_threshold=lambda: 0.4)

    with (
        patch(
            "identity_verification.features.face_verification.service.load_original_face_modules",
            return_value=(verification, config),
        ),
        patch(
            "identity_verification.features.face_verification.service.document_portrait_candidates",
            return_value=[PortraitCandidate(np.zeros((80, 80, 3), dtype=np.uint8), "left_62")],
        ),
    ):
        result = FaceVerificationService().verify_images(
            [np.zeros((100, 100, 3), dtype=np.uint8)],
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    assert result.status == FaceVerificationStatus.MATCH
    assert result.document_input_strategy == "left_62"
    assert "DOCUMENT_PORTRAIT_ADAPTER_USED" in result.reason_codes
