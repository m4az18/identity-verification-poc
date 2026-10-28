from unittest.mock import Mock

import numpy as np
import pytest

from identity_verification.features.face_verification.face_preprocessing import (
    align_face_crop,
    analyze_standardized_face,
)

pytest.importorskip("cv2")


def test_detector_receives_uploaded_orientation_without_rotation_or_mirroring():
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape((3, 4, 3))
    verification = Mock()
    verification.analyze_image.return_value = (None, object())

    result = analyze_standardized_face(verification, image)

    verification.analyze_image.assert_called_once()
    assert np.array_equal(verification.analyze_image.call_args.args[0], image)
    assert not result.preview_steps


def test_landmark_postprocessor_outputs_standard_crop_and_corrects_flip():
    crop = np.full((100, 120, 3), 160, dtype=np.uint8)
    # Coordinates include the padding added by the preserved AuraFace helper. Eye
    # ordering is reversed to exercise deterministic horizontal-flip correction.
    landmarks = np.array(
        [[145, 102], [95, 102], [120, 125], [100, 151], [140, 151]],
        dtype=np.float32,
    )
    aligned, mirrored = align_face_crop(crop, landmarks)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8
    assert mirrored is True
