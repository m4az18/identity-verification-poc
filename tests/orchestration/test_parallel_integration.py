from io import BytesIO
from time import perf_counter, sleep
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.input.loader import (
    DocumentInput,
    DocumentLoader,
    InputFile,
)
from identity_verification.orchestration.service import UnifiedVerificationService


def _image_bytes() -> bytes:
    output = BytesIO()
    Image.fromarray(np.full((80, 120, 3), 120, dtype=np.uint8)).save(output, format="PNG")
    return output.getvalue()


def test_current_id_is_prepared_once_and_pipelines_overlap():
    loader = DocumentLoader(Settings(warm_up_ocr_on_startup=False))
    historical = DocumentInput((InputFile("old.png", _image_bytes()),))
    current = DocumentInput((InputFile("new.png", _image_bytes()),))
    face_photo = InputFile("selfie.png", _image_bytes())
    calls = []
    original_prepare = loader._prepare_document

    def prepare(document):
        calls.append(document)
        return original_prepare(document)

    loader._prepare_document = prepare
    spans = {}

    def verify_text(old, new):
        spans["text_start"] = perf_counter()
        assert old is historical and new is current
        loader.prepare_document(old)
        loader.prepare_document(new)
        sleep(0.1)
        spans["text_end"] = perf_counter()
        return SimpleNamespace(case_id="case", status="consistent")

    def verify_face(images, live):
        spans["face_start"] = perf_counter()
        assert len(images) == 1 and images[0].shape == (80, 120, 3)
        assert live.shape == (80, 120, 3)
        sleep(0.1)
        spans["face_end"] = perf_counter()
        return SimpleNamespace(status="match")

    text_service = SimpleNamespace(
        document_service=SimpleNamespace(loader=loader),
        verify=verify_text,
    )
    face_service = SimpleNamespace(verify_document_images=verify_face)
    policy = SimpleNamespace(decide=lambda _text, _face: SimpleNamespace(status="verified"))

    with (
        patch("identity_verification.orchestration.service.build_document_evidence_score", return_value=None),
        patch("identity_verification.orchestration.service.UnifiedVerificationResult", side_effect=lambda **kw: kw),
    ):
        result = UnifiedVerificationService(text_service, face_service, policy, parallel=True).verify(
            historical, current, face_photo
        )

    assert calls.count(current) == 1
    assert calls.count(historical) == 1
    assert spans["face_start"] < spans["text_end"]
    assert spans["text_start"] < spans["face_end"]
    assert result["timing_ms"]["pipelines_wall"] > 0
