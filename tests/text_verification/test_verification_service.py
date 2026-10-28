from datetime import date
from io import BytesIO

import numpy as np
from PIL import Image

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.domain.models import (
    OCRLine,
    VerificationStatus,
)
from identity_verification.features.text_verification.input.loader import DocumentInput, InputFile
from identity_verification.features.text_verification.services.document_service import (
    DocumentService,
)
from identity_verification.features.text_verification.services.verification_service import (
    VerificationService,
)


class FakeOCREngine:
    def __init__(self) -> None:
        self.normal_calls = 0
        self.mrz_calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def read(self, image_rgb: np.ndarray, *, allowlist: str | None = None) -> list[OCRLine]:
        if allowlist:
            self.mrz_calls += 1
            return [
                OCRLine(text="I<UTOD231458907<<<<<<<<<<<<<<<", bbox=_box(0)),
                OCRLine(text="7408122F3004157UTO<<<<<<<<<<<2", bbox=_box(30)),
                OCRLine(text="ERIKSSON<<ANNA<MARIA<<<<<<<<<<", bbox=_box(60)),
            ]

        self.normal_calls += 1
        return [
            OCRLine(text="ID Number"),
            OCRLine(text="784-1995-1234567-8", confidence=0.99),
            OCRLine(text="Name"),
            OCRLine(text="ANNA MARIA ERIKSSON", confidence=0.98),
            OCRLine(text="Date of Birth"),
            OCRLine(text="12/08/1974", confidence=0.98),
            OCRLine(text="Nationality"),
            OCRLine(text="UTO", confidence=0.98),
            OCRLine(text="Sex"),
            OCRLine(text="F", confidence=0.99),
            OCRLine(text="Expiry Date"),
            OCRLine(text="15/04/2030", confidence=0.98),
        ]


class ExistingMRZOCREngine(FakeOCREngine):
    def read(self, image_rgb: np.ndarray, *, allowlist: str | None = None) -> list[OCRLine]:
        if allowlist:
            self.mrz_calls += 1
            return []

        self.normal_calls += 1
        return [
            OCRLine(text="ID Number"),
            OCRLine(text="784-1995-1234567-8", confidence=0.99),
            OCRLine(text="Name"),
            OCRLine(text="ANNA MARIA ERIKSSON", confidence=0.98),
            OCRLine(text="Date of Birth"),
            OCRLine(text="12/08/1974", confidence=0.98),
            OCRLine(text="Nationality"),
            OCRLine(text="UTO", confidence=0.98),
            OCRLine(text="Sex"),
            OCRLine(text="F", confidence=0.99),
            OCRLine(text="Expiry Date"),
            OCRLine(text="15/04/2030", confidence=0.98),
            OCRLine(text="I<UTOD231458907<<<<<<<<<<<<<<<", bbox=_box(500)),
            OCRLine(text="7408122F3004157UTO<<<<<<<<<<<2", bbox=_box(530)),
            OCRLine(text="ERIKSSON<<ANNA<MARIA<<<<<<<<<<", bbox=_box(560)),
        ]


def _box(y: float) -> list[list[float]]:
    return [[0, y], [300, y], [300, y + 10], [0, y + 10]]


def _png() -> bytes:
    image = Image.new("RGB", (900, 600), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _document(image: bytes) -> DocumentInput:
    return DocumentInput(
        files=(
            InputFile("front.png", image),
            InputFile("back.png", image),
        )
    )


def test_full_service_pipeline_with_mock_ocr() -> None:
    engine = FakeOCREngine()
    service = VerificationService(
        DocumentService(Settings(), engine),
        today_provider=lambda: date(2026, 9, 11),
    )
    image = _png()

    result = service.verify(_document(image), _document(image))

    assert result.status == VerificationStatus.CONSISTENT
    assert result.historical.mrz is not None
    assert result.historical.mrz.checks.all_available_pass is True
    assert result.current.record.id_number.normalized == "784199512345678"
    assert result.current.internal_checks["current_expiry_valid"] is True
    assert result.timing_ms["total"] >= 0


def test_existing_full_page_mrz_avoids_specialized_second_pass() -> None:
    engine = ExistingMRZOCREngine()
    service = VerificationService(
        DocumentService(Settings(), engine),
        today_provider=lambda: date(2026, 9, 11),
    )
    image = _png()

    result = service.verify(_document(image), _document(image))

    assert result.historical.mrz is not None
    assert result.historical.mrz.extraction_method == "existing_text"
    assert engine.mrz_calls == 0


def test_expired_current_document_requires_review() -> None:
    engine = FakeOCREngine()
    service = VerificationService(
        DocumentService(Settings(), engine),
        today_provider=lambda: date(2031, 1, 1),
    )
    image = _png()

    result = service.verify(_document(image), _document(image))

    assert result.status == VerificationStatus.REVIEW_REQUIRED
    assert result.current.internal_checks["current_expiry_valid"] is False
    assert any("expired" in reason.lower() for reason in result.reasons)
