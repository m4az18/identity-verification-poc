from datetime import date
from io import BytesIO

from PIL import Image

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.extraction.nanonets_ocr2 import (
    NanonetsPageExtraction,
    parse_nanonets_output,
)
from identity_verification.features.text_verification.input.loader import DocumentInput, InputFile
from identity_verification.features.text_verification.services.nanonets_document_service import (
    NanonetsDocumentService,
)
from identity_verification.features.text_verification.services.verification_service import (
    VerificationService,
)

VALID_MRZ = [
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F3004157UTO<<<<<<<<<<<2",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
]


class FakeNanonetsExtractor:
    name = "nanonets-ocr2-3b"

    def __init__(self) -> None:
        self.calls = 0
        self.warm_calls = 0

    def warm_up(self) -> None:
        self.warm_calls += 1

    def extract_page(self, _image) -> NanonetsPageExtraction:
        self.calls += 1
        # Simulate a front page followed by a back page for each logical document.
        is_back = self.calls % 2 == 0
        return NanonetsPageExtraction(
            fields={
                "id_number": "784-1995-1234567-8",
                "name_en": "ANNA MARIA ERIKSSON",
                "name_ar": None,
                "date_of_birth": "12/08/1974",
                "nationality": "SYRIAN",
                "sex": "F",
                "expiry_date": "15/04/2030",
            },
            mrz_lines=VALID_MRZ if is_back else [],
            raw_output="{}",
        )


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


def test_nanonets_json_parser_keeps_structured_fields_and_mrz() -> None:
    output = """```json
    {
      "id_number": "784-1995-1234567-8",
      "name_en": "ANNA MARIA ERIKSSON",
      "name_ar": null,
      "date_of_birth": "12/08/1974",
      "nationality": "SYRIAN",
      "sex": "F",
      "expiry_date": "15/04/2030",
      "mrz_lines": [
        "I<UTOD231458907<<<<<<<<<<<<<<<",
        "7408122F3004157UTO<<<<<<<<<<<2",
        "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"
      ]
    }
    ```"""

    parsed = parse_nanonets_output(output)

    assert parsed.fields["id_number"] == "784-1995-1234567-8"
    assert parsed.fields["name_ar"] is None
    assert parsed.mrz_lines == VALID_MRZ
    assert parsed.warnings == []


def test_nanonets_invalid_output_is_not_silently_reinterpreted() -> None:
    parsed = parse_nanonets_output("not valid JSON")

    assert all(value is None for value in parsed.fields.values())
    assert parsed.mrz_lines == []
    assert parsed.warnings


def test_nanonets_document_backend_rejoins_common_validation_and_matching() -> None:
    extractor = FakeNanonetsExtractor()
    settings = Settings(debug_include_raw_text=False)
    service = VerificationService(
        NanonetsDocumentService(settings, extractor),
        today_provider=lambda: date(2026, 9, 16),
    )
    image = _png()

    result = service.verify(_document(image), _document(image))

    assert result.status == "consistent"
    assert extractor.calls == 4
    assert result.historical.ocr_engine == "nanonets-ocr2-3b"
    assert result.historical.record.id_number.normalized == "784199512345678"
    assert result.historical.record.date_of_birth.normalized == "1974-08-12"
    assert result.historical.mrz is not None
    assert result.historical.mrz.checks.all_available_pass is True
    assert result.current.internal_checks["current_expiry_valid"] is True
    assert result.historical.record.id_number.confidence is None
