"""Regression tests for the defects found in the v0.2 baseline run.

Each test here pins a specific bug that produced a wrong result on a real document, so
the behaviour cannot silently return. None of them require OCR models or real identity
documents; the MRZ fixtures are synthetic and carry correctly computed check digits.
"""

from datetime import date

from identity_verification.features.text_verification.domain.models import (
    ComparisonStatus,
    FieldValue,
    IdentityRecord,
    OCRLine,
)
from identity_verification.features.text_verification.extraction.field_extractor import (
    VisibleFieldExtractor,
)
from identity_verification.features.text_verification.matching.matcher import compare_records
from identity_verification.features.text_verification.mrz.extractor import MRZExtractor
from identity_verification.features.text_verification.mrz.parser import parse_td1
from identity_verification.features.text_verification.mrz.repair import repair_td1
from identity_verification.features.text_verification.normalization.values import (
    normalize_id_number,
    normalize_mrz_date,
    normalize_sex,
)
from identity_verification.features.text_verification.services.mrz_backfill import backfill_from_mrz

# Synthetic TD1 with valid ICAO check digits. Line 1 deliberately contains no filler,
# which is the real Emirates ID case: the 15-digit national number fills optional data 1
# exactly.
VALID_TD1 = [
    "ILARE1234567897784200412345678",
    "0412111M3112113SYR<<<<<<<<<<<1",
    "AHMAD<<ZUHAIR<AHMAD<<<<<<<<<<<",
]


def _boxed(texts: list[str]) -> list[OCRLine]:
    """Give each line a bbox so the extractor's vertical grouping has geometry to use."""

    return [
        OCRLine(
            text=text,
            bbox=[[0, index * 40], [400, index * 40], [400, index * 40 + 20], [0, index * 40 + 20]],
        )
        for index, text in enumerate(texts)
    ]


def test_mrz_line_without_filler_is_accepted() -> None:
    # v0.2 required "<" in every candidate line. TD1 line 1 on an Emirates ID has none,
    # so it was discarded, the window never reached three lines, and no MRZ could ever
    # be assembled regardless of OCR quality.
    extractor = MRZExtractor(ocr_engine=None, crop_top_ratio=0.5)

    result = extractor.parse_existing(ocr_lines=_boxed(VALID_TD1))

    assert result is not None
    assert result.valid_structure
    assert result.emirates_id_candidate == "784200412345678"


def test_repair_recovers_dropped_filler_runs_and_reports_edits() -> None:
    damaged = [
        "ILARE1234567897784200412345678",
        "0412111M3112113SYR1",  # the 11-character filler run was collapsed by OCR
        "AHMAD<<ZUHAIR<AHMAD",
    ]

    repaired = repair_td1(damaged)

    assert repaired is not None
    assert repaired.check_score >= 3
    assert repaired.repairs_applied > 0, "a reconstruction must never look like a clean read"
    assert parse_td1(repaired.lines).emirates_id_candidate == "784200412345678"


def test_repair_corrects_letter_digit_confusion() -> None:
    damaged = [
        VALID_TD1[0],
        "O412111M3II2113SYR<<<<<<<<<<<1",  # O for 0, I for 1 in numeric positions
        VALID_TD1[2],
    ]

    repaired = repair_td1(damaged)

    assert repaired is not None
    assert repaired.fully_verified
    assert parse_td1(repaired.lines).date_of_birth_raw == "041211"


def test_repair_refuses_non_mrz_text() -> None:
    # Repair must fail loudly rather than manufacture an MRZ out of unrelated OCR output.
    assert repair_td1(["QWERTY", "ASDF", "ZXCV"]) is None


def test_sex_is_not_populated_from_the_nationality_label() -> None:
    # "الجنس" (sex) is a strict prefix of "الجنسية" (nationality). Substring label
    # matching let the sex rule fire on the nationality line and store a country name.
    lines = _boxed(
        [
            "Name: SAMIR EXAMPLE PERSON",
            "الجنسية: الجمهورية العربية السورية",
            "Nationality: Syrian Arab Republic",
        ]
    )

    record = VisibleFieldExtractor().extract(lines, source="test")

    assert record.nationality.normalized == "SYR"
    assert record.sex.normalized is None
    assert record.sex.raw is None


def test_identical_unrecognised_values_do_not_compare_as_a_match() -> None:
    # Both documents are captured the same way, so OCR tends to make the same mistake on
    # both. In v0.2 the unresolved value survived normalization and two identical
    # mistakes compared equal, reporting a match on a field neither document supplied.
    assert normalize_sex("وي") is None

    def record_with_sex(raw: str) -> IdentityRecord:
        record = IdentityRecord()
        record.id_number = FieldValue(raw="784200412345678", normalized="784200412345678")
        record.date_of_birth = FieldValue(raw="2004-12-11", normalized="2004-12-11")
        record.name_en = FieldValue(raw="SAMIR EXAMPLE PERSON", normalized="SAMIR EXAMPLE PERSON")
        record.sex = FieldValue(raw=raw, normalized=normalize_sex(raw))
        return record

    _, comparisons, _ = compare_records(record_with_sex("وي"), record_with_sex("وي"))
    sex = next(item for item in comparisons if item.field == "sex")

    assert sex.status == ComparisonStatus.MISSING


def test_backfill_fills_only_checksum_protected_gaps_from_mrz() -> None:
    # Only MRZ fields protected by the relevant checksum are promoted into the canonical
    # record. Name, nationality and sex remain separate supporting MRZ evidence.
    mrz = parse_td1(VALID_TD1)
    mrz.extraction_method = "existing_text"
    record = IdentityRecord()

    filled = backfill_from_mrz(record, mrz, today=date(2026, 9, 11))

    assert set(filled) == {"id_number", "date_of_birth", "expiry_date"}
    assert record.id_number.normalized == "784200412345678"
    assert record.date_of_birth.normalized == "2004-12-11"
    assert record.expiry_date.normalized == "2031-12-11"
    assert record.sex.normalized is None
    assert record.nationality.normalized is None
    assert record.name_en.normalized is None
    assert record.id_number.sources == ["mrz:existing_text"]


def test_backfill_never_overwrites_a_visible_value() -> None:
    # A silent overwrite would hide the visible/MRZ disagreement that the cross-check
    # exists to surface.
    mrz = parse_td1(VALID_TD1)
    record = IdentityRecord()
    record.id_number = FieldValue(raw="784200499999999", normalized="784200499999999")

    filled = backfill_from_mrz(record, mrz, today=date(2026, 9, 11))

    assert "id_number" not in filled
    assert record.id_number.normalized == "784200499999999"


def test_mrz_date_century_window_is_resolved_against_today() -> None:
    # ICAO carries no century, so "31" must expand forward for an expiry and "04"
    # backward for a birth date.
    reference = date(2026, 9, 11)
    assert normalize_mrz_date("311211", reference=reference) == "2031-12-11"
    assert normalize_mrz_date("041211", reference=reference) == "2004-12-11"
    assert normalize_mrz_date("999999", reference=reference) is None


def test_id_number_must_be_structurally_valid() -> None:
    assert normalize_id_number("784-2004-1234567-8") == "784200412345678"
    assert normalize_id_number("784200412345") is None, "wrong length"
    assert normalize_id_number("123200412345678") is None, "wrong issuer prefix"


def test_startup_warmup_is_called_and_survives_failure(monkeypatch) -> None:
    """The warmup must run at startup, and must not stop the service if it fails.

    A model that cannot be preloaded is a degraded start, not a broken one: the lazy path
    still exists and the first request will load it.
    """

    from fastapi.testclient import TestClient

    import identity_verification.api.main as main

    monkeypatch.setattr(main.get_settings(), "warm_up_ocr_on_startup", True)

    calls: list[str] = []

    class _Engine:
        name = "stub"

        def warm_up(self) -> None:
            calls.append("warmed")

        def read(self, image_rgb, *, allowlist=None):  # pragma: no cover
            return []

    class _Failing(_Engine):
        def warm_up(self) -> None:
            calls.append("attempted")
            raise RuntimeError("weights unavailable")

    for engine, expected in ((_Engine(), "warmed"), (_Failing(), "attempted")):
        calls.clear()
        from identity_verification.features.text_verification.dependencies import (
            get_verification_service,
        )

        service = get_verification_service()
        original = service.document_service.ocr_engine
        service.document_service.ocr_engine = engine
        try:
            with TestClient(main.app) as client:
                assert client.get("/api/v1/text/health").status_code == 200
        finally:
            service.document_service.ocr_engine = original
        assert calls == [expected]


def test_mrz_falls_back_to_the_general_engine_without_tesseract() -> None:
    """Tesseract is preferred for the MRZ but must not be required to run the pipeline."""

    from identity_verification.features.text_verification.mrz.extractor import MRZExtractor

    class _Engine:
        name = "stub"

        def read(self, image_rgb, *, allowlist=None):
            return [
                OCRLine(
                    text=text,
                    bbox=[
                        [0, i * 40],
                        [400, i * 40],
                        [400, i * 40 + 20],
                        [0, i * 40 + 20],
                    ],
                )
                for i, text in enumerate(VALID_TD1)
            ]

    import numpy as np

    extractor = MRZExtractor(ocr_engine=_Engine(), crop_top_ratio=0.5)
    extractor.tesseract = type("_None", (), {"read_lines": lambda self, crop: []})()

    result = extractor.extract_with_fallback(np.zeros((200, 400, 3), dtype=np.uint8))

    assert result.valid_structure
    assert result.document_number == "123456789"
