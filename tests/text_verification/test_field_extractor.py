from identity_verification.features.text_verification.domain.models import OCRLine
from identity_verification.features.text_verification.extraction.field_extractor import (
    VisibleFieldExtractor,
)


def test_sex_can_appear_before_label_in_paddle_reading_order() -> None:
    """PaddleOCR may return the short value before the visible Sex label."""

    lines = [
        OCRLine(text="Expiry Date تاريخ الانتهاء", confidence=0.95),
        OCRLine(text="M", confidence=0.99),
        OCRLine(text="Sex", confidence=0.98),
        OCRLine(text="Signature التوقيع", confidence=0.94),
    ]

    record = VisibleFieldExtractor().extract(
        lines,
        source="test",
    )

    assert record.sex.raw == "M"
    assert record.sex.normalized == "M"
    assert record.sex.confidence == 0.99


def test_arabic_name_can_appear_before_english_name_without_arabic_label() -> None:
    """Mixed RTL/LTR OCR may omit the Arabic name label but retain the value."""

    lines = [
        OCRLine(
            text="784-2004-5829832-8",
            confidence=0.99,
        ),
        OCRLine(
            text="احمد زهير احمد",
            confidence=0.96,
        ),
        OCRLine(
            text="Name: Samir Example Person",
            confidence=0.98,
        ),
        OCRLine(
            text="Date of Birth",
            confidence=0.97,
        ),
    ]

    record = VisibleFieldExtractor().extract(
        lines,
        source="test",
    )

    assert record.name_en.normalized == "SAMIR EXAMPLE PERSON"

    assert record.name_ar.raw == "احمد زهير احمد"
    assert record.name_ar.normalized == "احمد زهير احمد"
    assert record.name_ar.confidence == 0.96


def test_real_paddle_order_extracts_sex_and_arabic_name() -> None:
    """Regression test based on the observed Emirates ID PaddleOCR ordering."""

    lines = [
        OCRLine(text="رقم الهوية", confidence=0.99),
        OCRLine(
            text="784-2004-5829832-8",
            confidence=0.99,
        ),
        OCRLine(
            text="احمد زهير احمد",
            confidence=0.96,
        ),
        OCRLine(
            text="Name: Samir Example Person",
            confidence=0.98,
        ),
        OCRLine(
            text="Date of Birth :",
            confidence=0.98,
        ),
        OCRLine(
            text="11/12/2004",
            confidence=0.99,
        ),
        OCRLine(
            text="تاريخ الميلاد",
            confidence=0.95,
        ),
        OCRLine(
            text="الجنسية الجمهورية العربية السورية",
            confidence=0.92,
        ),
        OCRLine(
            text="Nationality: Syrian Arab Republic",
            confidence=0.98,
        ),
        OCRLine(
            text="تاريخ الاصدار",
            confidence=0.90,
        ),
        OCRLine(
            text="زكر",
            confidence=0.82,
        ),
        OCRLine(
            text="19/10/2021",
            confidence=0.99,
        ),
        OCRLine(
            text="Expiry Date تاريخ الانتهاء",
            confidence=0.93,
        ),
        OCRLine(
            text="M",
            confidence=0.99,
        ),
        OCRLine(
            text="Sex",
            confidence=0.97,
        ),
        OCRLine(
            text="Signature التوقيع",
            confidence=0.93,
        ),
    ]

    record = VisibleFieldExtractor().extract(
        lines,
        source="test",
    )

    assert record.id_number.normalized == "784200458298328"

    assert record.name_en.normalized == "SAMIR EXAMPLE PERSON"

    assert record.name_ar.normalized == "احمد زهير احمد"

    assert record.date_of_birth.normalized == "2004-12-11"

    assert record.nationality.normalized == "SYR"

    assert record.sex.normalized == "M"
