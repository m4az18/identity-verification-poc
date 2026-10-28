from identity_verification.features.text_verification.normalization.values import (
    comparison_key_arabic_name,
    comparison_key_english_name,
    normalize_arabic_name,
    normalize_date,
    normalize_id_number,
    normalize_nationality,
    normalize_sex,
)


def test_id_number_normalization() -> None:
    assert normalize_id_number("784-1995-1234567-8") == "784199512345678"


def test_date_normalization() -> None:
    assert normalize_date("13/02/1995") == "1995-02-13"
    assert normalize_date("1995-02-13") == "1995-02-13"


def test_english_name_key_ignores_formatting_only() -> None:
    assert comparison_key_english_name("AL MANSOORI") == comparison_key_english_name("AL-MANSOORI")


def test_arabic_name_normalization_is_conservative() -> None:
    assert normalize_arabic_name("أحمد") == "احمد"
    assert comparison_key_arabic_name("أحمد علي") == "احمدعلي"


def test_nationality_normalizes_to_iso_3166_alpha_3() -> None:
    # Alpha-3 is the interchange format because the MRZ nationality field already uses
    # it. Normalizing the visible zone into the same space is what makes the
    # visible/MRZ cross-check meaningful instead of a guaranteed mismatch.
    assert normalize_nationality("United Arab Emirates") == "ARE"
    assert normalize_nationality("ARE") == "ARE"
    assert normalize_nationality("Syrian Arab Republic") == "SYR"
    assert normalize_nationality("الجمهورية العربية السورية") == "SYR"


def test_sex_normalization() -> None:
    assert normalize_sex("Male") == "M"
    assert normalize_sex("أنثى") == "F"


def test_arabic_syrian_demonym_normalizes() -> None:
    assert normalize_nationality("سوري") == "SYR"


def test_bilingual_sex_forms_normalize() -> None:
    assert normalize_sex("M/ذكر") == "M"
    assert normalize_sex("F/أنثى") == "F"
