from identity_verification.features.text_verification.mrz.checksums import calculate_check_digit
from identity_verification.features.text_verification.mrz.parser import parse_td1


def test_official_style_td1_example_passes_check_digits() -> None:
    mrz = parse_td1(
        [
            "I<UTOD231458907<<<<<<<<<<<<<<<",
            "7408122F1204159UTO<<<<<<<<<<<6",
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
        ]
    )

    assert mrz.valid_structure is True
    assert mrz.document_number == "D23145890"
    assert mrz.date_of_birth_raw == "740812"
    assert mrz.expiry_date_raw == "120415"
    assert mrz.name == "ANNA MARIA ERIKSSON"
    assert mrz.checks.document_number is True
    assert mrz.checks.date_of_birth is True
    assert mrz.checks.expiry_date is True
    assert mrz.checks.composite is True


def test_known_icao_check_digit_example() -> None:
    assert calculate_check_digit("520727") == "3"


def test_td1_rejects_wrong_line_length() -> None:
    mrz = parse_td1(["ABC", "DEF", "GHI"])
    assert mrz.valid_structure is False
    assert "30 characters" in (mrz.error or "")


def test_parse_existing_records_mrz_attempt_history() -> None:
    from identity_verification.features.text_verification.mrz.extractor import MRZExtractor

    class _NoopOCR:
        name = "noop"

        def read(self, image_rgb, *, allowlist=None):  # noqa: ANN001
            return []

        def warm_up(self) -> None:
            return None

    extractor = MRZExtractor(_NoopOCR(), 0.52)
    mrz = extractor.parse_existing(
        text_layer=("I<UTOD231458907<<<<<<<<<<<<<<<\n7408122F1204159UTO<<<<<<<<<<<6\nERIKSSON<<ANNA<MARIA<<<<<<<<<<")
    )

    assert mrz is not None
    assert mrz.checks.all_available_pass is True
    assert len(mrz.attempts) == 1
    assert mrz.attempts[0].method == "existing_text"
    assert mrz.attempts[0].selected is True
    assert mrz.attempts[0].passing_checks == 4
