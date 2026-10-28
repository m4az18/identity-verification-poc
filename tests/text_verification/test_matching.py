from identity_verification.features.text_verification.domain.models import (
    FieldValue,
    IdentityRecord,
    VerificationStatus,
)
from identity_verification.features.text_verification.matching.matcher import compare_records


def value(text: str | None) -> FieldValue:
    return FieldValue(raw=text, normalized=text)


def test_consistent_exact_core_record() -> None:
    historical = IdentityRecord(
        id_number=value("784199512345678"),
        date_of_birth=value("1995-02-13"),
        name_en=value("MOHAMMED AL MANSOORI"),
        nationality=value("UAE"),
        sex=value("M"),
        expiry_date=value("2021-01-01"),
    )
    current = IdentityRecord(
        id_number=value("784199512345678"),
        date_of_birth=value("1995-02-13"),
        name_en=value("MOHAMMED AL MANSOORI"),
        nationality=value("UAE"),
        sex=value("M"),
        expiry_date=value("2031-01-01"),
    )

    status, comparisons, _ = compare_records(historical, current)
    assert status == VerificationStatus.CONSISTENT
    expiry = next(item for item in comparisons if item.field == "expiry_date")
    assert expiry.status == "not_compared"


def test_name_variation_requires_review_without_calibrated_threshold() -> None:
    historical = IdentityRecord(
        id_number=value("784199512345678"),
        date_of_birth=value("1995-02-13"),
        name_en=value("MOHAMMAD AHMED AL MANSOORI"),
    )
    current = IdentityRecord(
        id_number=value("784199512345678"),
        date_of_birth=value("1995-02-13"),
        name_en=value("MOHAMMED AHMAD ALMANSOORI"),
    )

    status, comparisons, _ = compare_records(historical, current)
    name = next(item for item in comparisons if item.field == "name_en")
    assert status == VerificationStatus.REVIEW_REQUIRED
    assert name.similarity is not None


def test_critical_mismatch_requires_review_not_automatic_rejection() -> None:
    historical = IdentityRecord(
        id_number=value("784199512345678"),
        date_of_birth=value("1995-02-13"),
    )
    current = IdentityRecord(
        id_number=value("784199512345679"),
        date_of_birth=value("1995-02-13"),
    )

    status, _, _ = compare_records(historical, current)
    assert status == VerificationStatus.REVIEW_REQUIRED
