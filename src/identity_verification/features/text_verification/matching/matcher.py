from collections.abc import Callable

from rapidfuzz.fuzz import ratio

from identity_verification.features.text_verification.domain.models import (
    ComparisonStatus,
    FieldComparison,
    IdentityRecord,
    VerificationStatus,
)
from identity_verification.features.text_verification.normalization.values import (
    comparison_key_arabic_name,
    comparison_key_english_name,
)

CRITICAL_FIELDS = ("id_number", "date_of_birth")


def compare_records(
    historical: IdentityRecord,
    current: IdentityRecord,
) -> tuple[VerificationStatus, list[FieldComparison], list[str]]:
    """Compare canonical old/current fields using field-specific rules.

    No single fuzzy score decides identity. Critical fields are exact after
    normalization, while name similarity is evidence for review until a threshold is
    calibrated on a representative dataset.
    """

    comparisons = [
        _exact("id_number", historical.id_number.normalized, current.id_number.normalized),
        _exact(
            "date_of_birth",
            historical.date_of_birth.normalized,
            current.date_of_birth.normalized,
        ),
        _name(
            "name_en",
            historical.name_en.normalized,
            current.name_en.normalized,
            comparison_key_english_name,
        ),
        _name(
            "name_ar",
            historical.name_ar.normalized,
            current.name_ar.normalized,
            comparison_key_arabic_name,
        ),
        _exact(
            "nationality",
            historical.nationality.normalized,
            current.nationality.normalized,
        ),
        _exact("sex", historical.sex.normalized, current.sex.normalized),
        FieldComparison(
            field="expiry_date",
            historical=historical.expiry_date.normalized,
            current=current.expiry_date.normalized,
            status=ComparisonStatus.NOT_COMPARED,
            note=(
                "Expiry is document-specific and is expected to change after renewal; "
                "the current document is validated separately for expiry."
            ),
        ),
    ]

    reasons: list[str] = []
    critical = [item for item in comparisons if item.field in CRITICAL_FIELDS]
    if any(item.status == ComparisonStatus.REVIEW for item in critical):
        reasons.append("A critical identity field differs between the historical and current records.")
        return VerificationStatus.REVIEW_REQUIRED, comparisons, reasons

    if any(item.status == ComparisonStatus.MISSING for item in critical):
        reasons.append("At least one critical identity field could not be extracted from both documents.")
        return VerificationStatus.INSUFFICIENT_DATA, comparisons, reasons

    # At least one name representation should support the critical fields. Requiring
    # both languages would be too strict because OCR quality and document variants differ.
    name_comparisons = [item for item in comparisons if item.field in {"name_en", "name_ar"}]
    if all(item.status == ComparisonStatus.MISSING for item in name_comparisons):
        reasons.append("No comparable English or Arabic name was extracted from both documents.")
        return VerificationStatus.REVIEW_REQUIRED, comparisons, reasons

    if any(item.status in {ComparisonStatus.REVIEW, ComparisonStatus.VARIATION} for item in comparisons):
        reasons.append("Critical identity fields are consistent, but at least one supporting field needs review.")
        return VerificationStatus.REVIEW_REQUIRED, comparisons, reasons

    reasons.append("Available critical identity fields and supporting evidence are consistent.")
    return VerificationStatus.CONSISTENT, comparisons, reasons


def _exact(field: str, historical: str | None, current: str | None) -> FieldComparison:
    if not historical or not current:
        return FieldComparison(
            field=field,
            historical=historical,
            current=current,
            status=ComparisonStatus.MISSING,
        )
    if historical == current:
        return FieldComparison(
            field=field,
            historical=historical,
            current=current,
            status=ComparisonStatus.MATCH,
        )
    return FieldComparison(
        field=field,
        historical=historical,
        current=current,
        status=ComparisonStatus.REVIEW,
        note="Values differ after normalization.",
    )


def _name(
    field: str,
    historical: str | None,
    current: str | None,
    key_fn: Callable[[str | None], str | None],
) -> FieldComparison:
    if not historical or not current:
        return FieldComparison(
            field=field,
            historical=historical,
            current=current,
            status=ComparisonStatus.MISSING,
        )

    historical_key = key_fn(historical)
    current_key = key_fn(current)
    if historical_key == current_key:
        return FieldComparison(
            field=field,
            historical=historical,
            current=current,
            status=ComparisonStatus.MATCH,
            similarity=100.0,
        )

    similarity = round(float(ratio(historical_key or "", current_key or "")), 2)
    return FieldComparison(
        field=field,
        historical=historical,
        current=current,
        status=ComparisonStatus.VARIATION,
        similarity=similarity,
        note=(
            "Name differs after normalization. Similarity is reviewer evidence only; "
            "no automatic fuzzy threshold has been calibrated yet."
        ),
    )
