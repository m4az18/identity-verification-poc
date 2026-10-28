"""Backfill only checksum-protected MRZ values into missing canonical fields.

The MRZ and the visible zone are independent readings of the same document. The MRZ
check digits are valuable because they can detect many OCR errors, but they are not an
authenticity mechanism and they do not protect every MRZ field.

For automatic canonical backfill this module therefore stays conservative:

* Emirates ID candidate -> only when the TD1 composite check passes, because the
  15-digit Emirates ID candidate is located in optional data covered by that composite.
* Date of birth -> only when the DOB check digit passes.
* Expiry date -> only when the expiry check digit passes.

Name, nationality and sex remain separate MRZ evidence. They can be displayed to a
reviewer, but they are not silently promoted into the canonical identity record because
those values do not have equivalent dedicated check-digit protection in this POC.

Backfill only fills a missing visible field. It never overwrites a visible value; any
visible-versus-MRZ disagreement is surfaced separately by the document cross-check.
"""

from __future__ import annotations

from datetime import date

from identity_verification.features.text_verification.domain.models import (
    FieldValue,
    IdentityRecord,
    MRZData,
)
from identity_verification.features.text_verification.normalization.values import (
    normalize_id_number,
    normalize_mrz_date,
)


def backfill_from_mrz(
    record: IdentityRecord,
    mrz: MRZData | None,
    *,
    today: date | None = None,
) -> list[str]:
    """Fill missing checksum-protected fields from MRZ evidence.

    The returned field names are kept as provenance in the response so the UI/JSON can
    show when a canonical value came from the MRZ rather than from visible-field OCR.
    """

    if mrz is None or not mrz.valid_structure:
        return []

    filled: list[str] = []

    # Emirates ID candidate lives in TD1 optional data; the composite check is the
    # relevant integrity check for that region.
    if (
        mrz.checks.composite is True
        and mrz.emirates_id_candidate
        and _fill(
            record,
            "id_number",
            mrz.emirates_id_candidate,
            normalize_id_number,
            mrz,
        )
    ):
        filled.append("id_number")

    if mrz.checks.date_of_birth is True and mrz.date_of_birth_raw:
        iso = normalize_mrz_date(mrz.date_of_birth_raw, reference=today)
        if iso and _fill(record, "date_of_birth", iso, lambda value: value, mrz):
            filled.append("date_of_birth")

    if mrz.checks.expiry_date is True and mrz.expiry_date_raw:
        iso = normalize_mrz_date(mrz.expiry_date_raw, reference=today)
        if iso and _fill(record, "expiry_date", iso, lambda value: value, mrz):
            filled.append("expiry_date")

    return filled


def _fill(record: IdentityRecord, field: str, raw: str, normalizer, mrz: MRZData) -> bool:
    existing: FieldValue = getattr(record, field)
    if existing.normalized:
        return False

    normalized = normalizer(raw)
    if not normalized:
        return False

    setattr(
        record,
        field,
        FieldValue(
            raw=raw,
            normalized=normalized,
            # This is not an OCR confidence score. The value is accepted because the
            # relevant MRZ checksum passed, so confidence remains None.
            confidence=None,
            sources=[f"mrz:{mrz.extraction_method or 'unknown'}"],
        ),
    )
    return True


__all__ = ["backfill_from_mrz"]
