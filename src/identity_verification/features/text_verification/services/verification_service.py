import logging
from collections.abc import Callable
from datetime import date
from time import perf_counter
from uuid import uuid4

from identity_verification.features.text_verification.domain.models import (
    VerificationResult,
    VerificationStatus,
)
from identity_verification.features.text_verification.input.loader import DocumentInput
from identity_verification.features.text_verification.matching.matcher import compare_records
from identity_verification.features.text_verification.services.document_service import (
    DocumentService,
)

logger = logging.getLogger(__name__)


class VerificationService:
    """Coordinate extraction, validation and old/current identity comparison.

    The operational decision deliberately does not use a hand-weighted overall identity
    score. At this POC stage there is no labelled genuine/impostor dataset from which to
    fit those weights or operating thresholds. Instead, the service keeps the evidence
    explicit:

    1. OCR extracts fields and preserves the OCR model's own confidence per field.
    2. MRZ check digits validate machine-readable data where that validation exists.
    3. Old/current identity fields are compared with field-specific deterministic rules.
    4. Unresolved extraction/validation issues are routed to review.

    A calibrated continuous score can be reintroduced later without changing this flow.
    """

    def __init__(
        self,
        document_service: DocumentService,
        *,
        today_provider: Callable[[], date] = date.today,
    ) -> None:
        self.document_service = document_service
        self.today_provider = today_provider

    def verify(
        self,
        historical_document: DocumentInput,
        current_document: DocumentInput,
    ) -> VerificationResult:
        case_id = str(uuid4())
        started = perf_counter()

        # Never log OCR text or identity fields. Case IDs, timings and model names are
        # enough for operational troubleshooting without leaking document PII.
        logger.info(
            "verification_started",
            extra={
                "event": "verification_started",
                "case_id": case_id,
                "historical_files": len(historical_document.files),
                "current_files": len(current_document.files),
                "ocr_engine": self.document_service.ocr_engine.name,
            },
        )

        historical = self.document_service.extract(historical_document)
        current = self.document_service.extract(current_document)

        matching_started = perf_counter()
        status, comparisons, reasons = compare_records(historical.record, current.record)
        matching_ms = _elapsed_ms(matching_started)

        status = self._apply_document_validation(status, historical, "historical", reasons)
        status = self._apply_document_validation(status, current, "current", reasons)
        status = self._apply_current_expiry_validation(status, current, reasons)

        total_ms = _elapsed_ms(started)
        result = VerificationResult(
            case_id=case_id,
            status=status,
            comparisons=comparisons,
            reasons=list(dict.fromkeys(reasons)),
            historical=historical,
            current=current,
            timing_ms={
                "historical_document": historical.timing_ms.get("total", 0.0),
                "current_document": current.timing_ms.get("total", 0.0),
                "record_matching": round(matching_ms, 2),
                "total": round(total_ms, 2),
            },
        )

        logger.info(
            "verification_completed",
            extra={
                "event": "verification_completed",
                "case_id": case_id,
                "status": result.status,
                "duration_ms": round(total_ms, 2),
                "ocr_engine": self.document_service.ocr_engine.name,
            },
        )
        return result

    def _apply_document_validation(
        self,
        status: VerificationStatus,
        document,
        label: str,
        reasons: list[str],
    ) -> VerificationStatus:
        """Apply document-level validation without confusing OCR errors with identity mismatch."""

        if not document.mrz or not document.mrz.valid_structure:
            reasons.append(
                f"The {label} document's machine-readable zone could not be reliably "
                "reconstructed, so that document needs review."
            )
            return _promote_to_review(status)

        if document.mrz.checks.all_available_pass is False:
            reasons.append(
                f"The {label} document contains an MRZ extraction for which at least one "
                "check digit still fails after the available MRZ-reading path. This is an "
                "unresolved extraction/validation issue, not proof that the identity differs."
            )
            status = _promote_to_review(status)

        for key, value in document.internal_checks.items():
            if key in {"mrz_structure_valid", "mrz_check_digits_pass", "fields_sourced_from_mrz"}:
                continue
            if not isinstance(value, dict) or value.get("pass") is not False:
                continue

            # Supporting checks such as the MRZ-name comparison are deliberately visible
            # to the reviewer but do not independently force a review. The TD1 name line
            # does not carry its own check digit, and benign ordering/transliteration
            # differences are common enough that this should remain supporting evidence.
            if value.get("severity") == "supporting":
                reasons.append(
                    f"Supporting note for the {label} document: printed {_readable(key)} "
                    f"and MRZ representation differ ({value.get('visible')} vs "
                    f"{value.get('mrz')})."
                )
                continue

            reasons.append(
                f"The {label} document's validated printed {_readable(key)} does not match "
                f"its validated MRZ value ({value.get('visible')} vs {value.get('mrz')})."
            )
            status = _promote_to_review(status)
        return status

    def _apply_current_expiry_validation(
        self,
        status: VerificationStatus,
        current,
        reasons: list[str],
    ) -> VerificationStatus:
        value = current.record.expiry_date.normalized
        if not value:
            current.internal_checks["current_expiry_valid"] = None
            reasons.append("The current Emirates ID expiry date could not be extracted.")
            return _promote_to_review(status)

        try:
            expiry = date.fromisoformat(value)
        except ValueError:
            current.internal_checks["current_expiry_valid"] = False
            reasons.append("The current Emirates ID expiry date is not a valid calendar date.")
            return _promote_to_review(status)

        valid = expiry >= self.today_provider()
        current.internal_checks["current_expiry_valid"] = valid
        if not valid:
            reasons.append("The current Emirates ID appears to be expired.")
            return _promote_to_review(status)
        return status


def _readable(check_key: str) -> str:
    return check_key.removesuffix("_visible_vs_mrz").replace("_", " ")


def _promote_to_review(status: VerificationStatus) -> VerificationStatus:
    # Missing critical identity fields remain INSUFFICIENT_DATA; otherwise validation
    # concerns should route to review rather than silently changing into an auto-reject.
    if status == VerificationStatus.INSUFFICIENT_DATA:
        return status
    return VerificationStatus.REVIEW_REQUIRED


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000
