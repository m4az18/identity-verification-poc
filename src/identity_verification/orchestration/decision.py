from __future__ import annotations

from identity_verification.features.face_verification.domain.models import (
    FaceReasonCode,
    FaceVerificationResult,
    FaceVerificationStatus,
)
from identity_verification.features.text_verification.domain.models import (
    ComparisonStatus,
    VerificationResult,
    VerificationStatus,
)
from identity_verification.orchestration.models import (
    Decision,
    DecisionReasonCode,
    OverallVerificationStatus,
    RecommendedAction,
)


class DecisionPolicy:
    """Deterministic evidence-fusion policy with hard gates and explicit reasons.

    The POC deliberately avoids turning OCR confidence, name similarity and face cosine
    similarity into one pseudo-probability. A separate document evidence index exists for
    diagnostics, but it never decides the case. The decision itself follows explicit
    safety/identity gates so a strong signal can never compensate for a critical mismatch.
    """

    VERSION = "poc-evidence-policy-v2"

    def decide(
        self,
        text_result: VerificationResult,
        face_result: FaceVerificationResult,
    ) -> Decision:
        reasons: list[DecisionReasonCode] = []
        actions: list[RecommendedAction] = []
        review_reasons: list[DecisionReasonCode] = []
        retry_reasons: list[DecisionReasonCode] = []

        self._collect_document_facts(
            text_result,
            reasons=reasons,
            actions=actions,
            review_reasons=review_reasons,
            retry_reasons=retry_reasons,
        )
        service_failure = self._collect_face_facts(
            face_result,
            reasons=reasons,
            actions=actions,
            review_reasons=review_reasons,
            retry_reasons=retry_reasons,
        )

        reasons = _dedupe(reasons)
        actions = _dedupe(actions)
        review_reasons = _dedupe(review_reasons)
        retry_reasons = _dedupe(retry_reasons)

        if service_failure:
            return Decision(
                status=OverallVerificationStatus.UNABLE_TO_VERIFY,
                title="Verification could not be completed",
                policy_version=self.VERSION,
                reason_codes=reasons,
                recommended_actions=_dedupe(actions + [RecommendedAction.RETRY_SERVICE]),
                explanation=(
                    "A required biometric service was unavailable. Any document findings are "
                    "preserved, but a complete automated identity decision cannot be made."
                ),
            )

        if review_reasons:
            title, explanation = _review_copy(review_reasons, retry_reasons)
            return Decision(
                status=OverallVerificationStatus.REVIEW_REQUIRED,
                title=title,
                policy_version=self.VERSION,
                reason_codes=reasons,
                recommended_actions=_dedupe(actions + [RecommendedAction.MANUAL_REVIEW]),
                explanation=explanation,
            )

        if retry_reasons:
            title, explanation = _retry_copy(retry_reasons)
            return Decision(
                status=OverallVerificationStatus.RETRY_REQUIRED,
                title=title,
                policy_version=self.VERSION,
                reason_codes=reasons,
                recommended_actions=actions,
                explanation=explanation,
            )

        if face_result.status == FaceVerificationStatus.MATCH:
            return Decision(
                status=OverallVerificationStatus.VERIFIED,
                title="Identity verified",
                policy_version=self.VERSION,
                reason_codes=[DecisionReasonCode.ALL_REQUIRED_EVIDENCE_PASSED],
                recommended_actions=[RecommendedAction.NONE],
                explanation=(
                    "The required document identity checks passed and the current photo matched "
                    "the portrait on the current Emirates ID at the configured POC threshold."
                ),
            )

        # Defensive future-proof fallback: unknown states never auto-verify.
        return Decision(
            status=OverallVerificationStatus.UNABLE_TO_VERIFY,
            title="Verification could not be completed",
            policy_version=self.VERSION,
            reason_codes=reasons,
            recommended_actions=_dedupe(actions + [RecommendedAction.MANUAL_REVIEW]),
            explanation="The evidence combination is not covered by the current POC policy.",
        )

    @staticmethod
    def _collect_document_facts(
        result: VerificationResult,
        *,
        reasons: list[DecisionReasonCode],
        actions: list[RecommendedAction],
        review_reasons: list[DecisionReasonCode],
        retry_reasons: list[DecisionReasonCode],
    ) -> None:
        review_before = len(review_reasons)
        retry_before = len(retry_reasons)

        critical = [item for item in result.comparisons if item.field in {"id_number", "date_of_birth"}]
        if any(item.status == ComparisonStatus.REVIEW for item in critical):
            _add_review(
                DecisionReasonCode.CRITICAL_IDENTITY_MISMATCH,
                reasons,
                review_reasons,
                actions,
                RecommendedAction.MANUAL_REVIEW,
            )
        if any(item.status == ComparisonStatus.MISSING for item in critical):
            _add_retry(
                DecisionReasonCode.DOCUMENT_REQUIRED_FIELDS_MISSING,
                reasons,
                retry_reasons,
                actions,
                RecommendedAction.REUPLOAD_DOCUMENTS,
            )

        for _label, document, unreadable_code, check_code, action in (
            (
                "historical",
                result.historical,
                DecisionReasonCode.HISTORICAL_MRZ_UNREADABLE,
                DecisionReasonCode.HISTORICAL_MRZ_CHECK_FAILED,
                RecommendedAction.REUPLOAD_HISTORICAL_DOCUMENT,
            ),
            (
                "current",
                result.current,
                DecisionReasonCode.CURRENT_MRZ_UNREADABLE,
                DecisionReasonCode.CURRENT_MRZ_CHECK_FAILED,
                RecommendedAction.REUPLOAD_CURRENT_DOCUMENT,
            ),
        ):
            mrz = document.mrz
            if mrz is None or not mrz.valid_structure or mrz.checks.all_available_pass is None:
                _add_retry(unreadable_code, reasons, retry_reasons, actions, action)
            elif mrz.checks.all_available_pass is False:
                # The dedicated MRZ fallback has already been attempted by the document
                # pipeline. Remaining checksum failure is review evidence, not a silent pass.
                _add_review(check_code, reasons, review_reasons, actions, action)

            for key, value in document.internal_checks.items():
                if not key.endswith("_visible_vs_mrz") or not isinstance(value, dict):
                    continue
                if value.get("severity") == "critical" and value.get("pass") is False:
                    _add_review(
                        DecisionReasonCode.DOCUMENT_INTERNAL_MISMATCH,
                        reasons,
                        review_reasons,
                        actions,
                        RecommendedAction.MANUAL_REVIEW,
                    )

        supporting = [item for item in result.comparisons if item.field in {"name_en", "name_ar", "nationality", "sex"}]
        name_items = [item for item in supporting if item.field in {"name_en", "name_ar"}]
        if name_items and all(item.status == ComparisonStatus.MISSING for item in name_items):
            _add_retry(
                DecisionReasonCode.DOCUMENT_NAME_EVIDENCE_MISSING,
                reasons,
                retry_reasons,
                actions,
                RecommendedAction.REUPLOAD_DOCUMENTS,
            )
        if any(item.status in {ComparisonStatus.REVIEW, ComparisonStatus.VARIATION} for item in supporting):
            _add_review(
                DecisionReasonCode.SUPPORTING_IDENTITY_VARIATION,
                reasons,
                review_reasons,
                actions,
                RecommendedAction.MANUAL_REVIEW,
            )

        expiry = result.current.internal_checks.get("current_expiry_valid")
        if expiry is False:
            _add_review(
                DecisionReasonCode.CURRENT_DOCUMENT_EXPIRED,
                reasons,
                review_reasons,
                actions,
                RecommendedAction.MANUAL_REVIEW,
            )
        elif expiry is None:
            _add_retry(
                DecisionReasonCode.CURRENT_DOCUMENT_EXPIRY_UNREADABLE,
                reasons,
                retry_reasons,
                actions,
                RecommendedAction.REUPLOAD_CURRENT_DOCUMENT,
            )

        # Never let a new/unknown text-pipeline review state silently become VERIFIED.
        # Specific structured reasons above remain preferred; this is a defensive fallback.
        if (
            result.status == VerificationStatus.REVIEW_REQUIRED
            and len(review_reasons) == review_before
            and len(retry_reasons) == retry_before
        ):
            _add_review(
                DecisionReasonCode.DOCUMENT_REVIEW_REQUIRED_OTHER,
                reasons,
                review_reasons,
                actions,
                RecommendedAction.MANUAL_REVIEW,
            )
        elif (
            result.status == VerificationStatus.INSUFFICIENT_DATA
            and len(retry_reasons) == retry_before
            and len(review_reasons) == review_before
        ):
            _add_retry(
                DecisionReasonCode.DOCUMENT_REQUIRED_FIELDS_MISSING,
                reasons,
                retry_reasons,
                actions,
                RecommendedAction.REUPLOAD_DOCUMENTS,
            )

    @staticmethod
    def _collect_face_facts(
        result: FaceVerificationResult,
        *,
        reasons: list[DecisionReasonCode],
        actions: list[RecommendedAction],
        review_reasons: list[DecisionReasonCode],
        retry_reasons: list[DecisionReasonCode],
    ) -> bool:
        if result.status == FaceVerificationStatus.UNAVAILABLE:
            reasons.append(DecisionReasonCode.FACE_SERVICE_UNAVAILABLE)
            actions.append(RecommendedAction.RETRY_SERVICE)
            return True

        if result.status == FaceVerificationStatus.NO_MATCH:
            _add_review(
                DecisionReasonCode.FACE_BELOW_THRESHOLD,
                reasons,
                review_reasons,
                actions,
                RecommendedAction.RECAPTURE_CURRENT_PHOTO,
            )
            return False

        if result.status == FaceVerificationStatus.RETRY_REQUIRED:
            if FaceReasonCode.DOCUMENT_FACE_NOT_USABLE in result.reason_codes:
                _add_retry(
                    DecisionReasonCode.DOCUMENT_FACE_NOT_USABLE,
                    reasons,
                    retry_reasons,
                    actions,
                    RecommendedAction.REUPLOAD_CURRENT_DOCUMENT,
                )
            else:
                _add_retry(
                    DecisionReasonCode.LIVE_FACE_NOT_USABLE,
                    reasons,
                    retry_reasons,
                    actions,
                    RecommendedAction.RECAPTURE_CURRENT_PHOTO,
                )
        return False


def _add_review(code, reasons, review_reasons, actions, action) -> None:
    reasons.append(code)
    review_reasons.append(code)
    actions.append(action)


def _add_retry(code, reasons, retry_reasons, actions, action) -> None:
    reasons.append(code)
    retry_reasons.append(code)
    actions.append(action)


def _review_copy(review_reasons, retry_reasons) -> tuple[str, str]:
    if DecisionReasonCode.CRITICAL_IDENTITY_MISMATCH in review_reasons:
        title = "Identity information does not match"
        explanation = (
            "A critical identity field differs between the previous and current Emirates IDs. "
            "The system will not auto-verify the case even if the face matches."
        )
    elif DecisionReasonCode.DOCUMENT_INTERNAL_MISMATCH in review_reasons:
        title = "Document information needs review"
        explanation = (
            "A checksum-protected value on the document conflicts with the corresponding "
            "printed value. Manual review is required."
        )
    elif DecisionReasonCode.CURRENT_DOCUMENT_EXPIRED in review_reasons:
        title = "Current Emirates ID is expired"
        explanation = "The current document cannot be auto-verified because it is expired."
    elif DecisionReasonCode.FACE_BELOW_THRESHOLD in review_reasons:
        title = "Face comparison needs review"
        explanation = (
            "Both faces were usable, but their similarity fell below the configured POC "
            "threshold. Because that threshold is not production-calibrated, the case is "
            "escalated rather than labelled as fraud."
        )
    elif any(
        code
        in {
            DecisionReasonCode.HISTORICAL_MRZ_CHECK_FAILED,
            DecisionReasonCode.CURRENT_MRZ_CHECK_FAILED,
        }
        for code in review_reasons
    ):
        title = "MRZ validation needs review"
        explanation = (
            "An MRZ was reconstructed but at least one check digit still failed after the "
            "available fallback read. The case requires review."
        )
    else:
        title = "Manual review required"
        explanation = "One or more identity checks produced evidence that requires review."

    if retry_reasons:
        explanation += " A separate input-quality issue should also be corrected before retrying."
    return title, explanation


def _retry_copy(retry_reasons) -> tuple[str, str]:
    if DecisionReasonCode.DOCUMENT_NAME_EVIDENCE_MISSING in retry_reasons:
        return (
            "Name information could not be compared",
            "At least one English or Arabic name must be comparable across the previous and "
            "current IDs. Upload clearer document images and try again.",
        )
    if DecisionReasonCode.DOCUMENT_FACE_NOT_USABLE in retry_reasons:
        return (
            "Current ID portrait could not be read",
            "The document checks completed, but the biometric pipeline could not obtain one "
            "usable portrait from the current Emirates ID. Upload a clearer current ID front.",
        )
    if DecisionReasonCode.LIVE_FACE_NOT_USABLE in retry_reasons:
        return (
            "Current photo needs to be retaken",
            "The current photo did not meet the face pipeline's usability requirements. "
            "Capture one clear, well-lit face and try again.",
        )
    if (
        DecisionReasonCode.CURRENT_MRZ_UNREADABLE in retry_reasons
        or DecisionReasonCode.CURRENT_DOCUMENT_EXPIRY_UNREADABLE in retry_reasons
    ):
        return (
            "Current Emirates ID needs a clearer upload",
            "Required information on the current Emirates ID could not be validated reliably. "
            "Upload a clearer front/back image or PDF and try again.",
        )
    if DecisionReasonCode.HISTORICAL_MRZ_UNREADABLE in retry_reasons:
        return (
            "Previous Emirates ID needs a clearer upload",
            "The previous Emirates ID MRZ could not be validated reliably. Upload a clearer "
            "front/back image or PDF and try again.",
        )
    return (
        "More information is needed",
        "One or more required identity inputs could not be read reliably. Correct the indicated "
        "input and run the verification again.",
    )


def _dedupe(items):
    return list(dict.fromkeys(items))
