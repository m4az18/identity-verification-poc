from types import SimpleNamespace

from identity_verification.features.face_verification.domain.models import (
    FaceReasonCode,
    FaceVerificationResult,
    FaceVerificationStatus,
)
from identity_verification.features.text_verification.domain.models import (
    ComparisonStatus,
    VerificationStatus,
)
from identity_verification.orchestration.decision import DecisionPolicy
from identity_verification.orchestration.models import (
    DecisionReasonCode,
    OverallVerificationStatus,
    RecommendedAction,
)


def _comparison(field, status):
    return SimpleNamespace(field=field, status=status)


def _document(*, mrz_ok=True, expiry=True, internal=None):
    if mrz_ok is None:
        mrz = SimpleNamespace(valid_structure=False, checks=SimpleNamespace(all_available_pass=None))
    else:
        mrz = SimpleNamespace(valid_structure=True, checks=SimpleNamespace(all_available_pass=mrz_ok))
    return SimpleNamespace(mrz=mrz, internal_checks={"current_expiry_valid": expiry, **(internal or {})})


def _text(
    *,
    id_status=ComparisonStatus.MATCH,
    dob_status=ComparisonStatus.MATCH,
    supporting=ComparisonStatus.MATCH,
    old_mrz=True,
    current_mrz=True,
    expiry=True,
    internal=None,
    status=VerificationStatus.CONSISTENT,
):
    return SimpleNamespace(
        status=status,
        comparisons=[
            _comparison("id_number", id_status),
            _comparison("date_of_birth", dob_status),
            _comparison("name_en", supporting),
            _comparison("name_ar", ComparisonStatus.MISSING),
            _comparison("nationality", ComparisonStatus.MATCH),
            _comparison("sex", ComparisonStatus.MATCH),
        ],
        historical=_document(mrz_ok=old_mrz, expiry=True),
        current=_document(mrz_ok=current_mrz, expiry=expiry, internal=internal),
    )


def _face(status, *reasons):
    return FaceVerificationResult(status=status, reason_codes=list(reasons))


def test_only_all_required_evidence_passes_auto_verify():
    decision = DecisionPolicy().decide(_text(), _face(FaceVerificationStatus.MATCH))
    assert decision.status == OverallVerificationStatus.VERIFIED
    assert decision.reason_codes == [DecisionReasonCode.ALL_REQUIRED_EVIDENCE_PASSED]
    assert decision.recommended_actions == [RecommendedAction.NONE]


def test_critical_id_mismatch_is_explicit_and_face_match_cannot_override_it():
    decision = DecisionPolicy().decide(
        _text(id_status=ComparisonStatus.REVIEW),
        _face(FaceVerificationStatus.MATCH),
    )
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert decision.title == "Identity information does not match"
    assert DecisionReasonCode.CRITICAL_IDENTITY_MISMATCH in decision.reason_codes
    assert RecommendedAction.MANUAL_REVIEW in decision.recommended_actions


def test_missing_critical_document_field_requests_new_documents():
    decision = DecisionPolicy().decide(
        _text(dob_status=ComparisonStatus.MISSING),
        _face(FaceVerificationStatus.MATCH),
    )
    assert decision.status == OverallVerificationStatus.RETRY_REQUIRED
    assert DecisionReasonCode.DOCUMENT_REQUIRED_FIELDS_MISSING in decision.reason_codes
    assert RecommendedAction.REUPLOAD_DOCUMENTS in decision.recommended_actions


def test_unreadable_current_mrz_is_retry_not_generic_review():
    decision = DecisionPolicy().decide(
        _text(current_mrz=None),
        _face(FaceVerificationStatus.MATCH),
    )
    assert decision.status == OverallVerificationStatus.RETRY_REQUIRED
    assert DecisionReasonCode.CURRENT_MRZ_UNREADABLE in decision.reason_codes
    assert RecommendedAction.REUPLOAD_CURRENT_DOCUMENT in decision.recommended_actions


def test_mrz_check_failure_after_fallback_requires_review():
    decision = DecisionPolicy().decide(
        _text(current_mrz=False),
        _face(FaceVerificationStatus.MATCH),
    )
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert DecisionReasonCode.CURRENT_MRZ_CHECK_FAILED in decision.reason_codes


def test_document_portrait_failure_requests_current_id_not_selfie():
    decision = DecisionPolicy().decide(
        _text(),
        _face(FaceVerificationStatus.RETRY_REQUIRED, FaceReasonCode.DOCUMENT_FACE_NOT_USABLE),
    )
    assert decision.status == OverallVerificationStatus.RETRY_REQUIRED
    assert decision.title == "Current ID portrait could not be read"
    assert DecisionReasonCode.DOCUMENT_FACE_NOT_USABLE in decision.reason_codes
    assert RecommendedAction.REUPLOAD_CURRENT_DOCUMENT in decision.recommended_actions
    assert RecommendedAction.RECAPTURE_CURRENT_PHOTO not in decision.recommended_actions


def test_live_photo_failure_requests_recapture():
    decision = DecisionPolicy().decide(
        _text(),
        _face(FaceVerificationStatus.RETRY_REQUIRED, FaceReasonCode.LIVE_FACE_NOT_USABLE),
    )
    assert decision.status == OverallVerificationStatus.RETRY_REQUIRED
    assert DecisionReasonCode.LIVE_FACE_NOT_USABLE in decision.reason_codes
    assert RecommendedAction.RECAPTURE_CURRENT_PHOTO in decision.recommended_actions


def test_face_below_threshold_is_review_not_fraud_or_auto_reject():
    decision = DecisionPolicy().decide(
        _text(),
        _face(FaceVerificationStatus.NO_MATCH, FaceReasonCode.BELOW_THRESHOLD),
    )
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert DecisionReasonCode.FACE_BELOW_THRESHOLD in decision.reason_codes
    assert RecommendedAction.MANUAL_REVIEW in decision.recommended_actions


def test_expired_current_document_cannot_be_overridden_by_face_match():
    decision = DecisionPolicy().decide(
        _text(expiry=False),
        _face(FaceVerificationStatus.MATCH),
    )
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert DecisionReasonCode.CURRENT_DOCUMENT_EXPIRED in decision.reason_codes


def test_face_service_failure_is_unable_to_verify():
    decision = DecisionPolicy().decide(
        _text(),
        _face(FaceVerificationStatus.UNAVAILABLE, FaceReasonCode.MODEL_UNAVAILABLE),
    )
    assert decision.status == OverallVerificationStatus.UNABLE_TO_VERIFY
    assert DecisionReasonCode.FACE_SERVICE_UNAVAILABLE in decision.reason_codes
    assert RecommendedAction.RETRY_SERVICE in decision.recommended_actions


def test_review_reason_keeps_separate_retry_action():
    decision = DecisionPolicy().decide(
        _text(id_status=ComparisonStatus.REVIEW),
        _face(FaceVerificationStatus.RETRY_REQUIRED, FaceReasonCode.LIVE_FACE_NOT_USABLE),
    )
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert DecisionReasonCode.CRITICAL_IDENTITY_MISMATCH in decision.reason_codes
    assert DecisionReasonCode.LIVE_FACE_NOT_USABLE in decision.reason_codes
    assert RecommendedAction.MANUAL_REVIEW in decision.recommended_actions
    assert RecommendedAction.RECAPTURE_CURRENT_PHOTO in decision.recommended_actions


def test_missing_both_names_requests_document_retry():
    text = _text()
    for item in text.comparisons:
        if item.field in {"name_en", "name_ar"}:
            item.status = ComparisonStatus.MISSING
    decision = DecisionPolicy().decide(text, _face(FaceVerificationStatus.MATCH))
    assert decision.status == OverallVerificationStatus.RETRY_REQUIRED
    assert DecisionReasonCode.DOCUMENT_NAME_EVIDENCE_MISSING in decision.reason_codes


def test_checksum_protected_visible_mrz_conflict_requires_review():
    text = _text(
        internal={
            "id_number_visible_vs_mrz": {
                "pass": False,
                "severity": "critical",
            }
        }
    )
    decision = DecisionPolicy().decide(text, _face(FaceVerificationStatus.MATCH))
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert DecisionReasonCode.DOCUMENT_INTERNAL_MISMATCH in decision.reason_codes


def test_unknown_text_review_state_never_silently_auto_verifies():
    text = _text(status=VerificationStatus.REVIEW_REQUIRED)
    decision = DecisionPolicy().decide(text, _face(FaceVerificationStatus.MATCH))
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert DecisionReasonCode.DOCUMENT_REVIEW_REQUIRED_OTHER in decision.reason_codes


def test_supporting_identity_variation_requires_review_even_with_face_match():
    decision = DecisionPolicy().decide(
        _text(supporting=ComparisonStatus.VARIATION, status=VerificationStatus.REVIEW_REQUIRED),
        _face(FaceVerificationStatus.MATCH),
    )
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert DecisionReasonCode.SUPPORTING_IDENTITY_VARIATION in decision.reason_codes


def test_unreadable_previous_mrz_requests_previous_id_only():
    decision = DecisionPolicy().decide(
        _text(old_mrz=None, status=VerificationStatus.REVIEW_REQUIRED),
        _face(FaceVerificationStatus.MATCH),
    )
    assert decision.status == OverallVerificationStatus.RETRY_REQUIRED
    assert DecisionReasonCode.HISTORICAL_MRZ_UNREADABLE in decision.reason_codes
    assert RecommendedAction.REUPLOAD_HISTORICAL_DOCUMENT in decision.recommended_actions
    assert RecommendedAction.REUPLOAD_CURRENT_DOCUMENT not in decision.recommended_actions


def test_unreadable_current_expiry_requests_current_id_retry():
    decision = DecisionPolicy().decide(
        _text(expiry=None, status=VerificationStatus.REVIEW_REQUIRED),
        _face(FaceVerificationStatus.MATCH),
    )
    assert decision.status == OverallVerificationStatus.RETRY_REQUIRED
    assert DecisionReasonCode.CURRENT_DOCUMENT_EXPIRY_UNREADABLE in decision.reason_codes
    assert RecommendedAction.REUPLOAD_CURRENT_DOCUMENT in decision.recommended_actions


def test_face_review_outranks_document_retry_but_preserves_both_actions():
    decision = DecisionPolicy().decide(
        _text(current_mrz=None, status=VerificationStatus.REVIEW_REQUIRED),
        _face(FaceVerificationStatus.NO_MATCH, FaceReasonCode.BELOW_THRESHOLD),
    )
    assert decision.status == OverallVerificationStatus.REVIEW_REQUIRED
    assert DecisionReasonCode.CURRENT_MRZ_UNREADABLE in decision.reason_codes
    assert DecisionReasonCode.FACE_BELOW_THRESHOLD in decision.reason_codes
    assert RecommendedAction.REUPLOAD_CURRENT_DOCUMENT in decision.recommended_actions
    assert RecommendedAction.RECAPTURE_CURRENT_PHOTO in decision.recommended_actions
    assert RecommendedAction.MANUAL_REVIEW in decision.recommended_actions


def test_bad_id_portrait_and_bad_document_mrz_keep_precise_retry_actions():
    decision = DecisionPolicy().decide(
        _text(current_mrz=None, status=VerificationStatus.REVIEW_REQUIRED),
        _face(FaceVerificationStatus.RETRY_REQUIRED, FaceReasonCode.DOCUMENT_FACE_NOT_USABLE),
    )
    assert decision.status == OverallVerificationStatus.RETRY_REQUIRED
    assert DecisionReasonCode.CURRENT_MRZ_UNREADABLE in decision.reason_codes
    assert DecisionReasonCode.DOCUMENT_FACE_NOT_USABLE in decision.reason_codes
    assert RecommendedAction.REUPLOAD_CURRENT_DOCUMENT in decision.recommended_actions
    assert RecommendedAction.RECAPTURE_CURRENT_PHOTO not in decision.recommended_actions
