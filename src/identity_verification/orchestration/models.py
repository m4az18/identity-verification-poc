from enum import StrEnum

from pydantic import BaseModel, Field

from identity_verification.features.face_verification.domain.models import FaceVerificationResult
from identity_verification.features.text_verification.domain.models import VerificationResult


class OverallVerificationStatus(StrEnum):
    VERIFIED = "verified"
    REVIEW_REQUIRED = "review_required"
    RETRY_REQUIRED = "retry_required"
    UNABLE_TO_VERIFY = "unable_to_verify"


class DecisionReasonCode(StrEnum):
    ALL_REQUIRED_EVIDENCE_PASSED = "ALL_REQUIRED_EVIDENCE_PASSED"

    CRITICAL_IDENTITY_MISMATCH = "CRITICAL_IDENTITY_MISMATCH"
    DOCUMENT_REQUIRED_FIELDS_MISSING = "DOCUMENT_REQUIRED_FIELDS_MISSING"
    HISTORICAL_MRZ_UNREADABLE = "HISTORICAL_MRZ_UNREADABLE"
    CURRENT_MRZ_UNREADABLE = "CURRENT_MRZ_UNREADABLE"
    HISTORICAL_MRZ_CHECK_FAILED = "HISTORICAL_MRZ_CHECK_FAILED"
    CURRENT_MRZ_CHECK_FAILED = "CURRENT_MRZ_CHECK_FAILED"
    DOCUMENT_INTERNAL_MISMATCH = "DOCUMENT_INTERNAL_MISMATCH"
    SUPPORTING_IDENTITY_VARIATION = "SUPPORTING_IDENTITY_VARIATION"
    DOCUMENT_NAME_EVIDENCE_MISSING = "DOCUMENT_NAME_EVIDENCE_MISSING"
    DOCUMENT_REVIEW_REQUIRED_OTHER = "DOCUMENT_REVIEW_REQUIRED_OTHER"
    CURRENT_DOCUMENT_EXPIRED = "CURRENT_DOCUMENT_EXPIRED"
    CURRENT_DOCUMENT_EXPIRY_UNREADABLE = "CURRENT_DOCUMENT_EXPIRY_UNREADABLE"

    FACE_BELOW_THRESHOLD = "FACE_BELOW_THRESHOLD"
    LIVE_FACE_NOT_USABLE = "LIVE_FACE_NOT_USABLE"
    DOCUMENT_FACE_NOT_USABLE = "DOCUMENT_FACE_NOT_USABLE"
    FACE_SERVICE_UNAVAILABLE = "FACE_SERVICE_UNAVAILABLE"


class RecommendedAction(StrEnum):
    NONE = "none"
    MANUAL_REVIEW = "manual_review"
    REUPLOAD_DOCUMENTS = "reupload_documents"
    REUPLOAD_HISTORICAL_DOCUMENT = "reupload_historical_document"
    REUPLOAD_CURRENT_DOCUMENT = "reupload_current_document"
    RECAPTURE_CURRENT_PHOTO = "recapture_current_photo"
    RETRY_SERVICE = "retry_service"


class EvidenceComponent(BaseModel):
    key: str
    label: str
    weight: float
    earned: float
    available: bool
    passed: bool | None = None
    detail: str


class PolicyEvidenceScore(BaseModel):
    """A transparent policy index, never a calibrated confidence/probability."""

    score: float
    maximum: float = 100.0
    available_weight: float = 100.0
    components: list[EvidenceComponent] = Field(default_factory=list)
    label: str = "Document evidence index"
    disclaimer: str


class Decision(BaseModel):
    status: OverallVerificationStatus
    title: str
    policy_version: str = "poc-evidence-policy-v2"
    reason_codes: list[DecisionReasonCode] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    explanation: str


class UnifiedVerificationResult(BaseModel):
    verification_id: str
    decision: Decision
    text_evidence: VerificationResult
    face_evidence: FaceVerificationResult
    document_evidence_score: PolicyEvidenceScore
    timing_ms: dict[str, float] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
