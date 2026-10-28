from enum import StrEnum

from pydantic import BaseModel, Field


class FacePreprocessingPreview(BaseModel):
    label: str
    operation: str
    image_data_url: str


class FaceVerificationStatus(StrEnum):
    """Operational outcome of the biometric comparison.

    MATCH/NO_MATCH are evidence outcomes after both faces passed the original
    detector and quality path. RETRY_REQUIRED means the submitted evidence was
    not usable enough to make that comparison. UNAVAILABLE is reserved for a
    service/model failure rather than user evidence quality.
    """

    MATCH = "match"
    NO_MATCH = "no_match"
    RETRY_REQUIRED = "retry_required"
    UNAVAILABLE = "unavailable"


class FaceReasonCode(StrEnum):
    MATCHED = "FACE_MATCHED"
    BELOW_THRESHOLD = "FACE_BELOW_THRESHOLD"
    LIVE_FACE_NOT_USABLE = "LIVE_FACE_NOT_USABLE"
    DOCUMENT_FACE_NOT_USABLE = "DOCUMENT_FACE_NOT_USABLE"
    DOCUMENT_FACE_FALLBACK_PAGE = "DOCUMENT_FACE_FALLBACK_PAGE"
    DOCUMENT_PORTRAIT_ADAPTER_USED = "DOCUMENT_PORTRAIT_ADAPTER_USED"
    FEATURE_EXTRACTION_FAILED = "FACE_FEATURE_EXTRACTION_FAILED"
    MODEL_UNAVAILABLE = "FACE_MODEL_UNAVAILABLE"


class FaceVerificationResult(BaseModel):
    status: FaceVerificationStatus
    reason_codes: list[FaceReasonCode] = Field(default_factory=list)
    similarity: float | None = None
    threshold: float | None = None
    threshold_profile: str = "poc_uncalibrated"
    document_page_used: int | None = None
    document_input_strategy: str | None = None
    document_preprocessing: list[str] = Field(default_factory=list)
    live_preprocessing: list[str] = Field(default_factory=list)
    quality_passed: bool | None = None
    detector: str = "YOLOv8-Face-Detection"
    recognizer: str = "AuraFace-v1"
    alignment: str = "five-point canonical alignment"
    liveness_assessed: bool = False
    document_original_preview_data_url: str | None = None
    document_preprocessed_preview_data_url: str | None = None
    document_face_preview_data_url: str | None = None
    live_face_preview_data_url: str | None = None
    document_preprocessing_previews: list[FacePreprocessingPreview] = Field(default_factory=list)
    detail: str | None = None
    user_message: str | None = None
    timing_ms: dict[str, float] = Field(default_factory=dict)
