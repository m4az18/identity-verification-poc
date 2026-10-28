from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class VerificationStatus(StrEnum):
    """Top-level decision returned by the text-verification POC."""

    CONSISTENT = "consistent"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_DATA = "insufficient_data"


class ComparisonStatus(StrEnum):
    """Field-level comparison outcome."""

    MATCH = "match"
    VARIATION = "variation"
    REVIEW = "review"
    NOT_COMPARED = "not_compared"
    MISSING = "missing"


class OCRLine(BaseModel):
    """One OCR detection with optional model confidence and geometry."""

    text: str
    confidence: float | None = None
    bbox: list[list[float]] | None = None


class FieldValue(BaseModel):
    """Raw and normalized forms of a single identity attribute.

    Keeping the raw value preserves auditability while the normalized value is used
    for deterministic comparison. `sources` records where the value came from without
    persisting the original document image.
    """

    raw: str | None = None
    normalized: str | None = None
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)


class IdentityRecord(BaseModel):
    id_number: FieldValue = Field(default_factory=FieldValue)
    name_en: FieldValue = Field(default_factory=FieldValue)
    name_ar: FieldValue = Field(default_factory=FieldValue)
    date_of_birth: FieldValue = Field(default_factory=FieldValue)
    nationality: FieldValue = Field(default_factory=FieldValue)
    sex: FieldValue = Field(default_factory=FieldValue)
    expiry_date: FieldValue = Field(default_factory=FieldValue)


class MRZChecks(BaseModel):
    document_number: bool | None = None
    date_of_birth: bool | None = None
    expiry_date: bool | None = None
    composite: bool | None = None

    @property
    def all_available_pass(self) -> bool | None:
        values = [
            self.document_number,
            self.date_of_birth,
            self.expiry_date,
            self.composite,
        ]
        available = [value for value in values if value is not None]

        if not available:
            return None

        return all(available)


class MRZAttempt(BaseModel):
    """One MRZ-reading attempt retained for transparency and debugging.

    The attempt stores only structured OCR/MRZ evidence already returned by the POC;
    it does not persist the uploaded image. `selected` marks the candidate ultimately
    used by the document pipeline.
    """

    method: str
    raw_lines: list[str] = Field(default_factory=list)
    valid_structure: bool = False
    checks: MRZChecks = Field(default_factory=MRZChecks)
    passing_checks: int = 0
    selected: bool = False
    error: str | None = None


class MRZData(BaseModel):
    format: str = "TD1"
    raw_lines: list[str] = Field(default_factory=list)
    document_code: str | None = None
    issuing_state: str | None = None
    document_number: str | None = None
    date_of_birth_raw: str | None = None
    sex: str | None = None
    expiry_date_raw: str | None = None
    nationality: str | None = None
    name: str | None = None
    optional_data_1: str | None = None
    optional_data_2: str | None = None
    emirates_id_candidate: str | None = None
    checks: MRZChecks = Field(default_factory=MRZChecks)
    valid_structure: bool = False
    error: str | None = None
    extraction_method: str | None = None
    attempts: list[MRZAttempt] = Field(default_factory=list)


class ImageDiagnostics(BaseModel):
    width: int | None = None
    height: int | None = None
    mean_brightness: float | None = None
    blur_variance: float | None = None
    warnings: list[str] = Field(default_factory=list)


class PageSummary(BaseModel):
    """Metadata describing how one uploaded page was processed.

    OCR evidence and page previews are returned only when debug output is enabled.
    """

    label: str
    source_type: str
    ocr_source: str
    diagnostics: ImageDiagnostics | None = None
    ocr_width: int | None = None
    ocr_height: int | None = None
    ocr_lines: list[OCRLine] | None = None
    preview_data_url: str | None = None


class DocumentExtraction(BaseModel):
    """Structured result for one historical or current identity document."""

    filenames: list[str]
    page_count: int
    record: IdentityRecord
    mrz: MRZData | None = None
    internal_checks: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    pages: list[PageSummary] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)
    ocr_engine: str
    raw_text: dict[str, str] | None = None


class FieldComparison(BaseModel):
    field: str
    historical: str | None
    current: str | None
    status: ComparisonStatus
    similarity: float | None = None
    note: str | None = None


class VerificationResult(BaseModel):
    case_id: str
    status: VerificationStatus
    comparisons: list[FieldComparison]
    reasons: list[str] = Field(default_factory=list)
    historical: DocumentExtraction
    current: DocumentExtraction
    timing_ms: dict[str, float] = Field(default_factory=dict)
