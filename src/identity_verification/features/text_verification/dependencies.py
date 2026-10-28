from functools import lru_cache

from identity_verification.features.text_verification.config import get_settings
from identity_verification.features.text_verification.extraction.factory import build_ocr_engine
from identity_verification.features.text_verification.services.document_service import (
    DocumentService,
)
from identity_verification.features.text_verification.services.verification_service import (
    VerificationService,
)


@lru_cache
def get_verification_service() -> VerificationService:
    """Build heavy OCR dependencies once and reuse them across API requests."""

    settings = get_settings()
    ocr_engine = build_ocr_engine(settings)
    return VerificationService(DocumentService(settings, ocr_engine))
