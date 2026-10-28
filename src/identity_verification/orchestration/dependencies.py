from functools import lru_cache

from identity_verification.features.face_verification.dependencies import (
    get_face_verification_service,
)
from identity_verification.features.text_verification.dependencies import get_verification_service
from identity_verification.orchestration.service import UnifiedVerificationService


@lru_cache(maxsize=1)
def get_unified_verification_service() -> UnifiedVerificationService:
    return UnifiedVerificationService(
        text_service=get_verification_service(),
        face_service=get_face_verification_service(),
    )
