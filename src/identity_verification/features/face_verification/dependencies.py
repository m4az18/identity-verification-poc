from functools import lru_cache

from identity_verification.features.face_verification.service import FaceVerificationService


@lru_cache(maxsize=1)
def get_face_verification_service() -> FaceVerificationService:
    return FaceVerificationService()
