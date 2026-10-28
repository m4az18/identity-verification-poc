from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from uuid import uuid4

from identity_verification.features.face_verification.service import (
    FaceVerificationService,
    decode_live_image,
)
from identity_verification.features.text_verification.input.loader import (
    DocumentInput,
    InputFile,
    reuse_prepared_documents,
)
from identity_verification.features.text_verification.services.verification_service import (
    VerificationService,
)
from identity_verification.orchestration.decision import DecisionPolicy
from identity_verification.orchestration.evidence import build_document_evidence_score
from identity_verification.orchestration.models import UnifiedVerificationResult

logger = logging.getLogger(__name__)


class UnifiedVerificationService:
    """Orchestrate document continuity and current-ID biometric verification."""

    def __init__(
        self,
        text_service: VerificationService,
        face_service: FaceVerificationService,
        decision_policy: DecisionPolicy | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        self.text_service = text_service
        self.face_service = face_service
        self.decision_policy = decision_policy or DecisionPolicy()
        self.parallel = (
            os.getenv("PARALLEL_VERIFICATION", "true").lower() in {"true", "1", "yes"} if parallel is None else parallel
        )

    def verify(
        self,
        historical_document: DocumentInput,
        current_document: DocumentInput,
        live_photo: InputFile,
    ) -> UnifiedVerificationResult:
        verification_id = str(uuid4())
        started = perf_counter()

        preparation_started = perf_counter()
        with reuse_prepared_documents():
            # The OCR service reuses these exact pages on its current-ID load. Cache
            # scope is this request only; uploaded images never enter a global cache.
            current_pages = self.text_service.document_service.loader.prepare_document(current_document)
            current_images = [page.image_rgb for page in current_pages if page.image_rgb is not None]
            live_image = decode_live_image(live_photo.content)
            preparation_ms = _elapsed_ms(preparation_started)

            def verify_face():
                face_started = perf_counter()
                result = self.face_service.verify_document_images(current_images, live_image)
                return result, _elapsed_ms(face_started)

            def verify_text():
                text_started = perf_counter()
                result = self.text_service.verify(historical_document, current_document)
                return result, _elapsed_ms(text_started)

            pipelines_started = perf_counter()
            if self.parallel:
                # One additional worker per active request, never one per page or
                # orientation. The face service retains its model-access lock.
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="identity-face") as executor:
                    face_future = executor.submit(verify_face)
                    text_result, text_ms = verify_text()
                    face_result, face_ms = face_future.result()
            else:
                text_result, text_ms = verify_text()
                face_result, face_ms = verify_face()
            pipelines_ms = _elapsed_ms(pipelines_started)

        decision_started = perf_counter()
        document_evidence_score = build_document_evidence_score(text_result)
        decision = self.decision_policy.decide(text_result, face_result)
        total_ms = _elapsed_ms(started)

        logger.info(
            "unified_verification_completed",
            extra={
                "event": "unified_verification_completed",
                "verification_id": verification_id,
                "text_case_id": text_result.case_id,
                "status": decision.status,
                "text_status": text_result.status,
                "face_status": face_result.status,
                "duration_ms": round(total_ms, 2),
            },
        )

        return UnifiedVerificationResult(
            verification_id=verification_id,
            decision=decision,
            text_evidence=text_result,
            face_evidence=face_result,
            document_evidence_score=document_evidence_score,
            timing_ms={
                "shared_preparation": round(preparation_ms, 2),
                "document_pipeline": round(text_ms, 2),
                "biometric_pipeline": round(face_ms, 2),
                "pipelines_wall": round(pipelines_ms, 2),
                "decision_and_assembly": round(_elapsed_ms(decision_started), 2),
                "total": round(total_ms, 2),
            },
            caveats=[
                "The facial threshold is an uncalibrated POC operating point.",
                "The preserved motion challenge is experimental and is not validated liveness/PAD.",
                (
                    "This POC checks identity consistency and biometric similarity; it does "
                    "not by itself prove physical-card authenticity."
                ),
                (
                    "The document evidence index is a transparent policy index, not a "
                    "probability or overall identity confidence."
                ),
            ],
        )


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000
