from __future__ import annotations

import base64
import logging
from contextlib import suppress
from io import BytesIO
from threading import Lock
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from identity_verification.features.face_verification.document_portrait import (
    document_portrait_candidates,
)
from identity_verification.features.face_verification.domain.models import (
    FaceReasonCode,
    FaceVerificationResult,
    FaceVerificationStatus,
)
from identity_verification.features.face_verification.face_preprocessing import (
    analyze_standardized_face,
)
from identity_verification.features.face_verification.original_loader import (
    load_original_face_modules,
)

logger = logging.getLogger(__name__)


class FaceVerificationService:
    """Adapter around the preserved standalone face-verification implementation.

    Detection, quality checks, AuraFace feature extraction, similarity and threshold
    logic remain in the teammate's original byte-for-byte-preserved source. This adapter
    only:
      * converts upload bytes to RGB arrays;
      * serializes model access like the original Gradio queue;
      * when a full rendered ID page is too large for the original detector, presents a
        deterministic document crop to that same original pipeline and stops at the first
        usable face (never the highest similarity score).
    """

    def __init__(self) -> None:
        self._model_lock = Lock()

    def warm_up(self) -> None:
        """Load shared face weights before requests when configured."""
        verification, _ = load_original_face_modules()
        with self._model_lock:
            verification.load_models()

    def verify_images(
        self,
        id_images_rgb: list[np.ndarray],
        live_image_rgb: np.ndarray,
    ) -> FaceVerificationResult:
        started = perf_counter()
        verification = None

        try:
            verification, config = load_original_face_modules()
            with self._model_lock:
                live_started = perf_counter()
                try:
                    live_face, live_operations, live_preview = self._find_live_face(
                        verification,
                        live_image_rgb,
                    )
                except ValueError as exc:
                    return self._retry_result(
                        FaceReasonCode.LIVE_FACE_NOT_USABLE,
                        str(exc),
                        started,
                        user_message=_friendly_live_message(str(exc)),
                        live_ms=_elapsed_ms(live_started),
                    )
                live_ms = _elapsed_ms(live_started)

                document_started = perf_counter()
                (
                    document_face,
                    page_used,
                    strategy,
                    document_operations,
                    document_face_preview,
                    document_preview_steps,
                    document_errors,
                ) = self._find_document_face(verification, id_images_rgb)
                document_ms = _elapsed_ms(document_started)

                if document_face is None:
                    detail = "No current-ID page produced one usable portrait. " + " | ".join(document_errors)
                    return self._retry_result(
                        FaceReasonCode.DOCUMENT_FACE_NOT_USABLE,
                        detail,
                        started,
                        user_message=(
                            "We could not detect a usable portrait on the current Emirates ID. "
                            "Upload a clear image/PDF of the ID front with the portrait fully visible."
                        ),
                        live_ms=live_ms,
                        document_ms=document_ms,
                    )

                compare_started = perf_counter()
                score = float(
                    np.clip(
                        np.dot(
                            document_face.normed_embedding,
                            live_face.normed_embedding,
                        ),
                        -1,
                        1,
                    )
                )
                threshold = float(config.get_threshold())
                matched = score >= threshold
                comparison_ms = _elapsed_ms(compare_started)

                reason_codes = [FaceReasonCode.MATCHED if matched else FaceReasonCode.BELOW_THRESHOLD]
                if page_used and page_used > 1:
                    reason_codes.append(FaceReasonCode.DOCUMENT_FACE_FALLBACK_PAGE)
                if strategy and strategy != "full_page":
                    reason_codes.append(FaceReasonCode.DOCUMENT_PORTRAIT_ADAPTER_USED)

                verification.log_event(
                    "comparison_completed",
                    match=matched,
                    integration="unified_fastapi",
                )
                return FaceVerificationResult(
                    status=(FaceVerificationStatus.MATCH if matched else FaceVerificationStatus.NO_MATCH),
                    reason_codes=reason_codes,
                    similarity=round(score, 6),
                    threshold=round(threshold, 6),
                    document_page_used=page_used,
                    document_input_strategy=strategy,
                    document_preprocessing=document_operations,
                    live_preprocessing=live_operations,
                    document_face_preview_data_url=_image_data_url(
                        document_face_preview,
                        max_edge=320,
                        transparent=True,
                    ),
                    live_face_preview_data_url=_image_data_url(live_preview, max_edge=320),
                    document_preprocessing_previews=[
                        {
                            "label": label,
                            "operation": operation,
                            "image_data_url": _image_data_url(image, max_edge=900),
                        }
                        for label, operation, image in document_preview_steps
                    ],
                    quality_passed=True,
                    detail=(
                        "Both faces passed the preserved detector and quality gate, then "
                        "used the shared five-point canonical alignment path. Passive "
                        "liveness/PAD was not assessed."
                    ),
                    user_message=(
                        "The current photo matches the portrait on the current Emirates ID."
                        if matched
                        else "The face comparison fell below the configured POC threshold."
                    ),
                    timing_ms={
                        "live_face": round(live_ms, 2),
                        "document_face": round(document_ms, 2),
                        "comparison": round(comparison_ms, 2),
                        "total": round(_elapsed_ms(started), 2),
                    },
                )
        except Exception as exc:  # model/network/runtime boundary
            logger.exception("face_verification_unavailable")
            if verification is not None:
                with suppress(Exception):
                    verification.log_event(
                        "comparison_failed",
                        error_type=type(exc).__name__,
                        integration="unified_fastapi",
                    )
            return FaceVerificationResult(
                status=FaceVerificationStatus.UNAVAILABLE,
                reason_codes=[FaceReasonCode.MODEL_UNAVAILABLE],
                quality_passed=None,
                detail="Face verification could not be completed. Check server logs and retry.",
                user_message="Face verification is temporarily unavailable. Please retry.",
                timing_ms={"total": round(_elapsed_ms(started), 2)},
            )

    def _find_document_face(self, verification, id_images_rgb):
        """Preserve standalone behavior first, then use deterministic document crops."""

        errors: list[str] = []

        # Pass 1: exact standalone behavior on each uploaded/rendered page.
        for page_number, image_rgb in enumerate(id_images_rgb, start=1):
            try:
                result = analyze_standardized_face(verification, image_rgb)
                return (
                    result.face,
                    page_number,
                    "full_page",
                    list(result.operations),
                    result.aligned_image_rgb,
                    result.preview_steps,
                    errors,
                )
            except ValueError as exc:
                errors.append(f"page {page_number}/full_page: {exc}")

        # Pass 2: integration-only document normalization. The original detector,
        # quality gate and AuraFace path are still called unchanged for every crop.
        for page_number, image_rgb in enumerate(id_images_rgb, start=1):
            for candidate in document_portrait_candidates(image_rgb):
                try:
                    result = analyze_standardized_face(verification, candidate.image_rgb)
                    operations = [candidate.strategy, *result.operations]
                    return (
                        result.face,
                        page_number,
                        candidate.strategy,
                        operations,
                        result.aligned_image_rgb,
                        result.preview_steps,
                        errors,
                    )
                except ValueError as exc:
                    errors.append(f"page {page_number}/{candidate.strategy}: {exc}")

        return None, None, None, [], None, (), errors

    @staticmethod
    def _find_live_face(verification, image_rgb):
        result = analyze_standardized_face(verification, image_rgb)
        return result.face, list(result.operations), result.aligned_image_rgb

    def verify_uploaded_images(self, id_image: bytes, live_image: bytes) -> FaceVerificationResult:
        original_document = _decode_image(id_image, label="identity document image")
        return self.verify_document_images(
            [original_document],
            _decode_image(live_image, label="current photo"),
            original_upload=True,
        )

    def verify_document_images(
        self,
        id_images_rgb: list[np.ndarray],
        live_image_rgb: np.ndarray,
        *,
        original_upload: bool = False,
    ) -> FaceVerificationResult:
        """Keep the branch's full visual pipeline for both upload and unified flows."""

        result = self.verify_images(
            id_images_rgb,
            live_image_rgb,
        )
        page_index = (result.document_page_used or 1) - 1
        document_image = id_images_rgb[page_index] if id_images_rgb else None
        document_preview = _image_data_url(document_image, max_edge=900)
        previews = (
            [
                {
                    "label": "Original ID" if original_upload else f"Current ID page {page_index + 1}",
                    "operation": (
                        "RGB decoded with EXIF orientation applied"
                        if original_upload
                        else "Rendered and normalized page shared with OCR"
                    ),
                    "image_data_url": document_preview,
                }
            ]
            if document_preview
            else []
        )
        previews.extend(item.model_dump() for item in result.document_preprocessing_previews)
        if result.document_face_preview_data_url:
            previews.append(
                {
                    "label": "Transparent ID face",
                    "operation": "Background removed and exported as transparent PNG",
                    "image_data_url": result.document_face_preview_data_url,
                }
            )
        return result.model_copy(
            update={
                "document_preprocessing": list(dict.fromkeys(result.document_preprocessing)),
                "document_original_preview_data_url": document_preview if original_upload else None,
                "document_preprocessed_preview_data_url": None if original_upload else document_preview,
                "document_preprocessing_previews": previews,
            }
        )

    @staticmethod
    def _retry_result(
        code: FaceReasonCode,
        detail: str,
        started: float,
        *,
        user_message: str,
        live_ms: float | None = None,
        document_ms: float | None = None,
    ) -> FaceVerificationResult:
        timing = {"total": round(_elapsed_ms(started), 2)}
        if live_ms is not None:
            timing["live_face"] = round(live_ms, 2)
        if document_ms is not None:
            timing["document_face"] = round(document_ms, 2)
        return FaceVerificationResult(
            status=FaceVerificationStatus.RETRY_REQUIRED,
            reason_codes=[code],
            quality_passed=False,
            detail=detail,
            user_message=user_message,
            timing_ms=timing,
        )


def decode_live_image(content: bytes) -> np.ndarray:
    return _decode_image(content, label="current photo")


def _decode_image(content: bytes, *, label: str) -> np.ndarray:
    if not content:
        raise ValueError(f"The {label} is empty.")
    try:
        with Image.open(BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(image)
    except Exception as exc:
        raise ValueError(f"The {label} must be a readable PNG/JPG/JPEG image.") from exc


def _friendly_live_message(detail: str) -> str:
    lowered = detail.lower()
    if "too small" in lowered:
        return "The face in the current photo is too small. Move closer and try again."
    if "too dark" in lowered or "too bright" in lowered or "lighting" in lowered:
        return "The current photo lighting is not suitable. Use even lighting and try again."
    if "blur" in lowered:
        return "The current photo is too blurry. Hold still, refocus, and try again."
    if "exactly one face" in lowered:
        return "Use a current photo containing exactly one clearly visible face."
    return "The current photo could not be used reliably. Capture a clearer photo and try again."


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def _image_data_url(
    image_rgb: np.ndarray | None,
    *,
    max_edge: int,
    transparent: bool = False,
) -> str | None:
    if image_rgb is None or image_rgb.size == 0:
        return None
    pixels = np.asarray(image_rgb, dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    output = BytesIO()
    if transparent:
        image = _transparent_portrait(image)
        image.save(output, format="PNG", optimize=True)
        mime = "image/png"
    else:
        image.save(output, format="JPEG", quality=86, optimize=True)
        mime = "image/jpeg"
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _transparent_portrait(image: Image.Image) -> Image.Image:
    """Remove the aligned crop background with a stable feathered portrait mask."""

    import cv2

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, int(height * 0.51))
    axes = (max(1, int(width * 0.43)), max(1, int(height * 0.49)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    blur = max(3, int(min(width, height) * 0.06) | 1)
    alpha = cv2.GaussianBlur(mask, (blur, blur), 0)
    return Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
