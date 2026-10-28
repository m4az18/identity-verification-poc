import logging
from time import perf_counter

from identity_verification.features.text_verification.config import get_settings
from identity_verification.features.text_verification.dependencies import get_verification_service

logger = logging.getLogger(__name__)


def warm_up_text_verification() -> None:
    """Warm the configured OCR engine once at application startup when enabled."""

    settings = get_settings()
    if not settings.warm_up_ocr_on_startup:
        return

    started = perf_counter()
    try:
        get_verification_service().document_service.ocr_engine.warm_up()
        logger.info(
            "ocr_warm_up_complete",
            extra={
                "event": "ocr_warm_up_complete",
                "ocr_engine": settings.ocr_engine,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "ocr_warm_up_failed",
            extra={"event": "ocr_warm_up_failed", "ocr_engine": settings.ocr_engine},
        )
