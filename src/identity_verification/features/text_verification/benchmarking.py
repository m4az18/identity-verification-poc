from __future__ import annotations

import os
from functools import lru_cache
from threading import Lock

from identity_verification.features.text_verification.config import get_settings
from identity_verification.features.text_verification.dependencies import (
    get_verification_service,
)
from identity_verification.features.text_verification.extraction.easyocr_engine import (
    EasyOCREngine,
)
from identity_verification.features.text_verification.extraction.nanonets_ocr2 import (
    NanonetsOCR2Extractor,
)
from identity_verification.features.text_verification.extraction.paddleocr_engine import (
    PaddleOCREngine,
)
from identity_verification.features.text_verification.extraction.tesseract_engine import (
    TesseractOCREngine,
)
from identity_verification.features.text_verification.services.document_service import (
    DocumentService,
)
from identity_verification.features.text_verification.services.nanonets_document_service import (
    NanonetsDocumentService,
)
from identity_verification.features.text_verification.services.verification_service import (
    VerificationService,
)

BENCHMARK_BACKENDS = (
    "paddleocr",
    "easyocr",
    "tesseract",
    "nanonets_ocr2",
)

_WARMED_SERVICES: set[tuple[str, int]] = set()
_WARM_LOCK = Lock()


@lru_cache(maxsize=len(BENCHMARK_BACKENDS))
def get_benchmark_verification_service(
    backend: str,
) -> VerificationService:
    """Build one cached document-only benchmark service per backend.

    This module is deliberately separate from the normal dependency factory so the
    unified verification path and existing text-verification core remain untouched.
    """

    selected = normalize_benchmark_backend(backend)

    settings = get_settings()

    configured = settings.ocr_engine.strip().lower()

    if selected == configured:
        return get_verification_service()

    if selected == "paddleocr":
        engine = PaddleOCREngine(
            device=settings.paddle_device,
            cpu_threads=settings.paddle_cpu_threads,
            cache_dir=settings.cache_dir,
        )

        return VerificationService(
            DocumentService(
                settings,
                engine,
            )
        )

    if selected == "easyocr":
        engine = EasyOCREngine(
            settings.ocr_language_list,
            settings.ocr_gpu,
            settings.cache_dir,
        )

        return VerificationService(
            DocumentService(
                settings,
                engine,
            )
        )

    if selected == "tesseract":
        engine = TesseractOCREngine(
            settings.ocr_language_list,
        )

        return VerificationService(
            DocumentService(
                settings,
                engine,
            )
        )

    # Nanonets uses a different document-extraction architecture.
    model_id = os.getenv(
        "NANONETS_LOCAL_MODEL_ID",
        "nanonets/Nanonets-OCR2-3B",
    )

    try:
        max_new_tokens = int(
            os.getenv(
                "NANONETS_MAX_NEW_TOKENS",
                "1024",
            )
        )
    except ValueError as exc:
        raise ValueError("NANONETS_MAX_NEW_TOKENS must be an integer.") from exc

    if max_new_tokens < 64:
        raise ValueError("NANONETS_MAX_NEW_TOKENS must be at least 64.")

    extractor = NanonetsOCR2Extractor(
        model_id=model_id,
        cache_dir=settings.cache_dir,
        max_new_tokens=max_new_tokens,
    )

    return VerificationService(
        NanonetsDocumentService(
            settings,
            extractor,
        )
    )


def normalize_benchmark_backend(
    value: str | None,
) -> str:

    selected = (
        (value or "paddleocr")
        .strip()
        .lower()
        .replace(
            "-",
            "_",
        )
    )

    aliases = {
        "paddle": "paddleocr",
        "easy": "easyocr",
        "tess": "tesseract",
        "nanonets": "nanonets_ocr2",
        "nanonetsocr2": "nanonets_ocr2",
        "nanonets_ocr2_3b": "nanonets_ocr2",
    }

    selected = aliases.get(
        selected,
        selected,
    )

    if selected not in BENCHMARK_BACKENDS:
        expected = ", ".join(BENCHMARK_BACKENDS)

        raise ValueError(f"Unsupported extraction backend '{value}'. Expected one of: {expected}.")

    return selected


def warm_benchmark_backend(
    backend: str,
    service: VerificationService,
) -> None:
    """Warm each concrete benchmark service once per process.

    Model download/load and first-run setup happen before VerificationService.verify
    starts its timer, so one backend is not unfairly measured cold against another
    that was already resident in memory.
    """

    selected = normalize_benchmark_backend(backend)

    key = (
        selected,
        id(service),
    )

    if key in _WARMED_SERVICES:
        return

    with _WARM_LOCK:
        if key in _WARMED_SERVICES:
            return

        service.document_service.ocr_engine.warm_up()

        _WARMED_SERVICES.add(key)
