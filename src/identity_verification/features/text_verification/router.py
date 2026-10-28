import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from identity_verification.features.text_verification.benchmarking import (
    get_benchmark_verification_service,
    normalize_benchmark_backend,
    warm_benchmark_backend,
)
from identity_verification.features.text_verification.config import get_settings
from identity_verification.features.text_verification.dependencies import get_verification_service
from identity_verification.features.text_verification.domain.models import VerificationResult
from identity_verification.features.text_verification.input.loader import (
    DocumentInput,
    InputFile,
    UnsupportedDocumentError,
)
from identity_verification.features.text_verification.services.verification_service import (
    VerificationService,
)

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1/text", tags=["text-verification"])
WEB_DIR = Path(__file__).resolve().parent / "web"
DIAGNOSTICS_WEB_DIR = Path(__file__).resolve().parent / "diagnostics_web"


@router.get("/health")
def health() -> dict[str, str]:
    """Report text-verification readiness without initializing the OCR model."""

    return {
        "status": "ok",
        "feature": "text-verification",
        "version": "0.7.0",
        "ocr_engine": settings.ocr_engine,
    }


@router.post("/verify", response_model=VerificationResult)
async def verify(
    historical_document: Annotated[
        list[UploadFile],
        File(description="Historical Emirates ID: one 1-2 page PDF or up to two images"),
    ],
    current_document: Annotated[
        list[UploadFile],
        File(description="Current Emirates ID: one 1-2 page PDF or up to two images"),
    ],
    service: Annotated[VerificationService, Depends(get_verification_service)],
    backend: Annotated[
        str | None,
        Query(
            description=(
                "Optional document-check benchmark backend. The unified verification "
                "flow continues to use the configured default backend."
            )
        ),
    ] = None,
) -> VerificationResult:
    """Verify two logical Emirates ID documents without persisting uploaded files.

    ``backend`` is intentionally scoped to the document-only feature. PaddleOCR and
    EasyOCR share the same line-level field extractor. Nanonets OCR2 uses its native
    structured document adapter, then rejoins the same normalization, MRZ validation,
    matching and decision logic.
    """

    try:
        historical = await _read_document_uploads(historical_document)
        current = await _read_document_uploads(current_document)

        selected_backend = normalize_benchmark_backend(backend or settings.ocr_engine)
        selected_service = service
        if selected_backend != settings.ocr_engine.strip().lower():
            selected_service = get_benchmark_verification_service(selected_backend)

        # Benchmark model setup is paid once outside the reported verification timing.
        # This avoids comparing one model cold against another already-warm model.
        await run_in_threadpool(warm_benchmark_backend, selected_backend, selected_service)
        return await run_in_threadpool(selected_service.verify, historical, current)
    except (UnsupportedDocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("verification_runtime_error")
        raise HTTPException(
            status_code=503,
            detail="Verification could not be completed because the selected extraction backend failed.",
        ) from exc


async def _read_document_uploads(uploads: list[UploadFile]) -> DocumentInput:
    if not uploads:
        raise UnsupportedDocumentError("A document upload is required.")
    if len(uploads) > 2:
        raise UnsupportedDocumentError("Upload one PDF, or at most two images (front and back) for each ID.")

    files: list[InputFile] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024
    for upload in uploads:
        content = await upload.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise UnsupportedDocumentError(
                f"{upload.filename or 'upload'}: file exceeds the {settings.max_upload_mb} MB limit"
            )
        files.append(InputFile(filename=upload.filename or "upload", content=content))
    return DocumentInput(files=tuple(files))


def mount_text_verification_web(app: FastAPI) -> None:
    """Mount the text-verification demo UI under a feature-specific URL namespace."""

    app.mount(
        "/text/static",
        StaticFiles(directory=WEB_DIR / "static"),
        name="text-verification-static",
    )
    app.add_api_route(
        "/text",
        lambda: FileResponse(WEB_DIR / "index.html"),
        methods=["GET"],
        include_in_schema=False,
        name="text-verification-ui",
    )
    app.mount(
        "/text/diagnostics/static",
        StaticFiles(directory=DIAGNOSTICS_WEB_DIR / "static"),
        name="text-verification-diagnostics-static",
    )
    app.add_api_route(
        "/text/diagnostics",
        lambda: FileResponse(DIAGNOSTICS_WEB_DIR / "index.html"),
        methods=["GET"],
        include_in_schema=False,
        name="text-verification-diagnostics-ui",
    )
