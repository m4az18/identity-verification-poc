import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from identity_verification.features.text_verification.config import get_settings
from identity_verification.features.text_verification.input.loader import (
    DocumentInput,
    InputFile,
    UnsupportedDocumentError,
)
from identity_verification.orchestration.dependencies import get_unified_verification_service
from identity_verification.orchestration.models import UnifiedVerificationResult
from identity_verification.orchestration.service import UnifiedVerificationService

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1/identity", tags=["identity-verification"])
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "feature": "unified-identity-verification",
        "policy": "poc-evidence-policy-v2",
    }


@router.post("/verify", response_model=UnifiedVerificationResult)
async def verify_identity(
    historical_document: Annotated[
        list[UploadFile],
        File(description="Historical Emirates ID: one PDF or front/back images"),
    ],
    current_document: Annotated[
        list[UploadFile],
        File(description="Current Emirates ID: one PDF or front/back images"),
    ],
    live_photo: Annotated[
        UploadFile,
        File(description="Current face photo (PNG/JPG/JPEG)"),
    ],
    service: Annotated[UnifiedVerificationService, Depends(get_unified_verification_service)],
) -> UnifiedVerificationResult:
    try:
        historical = await _read_document_uploads(historical_document)
        current = await _read_document_uploads(current_document)
        live = await _read_single_upload(live_photo, image_only=True)
        return await run_in_threadpool(service.verify, historical, current, live)
    except (UnsupportedDocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("unified_verification_runtime_error")
        raise HTTPException(
            status_code=503,
            detail=("Identity verification could not be completed because a required service failed."),
        ) from exc


async def _read_document_uploads(uploads: list[UploadFile]) -> DocumentInput:
    if not uploads:
        raise UnsupportedDocumentError("A document upload is required.")
    if len(uploads) > 2:
        raise UnsupportedDocumentError("Upload one PDF, or at most two images (front and back) for each ID.")
    return DocumentInput(files=tuple([await _read_single_upload(item) for item in uploads]))


async def _read_single_upload(upload: UploadFile, *, image_only: bool = False) -> InputFile:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await upload.read(max_bytes + 1)
    filename = upload.filename or "upload"
    if len(content) > max_bytes:
        raise UnsupportedDocumentError(f"{filename}: file exceeds the {settings.max_upload_mb} MB limit")
    if not content:
        raise UnsupportedDocumentError(f"{filename}: empty upload")
    if image_only and content.startswith(b"%PDF"):
        raise UnsupportedDocumentError("The current photo must be an image, not a PDF.")
    return InputFile(filename=filename, content=content)


def mount_unified_web(app: FastAPI, *, face_only: bool = False) -> None:
    app.mount(
        "/static",
        StaticFiles(directory=WEB_DIR / "static"),
        name="unified-static",
    )

    def page() -> Response:
        if face_only:
            html = (
                (WEB_DIR / "index.html")
                .read_text(encoding="utf-8")
                .replace("<body>", '<body data-face-only="true">', 1)
            )
            return HTMLResponse(html)
        return FileResponse(WEB_DIR / "index.html")

    app.add_api_route("/", page, methods=["GET"], include_in_schema=False, name="unified-verification-ui")
