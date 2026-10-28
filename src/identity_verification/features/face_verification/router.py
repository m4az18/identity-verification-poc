import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from identity_verification.features.face_verification.dependencies import (
    get_face_verification_service,
)
from identity_verification.features.face_verification.domain.models import FaceVerificationResult
from identity_verification.features.face_verification.service import FaceVerificationService
from identity_verification.features.text_verification.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/face", tags=["face-verification"])
WEB_DIR = Path(__file__).resolve().parent / "web"


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "feature": "face-verification",
        "model_loading": "lazy",
        "threshold_profile": "poc_uncalibrated",
    }


@router.post("/verify", response_model=FaceVerificationResult)
async def verify(
    id_photo: Annotated[UploadFile, File(description="Identity-document photo")],
    live_photo: Annotated[UploadFile, File(description="Current face photo")],
    service: Annotated[FaceVerificationService, Depends(get_face_verification_service)],
) -> FaceVerificationResult:
    try:
        max_bytes = get_settings().max_upload_mb * 1024 * 1024
        id_content = await _read_upload(id_photo, max_bytes)
        live_content = await _read_upload(live_photo, max_bytes)
        return await run_in_threadpool(service.verify_uploaded_images, id_content, live_content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("face_route_failure")
        raise HTTPException(status_code=503, detail="Face verification is unavailable.") from exc


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    content = await upload.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"{upload.filename or 'upload'} exceeds the upload limit.")
    if not content:
        raise ValueError(f"{upload.filename or 'upload'} is empty.")
    return content


def mount_face_verification_web(app: FastAPI) -> None:
    app.mount(
        "/face/static",
        StaticFiles(directory=WEB_DIR / "static"),
        name="face-verification-static",
    )
    app.add_api_route(
        "/face",
        lambda: FileResponse(WEB_DIR / "index.html"),
        methods=["GET"],
        include_in_schema=False,
        name="face-verification-ui",
    )
