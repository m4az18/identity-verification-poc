import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from identity_verification.common.logging import configure_logging
from identity_verification.features.face_verification.dependencies import get_face_verification_service
from identity_verification.features.face_verification.router import (
    mount_face_verification_web,
)
from identity_verification.features.face_verification.router import (
    router as face_verification_router,
)
from identity_verification.features.text_verification.config import get_settings
from identity_verification.features.text_verification.lifecycle import warm_up_text_verification
from identity_verification.features.text_verification.router import (
    mount_text_verification_web,
)
from identity_verification.features.text_verification.router import (
    router as text_verification_router,
)
from identity_verification.orchestration.router import (
    mount_unified_web,
)
from identity_verification.orchestration.router import (
    router as unified_verification_router,
)

configure_logging()
logger = logging.getLogger(__name__)
FACE_ONLY_MODE = os.getenv("FACE_ONLY_MODE", "false").lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not FACE_ONLY_MODE:
        warm_up_text_verification()
    if get_settings().warm_up_face_on_startup:
        try:
            get_face_verification_service().warm_up()
        except Exception:
            logger.exception("face_warm_up_failed")
    yield


app = FastAPI(
    title="Identity Verification Lab Emirates ID Verification",
    version="0.5.0",
    description=(
        "Modular Emirates ID verification combining document identity continuity "
        "and current-ID biometric face verification through an explicit decision policy."
    ),
    lifespan=lifespan,
)

app.include_router(face_verification_router)
if FACE_ONLY_MODE:
    mount_unified_web(app, face_only=True)
    app.add_api_route("/face", lambda: RedirectResponse("/"), methods=["GET"], include_in_schema=False)
    app.add_api_route("/text", lambda: RedirectResponse("/"), methods=["GET"], include_in_schema=False)
else:
    mount_face_verification_web(app)
    app.include_router(unified_verification_router)
    app.include_router(text_verification_router)
    mount_unified_web(app)
    mount_text_verification_web(app)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "identity-verification-poc",
        "version": "0.5.0",
        "decision_policy": "poc-evidence-policy-v2",
    }


def run() -> None:
    import uvicorn

    uvicorn.run(
        "identity_verification.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
