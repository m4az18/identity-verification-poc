from typing import Protocol

import numpy as np

from identity_verification.features.text_verification.domain.models import OCRLine


class OCREngine(Protocol):
    """Minimal OCR contract used by the document pipeline.

    Keeping the application dependent on this protocol instead of a specific library
    lets EasyOCR and PaddleOCR be benchmarked behind the same downstream validation and
    matching logic.
    """

    @property
    def name(self) -> str: ...

    def read(
        self,
        image_rgb: np.ndarray,
        *,
        allowlist: str | None = None,
    ) -> list[OCRLine]: ...

    def warm_up(self) -> None:
        """Load model weights now rather than inside the first user request.

        Lazy loading is right for tests and health checks, which must not pay model
        startup. It is wrong for the first real request, which was absorbing the weight
        download, the model load and the first-inference graph setup and reporting the
        total as document processing time. Separating the two means the lazy path stays
        for callers that never OCR, while a server can pay the cost once at startup.
        """
        ...
