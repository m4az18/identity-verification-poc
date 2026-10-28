from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import pymupdf
from PIL import Image, ImageOps

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.domain.models import ImageDiagnostics
from identity_verification.features.text_verification.input.document_normalizer import (
    normalize_document_image,
)
from identity_verification.features.text_verification.quality.metrics import analyze_image


class UnsupportedDocumentError(ValueError):
    """Raised when an upload cannot be handled safely or unambiguously."""


@dataclass(slots=True, frozen=True)
class InputFile:
    filename: str
    content: bytes


@dataclass(slots=True, frozen=True)
class DocumentInput:
    """One logical ID document submitted through a single UI control.

    Supported forms are deliberately simple:
    - one PDF containing one or two pages; or
    - one or two image files selected together.
    """

    files: tuple[InputFile, ...]


@dataclass(slots=True)
class PreparedPage:
    filename: str
    page_number: int
    image_rgb: np.ndarray | None
    text_layer: str | None
    source_type: str
    diagnostics: ImageDiagnostics | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"page_{self.page_number}"


_prepared_documents: ContextVar[dict[tuple[int, DocumentInput], list[PreparedPage]] | None] = ContextVar(
    "prepared_documents", default=None
)


@contextmanager
def reuse_prepared_documents() -> Iterator[None]:
    """Reuse rendered pages within one request; release all images on exit."""

    token = _prepared_documents.set({})
    try:
        yield
    finally:
        _prepared_documents.reset(token)


class DocumentLoader:
    """Normalize PDFs and images into a small collection of prepared pages."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare_document(self, document: DocumentInput) -> list[PreparedPage]:
        cache = _prepared_documents.get()
        if cache is None:
            return self._prepare_document(document)
        key = (id(self), document)
        if key not in cache:
            cache[key] = self._prepare_document(document)
        return cache[key]

    def _prepare_document(self, document: DocumentInput) -> list[PreparedPage]:
        if not document.files:
            raise UnsupportedDocumentError("At least one document file is required.")
        if len(document.files) > 2:
            raise UnsupportedDocumentError("Upload one PDF, or at most two images (front and back) for each ID.")

        for item in document.files:
            self._validate_size(item)

        if len(document.files) == 1:
            item = document.files[0]
            if item.content.startswith(b"%PDF"):
                return self._prepare_pdf(item)
            return [self._prepare_image(item, page_number=1)]

        # Two-file mode is intentionally image-only. Mixing PDFs and images makes page
        # ordering ambiguous and complicates the user experience without adding value
        # for the current Emirates ID POC.
        if any(item.content.startswith(b"%PDF") for item in document.files):
            raise UnsupportedDocumentError(
                "When uploading two files, select two images. For PDF input, upload one "
                "PDF containing the front and back pages."
            )

        return [
            self._prepare_image(document.files[0], page_number=1),
            self._prepare_image(document.files[1], page_number=2),
        ]

    def _validate_size(self, item: InputFile) -> None:
        if not item.content:
            raise UnsupportedDocumentError(f"{item.filename}: empty upload")
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(item.content) > max_bytes:
            raise UnsupportedDocumentError(f"{item.filename}: file exceeds the {self.settings.max_upload_mb} MB limit")

    def _prepare_image(self, item: InputFile, page_number: int) -> PreparedPage:
        try:
            with Image.open(BytesIO(item.content)) as image:
                # Phone cameras may store pixels sideways and rely on EXIF metadata for
                # display. Applying the transform before OCR gives the model the same
                # orientation the user sees in their photo viewer.
                image = ImageOps.exif_transpose(image).convert("RGB")
                image_rgb = np.asarray(image)
        except Exception as exc:
            raise UnsupportedDocumentError(f"{item.filename}: expected a readable PNG/JPG/JPEG image or PDF") from exc

        normalized = normalize_document_image(image_rgb)
        return PreparedPage(
            filename=item.filename,
            page_number=page_number,
            image_rgb=normalized.image_rgb,
            text_layer=None,
            source_type="image",
            diagnostics=analyze_image(normalized.image_rgb, self.settings),
            warnings=[f"preprocessing:{operation}" for operation in normalized.operations],
        )

    def _prepare_pdf(self, item: InputFile) -> list[PreparedPage]:
        try:
            with pymupdf.open(stream=item.content, filetype="pdf") as document:
                if document.page_count == 0:
                    raise UnsupportedDocumentError(f"{item.filename}: PDF has no pages")
                if document.page_count > self.settings.max_document_pages:
                    raise UnsupportedDocumentError(
                        f"{item.filename}: expected at most {self.settings.max_document_pages} pages (front and back)."
                    )

                pages: list[PreparedPage] = []
                for index in range(document.page_count):
                    page = document[index]
                    text = page.get_text("text").strip()

                    # Render the complete page rather than extracting an embedded image.
                    # This keeps scanned PDFs and normal PDFs on the same preprocessing
                    # path and retains an image for MRZ analysis even when a text layer exists.
                    pixmap = page.get_pixmap(dpi=self.settings.pdf_render_dpi, alpha=False)
                    image_rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                        pixmap.height, pixmap.width, pixmap.n
                    )
                    if pixmap.n == 4:
                        image_rgb = image_rgb[:, :, :3]

                    normalized = normalize_document_image(image_rgb)

                    usable_text = text if self._has_usable_text(text) else None
                    pages.append(
                        PreparedPage(
                            filename=item.filename,
                            page_number=index + 1,
                            image_rgb=normalized.image_rgb,
                            text_layer=usable_text,
                            source_type="pdf_text" if usable_text else "pdf_scan",
                            diagnostics=analyze_image(normalized.image_rgb, self.settings),
                            warnings=[f"preprocessing:{operation}" for operation in normalized.operations],
                        )
                    )
                return pages
        except UnsupportedDocumentError:
            raise
        except Exception as exc:
            raise UnsupportedDocumentError(f"{item.filename}: unreadable PDF") from exc

    def _has_usable_text(self, text: str) -> bool:
        visible = "".join(char for char in text if not char.isspace())
        return len(visible) >= self.settings.pdf_text_min_chars
