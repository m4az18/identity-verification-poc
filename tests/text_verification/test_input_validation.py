from io import BytesIO

import pymupdf
import pytest
from PIL import Image

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.input.loader import (
    DocumentInput,
    DocumentLoader,
    InputFile,
    UnsupportedDocumentError,
)


def _png() -> bytes:
    image = Image.new("RGB", (900, 600), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pdf(page_count: int) -> bytes:
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page()
    content = document.tobytes()
    document.close()
    return content


def test_rejects_more_than_two_pages_for_one_emirates_id() -> None:
    loader = DocumentLoader(Settings())

    with pytest.raises(UnsupportedDocumentError, match="at most 2 pages"):
        loader.prepare_document(DocumentInput(files=(InputFile("id.pdf", _pdf(3)),)))


def test_rejects_mixed_pdf_and_image_two_file_mode() -> None:
    loader = DocumentLoader(Settings())

    with pytest.raises(UnsupportedDocumentError, match="two images"):
        loader.prepare_document(
            DocumentInput(
                files=(
                    InputFile("front.pdf", _pdf(1)),
                    InputFile("back.png", _png()),
                )
            )
        )
