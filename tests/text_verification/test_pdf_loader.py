from io import BytesIO

import pymupdf
from PIL import Image

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.input.loader import (
    DocumentInput,
    DocumentLoader,
    InputFile,
)


def _png() -> bytes:
    image = Image.new("RGB", (900, 600), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_two_page_pdf_is_processed_as_one_logical_document() -> None:
    document = pymupdf.open()
    front = document.new_page()
    front.insert_text((72, 72), "Emirates ID Number 784-1995-1234567-8")
    back = document.new_page()
    back.insert_text((72, 72), "Date of Birth 13/02/1995")
    content = document.tobytes()
    document.close()

    pages = DocumentLoader(Settings()).prepare_document(DocumentInput(files=(InputFile("id.pdf", content),)))

    assert len(pages) == 2
    assert pages[0].source_type == "pdf_text"
    assert pages[1].source_type == "pdf_text"
    assert "784-1995-1234567-8" in (pages[0].text_layer or "")
    assert "13/02/1995" in (pages[1].text_layer or "")


def test_scanned_pdf_page_is_rendered_for_ocr() -> None:
    document = pymupdf.open()
    document.new_page(width=400, height=250)
    content = document.tobytes()
    document.close()

    pages = DocumentLoader(Settings()).prepare_document(DocumentInput(files=(InputFile("scan.pdf", content),)))

    assert len(pages) == 1
    assert pages[0].source_type == "pdf_scan"
    assert pages[0].text_layer is None
    assert pages[0].image_rgb is not None


def test_two_images_can_be_selected_as_one_document() -> None:
    image = _png()
    pages = DocumentLoader(Settings()).prepare_document(
        DocumentInput(
            files=(
                InputFile("front.png", image),
                InputFile("back.png", image),
            )
        )
    )

    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.source_type == "image" for page in pages)
