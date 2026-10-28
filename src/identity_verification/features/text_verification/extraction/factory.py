from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.extraction.base import OCREngine
from identity_verification.features.text_verification.extraction.easyocr_engine import EasyOCREngine
from identity_verification.features.text_verification.extraction.paddleocr_engine import (
    PaddleOCREngine,
)
from identity_verification.features.text_verification.extraction.tesseract_engine import (
    TesseractOCREngine,
)


def build_ocr_engine(settings: Settings) -> OCREngine:
    """Create the configured OCR adapter behind one stable interface."""

    engine = settings.ocr_engine.strip().lower()
    if engine == "easyocr":
        return EasyOCREngine(settings.ocr_language_list, settings.ocr_gpu, settings.cache_dir)
    if engine == "paddleocr":
        return PaddleOCREngine(
            device=settings.paddle_device,
            cpu_threads=settings.paddle_cpu_threads,
            cache_dir=settings.cache_dir,
        )
    if engine == "tesseract":
        return TesseractOCREngine(settings.ocr_language_list)
    raise ValueError(f"Unsupported OCR engine '{settings.ocr_engine}'. Expected 'easyocr', 'paddleocr' or 'tesseract'.")
