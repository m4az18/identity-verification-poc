from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.extraction.factory import build_ocr_engine
from identity_verification.features.text_verification.extraction.tesseract_engine import TesseractOCREngine


def test_tesseract_is_available_to_the_primary_ocr_pipeline():
    engine = build_ocr_engine(Settings(ocr_engine="tesseract", ocr_languages="en,ar"))
    assert isinstance(engine, TesseractOCREngine)
    assert engine._lang_spec == "eng+ara"
