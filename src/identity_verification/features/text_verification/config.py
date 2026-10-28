from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`.

    Thresholds exposed here are operational settings, not claims of calibrated
    verification accuracy. They should be revisited after the evaluation dataset is
    available.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Identity Verification Lab Emirates ID Identity Consistency Verification POC"
    ocr_engine: str = "easyocr"
    ocr_languages: str = "en,ar"
    ocr_gpu: bool = False
    # Disabled in tests and in any caller that never runs OCR, so they do not pay model
    # startup for nothing.
    warm_up_ocr_on_startup: bool = False
    warm_up_face_on_startup: bool = False
    max_upload_mb: int = 12
    max_document_pages: int = 2
    pdf_render_dpi: int = 220
    pdf_text_min_chars: int = 20
    mrz_crop_top_ratio: float = 0.52
    quality_min_dimension: int = 500
    quality_dark_threshold: float = 35.0
    quality_bright_threshold: float = 225.0
    quality_blur_warning_threshold: float = 35.0
    paddle_device: str = "cpu"
    paddle_cpu_threads: int = 4
    debug_include_raw_text: bool = False
    # Optional Nanonets benchmark settings. These are never used by the primary
    # verification path; the standalone benchmark script requires explicit opt-in.
    nanonets_api_key: str | None = None
    nanonets_model_id: str | None = None
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")

    @property
    def ocr_language_list(self) -> list[str]:
        return [item.strip() for item in self.ocr_languages.split(",") if item.strip()]

    @field_validator("mrz_crop_top_ratio")
    @classmethod
    def validate_crop_ratio(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("MRZ_CROP_TOP_RATIO must be between 0 and 1")
        return value

    @field_validator("max_document_pages")
    @classmethod
    def validate_page_limit(cls, value: int) -> int:
        if value not in {1, 2}:
            raise ValueError("MAX_DOCUMENT_PAGES must be 1 or 2 for this Emirates ID POC")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the process."""

    return Settings()
