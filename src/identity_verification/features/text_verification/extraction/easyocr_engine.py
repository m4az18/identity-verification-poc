from pathlib import Path

import numpy as np

from identity_verification.features.text_verification.domain.models import OCRLine


class EasyOCREngine:
    """Lazy EasyOCR adapter.

    Model loading is deferred until the first OCR request so health checks and unit tests
    do not pay the model startup cost.
    """

    def __init__(self, languages: list[str], gpu: bool, cache_dir: Path) -> None:
        self.languages = languages
        self.gpu = gpu
        self.cache_dir = cache_dir
        self._reader = None

    @property
    def name(self) -> str:
        return "easyocr"

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("EasyOCR is not installed. Run `uv sync --dev --extra ocr-easy`.") from exc

            model_dir = self.cache_dir / "easyocr"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=str(model_dir),
                user_network_directory=str(model_dir / "user_network"),
            )
        return self._reader

    def warm_up(self) -> None:
        """Build the reader and run one tiny inference.

        Constructing the reader is not sufficient. Torch defers a further one-off cost to
        the first actual inference, so a trivial image is pushed through to absorb it.
        """

        reader = self._get_reader()
        reader.readtext(np.zeros((32, 32, 3), dtype=np.uint8), detail=1, paragraph=False)

    def read(self, image_rgb: np.ndarray, *, allowlist: str | None = None) -> list[OCRLine]:
        reader = self._get_reader()
        results = reader.readtext(
            image_rgb,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
        )
        lines: list[OCRLine] = []
        for bbox, text, confidence in results:
            cleaned = str(text).strip()
            if cleaned:
                lines.append(
                    OCRLine(
                        text=cleaned,
                        confidence=float(confidence),
                        bbox=[[float(x), float(y)] for x, y in bbox],
                    )
                )
        return sorted(lines, key=_reading_order_key)


def _reading_order_key(line: OCRLine) -> tuple[float, float]:
    if not line.bbox:
        return (0.0, 0.0)
    x = min(point[0] for point in line.bbox)
    y = min(point[1] for point in line.bbox)
    return (round(y / 8) * 8, x)
