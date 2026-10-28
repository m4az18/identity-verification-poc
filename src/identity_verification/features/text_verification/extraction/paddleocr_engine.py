import os
from contextlib import suppress
from pathlib import Path

import numpy as np

from identity_verification.features.text_verification.domain.models import OCRLine


class PaddleOCREngine:
    """PP-OCRv5 mobile adapter for visible Emirates ID text.

    This adapter is intentionally configured for a lightweight production-style baseline:

    * PP-OCRv5 mobile detection keeps the detector small.
    * The Arabic multilingual recognition model supports Arabic, English and digits.
    * Document orientation/unwarping sub-models stay disabled because they add extra model
      calls. They should only be enabled later if an evaluation set proves they improve
      difficult captures enough to justify the latency.
    * Paddle's reported recognition score is kept on each OCR line. Downstream code can
      therefore display *real extraction confidence* instead of inventing an identity
      probability from hand-written weights.

    The model is a candidate, not an assumed winner. It must still be benchmarked against
    EasyOCR on the same authorised Emirates ID set before it is called "better".
    """

    def __init__(self, *, device: str, cpu_threads: int, cache_dir: Path) -> None:
        self.device = device
        self.cpu_threads = cpu_threads
        self.cache_dir = cache_dir
        self._pipeline = None

    @property
    def name(self) -> str:
        return "paddleocr-ppocrv5-arabic-mobile"

    def _get_pipeline(self):
        if self._pipeline is None:
            # PaddleX/PaddleOCR resolve model/cache paths during initialization. Set the
            # cache directory before importing so downloaded weights stay inside the POC.
            paddlex_home = self.cache_dir / "paddlex"
            paddlex_home.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PADDLEX_HOME", str(paddlex_home))

            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("PaddleOCR is not installed. Run `uv sync --dev --extra ocr-paddle`.") from exc

            self._pipeline = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=self.device,
                engine="paddle_static",
                cpu_threads=self.cpu_threads,
            )
        return self._pipeline

    def warm_up(self) -> None:
        """Load weights and absorb one-off initialization before the first user request."""

        pipeline = self._get_pipeline()
        # A very small image is enough to make the runtime initialize its execution
        # graph. Any failure here is intentionally ignored by the caller's startup
        # wrapper so the real request can still attempt normal lazy initialization.
        with suppress(Exception):
            list(pipeline.predict(np.zeros((64, 64, 3), dtype=np.uint8)))

    def read(self, image_rgb: np.ndarray, *, allowlist: str | None = None) -> list[OCRLine]:
        pipeline = self._get_pipeline()
        predictions = pipeline.predict(image_rgb)

        lines: list[OCRLine] = []
        allowed = set(allowlist) if allowlist else None

        for prediction in predictions:
            # PaddleOCR 3.x result objects expose a JSON-compatible payload. Keeping this
            # adapter narrow isolates any library-specific schema from the rest of the app.
            payload = prediction.json
            result = payload.get("res", payload)
            texts = list(result.get("rec_texts", []))
            scores = list(result.get("rec_scores", []))
            polygons = list(result.get("rec_polys", []))

            for index, text in enumerate(texts):
                cleaned = str(text).strip()
                if allowed is not None:
                    # PaddleOCR does not expose EasyOCR's per-call allowlist. MRZ calls use
                    # this conservative post-filter only; normal visible-zone OCR is left
                    # untouched so Arabic is preserved.
                    cleaned = "".join(char for char in cleaned.upper() if char in allowed)
                if not cleaned:
                    continue

                confidence = float(scores[index]) if index < len(scores) else None
                bbox = None
                if index < len(polygons):
                    polygon = np.asarray(polygons[index]).tolist()
                    bbox = [[float(x), float(y)] for x, y in polygon]

                lines.append(OCRLine(text=cleaned, confidence=confidence, bbox=bbox))

        return sorted(lines, key=_reading_order_key)


def _reading_order_key(line: OCRLine) -> tuple[float, float]:
    if not line.bbox:
        return (0.0, 0.0)
    x = min(point[0] for point in line.bbox)
    y = min(point[1] for point in line.bbox)
    return (round(y / 8) * 8, x)
