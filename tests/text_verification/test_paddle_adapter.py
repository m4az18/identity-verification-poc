from pathlib import Path

import numpy as np

from identity_verification.features.text_verification.extraction.paddleocr_engine import (
    PaddleOCREngine,
)


class _Prediction:
    json = {
        "res": {
            "rec_texts": ["Name", "أحمد", "ABC<123"],
            "rec_scores": [0.99, 0.91, 0.88],
            "rec_polys": [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[0, 20], [10, 20], [10, 30], [0, 30]],
                [[0, 40], [10, 40], [10, 50], [0, 50]],
            ],
        }
    }


class _Pipeline:
    def predict(self, image_rgb: np.ndarray):
        return [_Prediction()]


def test_paddle_adapter_maps_prediction_to_common_schema_without_importing_model() -> None:
    engine = PaddleOCREngine(device="cpu", cpu_threads=2, cache_dir=Path(".cache-test"))
    engine._pipeline = _Pipeline()

    lines = engine.read(np.zeros((50, 50, 3), dtype=np.uint8))

    assert [line.text for line in lines] == ["Name", "أحمد", "ABC<123"]
    assert lines[1].confidence == 0.91


def test_paddle_adapter_filters_non_mrz_characters_for_mrz_fallback() -> None:
    engine = PaddleOCREngine(device="cpu", cpu_threads=2, cache_dir=Path(".cache-test"))
    engine._pipeline = _Pipeline()

    lines = engine.read(
        np.zeros((50, 50, 3), dtype=np.uint8),
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
    )

    assert [line.text for line in lines] == ["NAME", "ABC<123"]
