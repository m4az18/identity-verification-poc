"""Constrained MRZ recognition with Tesseract.

The MRZ is not a scene-text problem and should not be handed to a scene-text model.
EasyOCR and PaddleOCR are trained on natural imagery — signs, packaging, receipts — and
carry language priors that actively work against OCR-B. On this project's own test card
EasyOCR read `ZUHAIR` as `7UHAIR` and misread the document number badly enough that two
of four check digits failed on a genuine, undamaged document.

Everything about the MRZ is known in advance: a fixed monospaced font designed for
machine reading, three lines of exactly thirty characters, and an alphabet of
`A-Z0-9<`. That is a constrained recognition problem, and Tesseract with a character
allowlist and a fixed page-segmentation mode is the appropriate tool. It also has no
neural detection stage, so it costs milliseconds rather than seconds — the MRZ pass stops
being a meaningful part of the latency budget.

This reader is deliberately not the general OCR engine. The visible zone is proportional
print in two scripts over a security background, which *is* a scene-text problem and
where the neural engines belong. Using different tools for the two zones is the point,
not an inconsistency.

Tesseract is an optional dependency. When it is unavailable the reader reports that
plainly and the caller falls back to the configured OCR engine, so the pipeline degrades
rather than breaking.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from identity_verification.features.text_verification.domain.models import OCRLine

MRZ_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

# PSM 6 treats the crop as a single uniform block of text, which is exactly what an MRZ
# is. The default mode looks for varied page layout and fragments the three lines.
TESSERACT_CONFIG = (
    f"--psm 6 --oem 1 -c tessedit_char_whitelist={MRZ_ALLOWLIST} -c load_system_dawg=0 -c load_freq_dawg=0"
)


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    """Report whether both the Python binding and the Tesseract binary are present.

    Cached because it shells out, and the answer cannot change within a process.
    """

    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:  # noqa: BLE001
        return False
    return True


class TesseractMRZReader:
    """Read MRZ lines from a prepared crop.

    Returns raw line strings rather than `OCRLine` objects with geometry: the TD1 layout
    already fixes what each position means, so line order is the only structural
    information the parser needs, and Tesseract preserves it.
    """

    @property
    def name(self) -> str:
        return "tesseract"

    def read_lines(self, crop_rgb: np.ndarray) -> list[str]:
        """Return candidate MRZ lines, longest-looking first in document order."""

        if crop_rgb is None or crop_rgb.size == 0 or not tesseract_available():
            return []

        import pytesseract
        from PIL import Image

        # `load_system_dawg` and `load_freq_dawg` are disabled in the config above because
        # Tesseract's dictionaries assume natural language. An MRZ is not language, and
        # dictionary correction actively damages it.
        try:
            text = pytesseract.image_to_string(
                Image.fromarray(crop_rgb),
                config=TESSERACT_CONFIG,
            )
        except Exception:  # noqa: BLE001
            return []

        lines: list[str] = []
        for raw in text.splitlines():
            cleaned = re.sub(r"[^A-Z0-9<]", "", raw.upper())
            # Fragments shorter than this are noise from the crop edges, not MRZ rows.
            if len(cleaned) >= 14:
                lines.append(cleaned)
        return lines

    def read(self, image_rgb: np.ndarray, *, allowlist: str | None = None) -> list[OCRLine]:
        """Adapter shape, so this can stand in for an OCREngine where one is expected."""

        del allowlist
        return [OCRLine(text=line, confidence=None) for line in self.read_lines(image_rgb)]
