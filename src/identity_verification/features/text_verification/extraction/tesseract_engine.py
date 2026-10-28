from __future__ import annotations

from collections import defaultdict

import numpy as np

from identity_verification.features.text_verification.domain.models import OCRLine

_TESSERACT_LANGUAGE_CODES = {
    "en": "eng",
    "ar": "ara",
}


class TesseractOCREngine:
    """General full-document Tesseract OCR adapter."""

    def __init__(self, languages: list[str]) -> None:
        self.languages = languages

        self._lang_spec = (
            "+".join(
                _TESSERACT_LANGUAGE_CODES.get(
                    language.strip().lower(),
                    language.strip(),
                )
                for language in languages
                if language.strip()
            )
            or "eng"
        )

        self._checked = False

    @property
    def name(self) -> str:
        return "tesseract-eng-ara"

    def _ensure_available(self) -> None:
        if self._checked:
            return

        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("pytesseract is not installed. Run `uv sync --extra ocr-tesseract`.") from exc

        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "The Tesseract executable is not available. On macOS run `brew install tesseract tesseract-lang`."
            ) from exc

        installed_languages = set(pytesseract.get_languages(config=""))

        required_languages = set(self._lang_spec.split("+"))

        missing = sorted(required_languages - installed_languages)

        if missing:
            raise RuntimeError(
                f"Missing Tesseract language data: {', '.join(missing)}. Run `brew install tesseract-lang`."
            )

        self._checked = True

    def warm_up(self) -> None:
        """Verify that Tesseract and its language files work."""

        self._ensure_available()

        self.read(
            np.full(
                (64, 64, 3),
                255,
                dtype=np.uint8,
            )
        )

    def read(
        self,
        image_rgb: np.ndarray,
        *,
        allowlist: str | None = None,
    ) -> list[OCRLine]:

        self._ensure_available()

        import pytesseract

        # General Emirates-ID OCR.
        config = "--oem 1 --psm 11"

        # More constrained mode if this method is ever used
        # for MRZ-like text.
        if allowlist:
            config = f"--oem 1 --psm 6 -c tessedit_char_whitelist={allowlist} -c load_system_dawg=0 -c load_freq_dawg=0"

        data = pytesseract.image_to_data(
            image_rgb,
            lang=self._lang_spec,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        grouped: dict[
            tuple[int, int, int, int],
            list[dict[str, float | str]],
        ] = defaultdict(list)

        count = len(data.get("text", []))

        for index in range(count):
            text = str(data["text"][index]).strip()

            if not text:
                continue

            try:
                confidence_pct = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence_pct = -1.0

            # Tesseract uses -1 for structural rows
            # rather than recognized words.
            if confidence_pct < 0:
                continue

            key = (
                int(data["page_num"][index]),
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )

            grouped[key].append(
                {
                    "text": text,
                    "confidence": confidence_pct / 100.0,
                    "left": float(data["left"][index]),
                    "top": float(data["top"][index]),
                    "width": float(data["width"][index]),
                    "height": float(data["height"][index]),
                }
            )

        lines: list[OCRLine] = []

        for words in grouped.values():
            if not words:
                continue

            text = " ".join(str(word["text"]) for word in words).strip()

            if allowlist:
                allowed = set(allowlist)

                text = "".join(character for character in text.upper() if character in allowed)

            if not text:
                continue

            # Build one confidence score for the whole line.
            # Longer words contribute more than tiny one-character words.
            weights = [
                max(
                    1,
                    len(str(word["text"])),
                )
                for word in words
            ]

            total_weight = sum(weights)

            confidence = (
                sum(
                    float(word["confidence"]) * weight
                    for word, weight in zip(
                        words,
                        weights,
                        strict=True,
                    )
                )
                / total_weight
            )

            left = min(float(word["left"]) for word in words)

            top = min(float(word["top"]) for word in words)

            right = max(float(word["left"]) + float(word["width"]) for word in words)

            bottom = max(float(word["top"]) + float(word["height"]) for word in words)

            bbox = [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ]

            lines.append(
                OCRLine(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        return sorted(
            lines,
            key=_reading_order_key,
        )


def _reading_order_key(
    line: OCRLine,
) -> tuple[float, float]:

    if not line.bbox:
        return (0.0, 0.0)

    x = min(point[0] for point in line.bbox)

    y = min(point[1] for point in line.bbox)

    return (
        round(y / 8) * 8,
        x,
    )
