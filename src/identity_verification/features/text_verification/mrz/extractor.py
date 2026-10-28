import re

import numpy as np

from identity_verification.features.text_verification.domain.models import (
    MRZAttempt,
    MRZData,
    OCRLine,
)
from identity_verification.features.text_verification.extraction.base import OCREngine
from identity_verification.features.text_verification.input.preprocess import enhance_mrz_crop
from identity_verification.features.text_verification.mrz.parser import parse_td1
from identity_verification.features.text_verification.mrz.repair import repair_td1
from identity_verification.features.text_verification.mrz.tesseract_reader import (
    TesseractMRZReader,
    tesseract_available,
)

MRZ_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


class MRZExtractor:
    """Parse MRZ evidence while avoiding redundant model inference.

    The full-page OCR output is always tried first. A dedicated crop OCR is used only
    when the already-available text cannot reconstruct a valid TD1 MRZ. This preserves
    the validation value of the MRZ without paying for a second OCR pass on every back
    side.
    """

    def __init__(self, ocr_engine: OCREngine, crop_top_ratio: float) -> None:
        self.ocr_engine = ocr_engine
        self.crop_top_ratio = crop_top_ratio
        self.tesseract = TesseractMRZReader()

    def parse_existing(
        self,
        *,
        text_layer: str | None = None,
        ocr_lines: list[OCRLine] | None = None,
    ) -> MRZData | None:
        candidates: list[str] = []
        if text_layer:
            candidates.extend(self._candidates_from_text(text_layer))
        if ocr_lines:
            candidates.extend(self._reconstruct_lines(ocr_lines))

        parsed = self._best_candidate(candidates)
        if parsed:
            parsed.extraction_method = "existing_text"
            parsed.attempts = [
                MRZAttempt(
                    method="existing_text",
                    raw_lines=list(parsed.raw_lines),
                    valid_structure=parsed.valid_structure,
                    checks=parsed.checks.model_copy(deep=True),
                    passing_checks=_passing_checks(parsed),
                    selected=True,
                    error=parsed.error,
                )
            ]
        return parsed

    def extract_with_fallback(
        self,
        image_rgb: np.ndarray | None,
        *,
        text_layer: str | None = None,
        ocr_lines: list[OCRLine] | None = None,
    ) -> MRZData:
        """Return the strongest MRZ read while retaining every attempted read.

        Workflow:
        1. Reuse full-page OCR first (zero extra model inference).
        2. If that MRZ does not fully validate, re-read only the MRZ crop with
           Tesseract.
        3. If still unresolved, let the configured OCR engine read the same crop.
        4. Select the candidate with the most passing ICAO checks.

        The attempt history is returned in JSON so the UI can explain why, for
        example, a composite check failed on the first read but passed after the
        dedicated MRZ re-read.
        """

        parsed = self.parse_existing(text_layer=text_layer, ocr_lines=ocr_lines)
        attempts: list[tuple[str, MRZData]] = []

        if parsed and parsed.valid_structure:
            method = parsed.extraction_method or "existing_text"
            attempts.append((method, parsed))
            if parsed.checks.all_available_pass is True:
                return _with_attempt_history(parsed, attempts, selected_method=method)

        if image_rgb is None:
            if parsed:
                return _with_attempt_history(
                    parsed,
                    attempts,
                    selected_method=parsed.extraction_method or "existing_text",
                )
            return MRZData(error="No image was available for MRZ OCR.")

        height = image_rgb.shape[0]
        crop = image_rgb[int(height * self.crop_top_ratio) :, :, :]
        # OCR-B at crop scale can be near the recogniser's resolution limit, so the
        # crop is enlarged and contrast-normalised before either reader sees it.
        prepared = enhance_mrz_crop(crop)

        # Tesseract is tried first on the dedicated MRZ crop because this is a fixed
        # alphabet / fixed-line-layout recognition problem. It is still only a
        # fallback: clean documents pay no second-reader latency.
        if tesseract_available():
            candidate = self._best_candidate(self.tesseract.read_lines(prepared))
            if candidate:
                candidate.extraction_method = "mrz_crop_tesseract"
                attempts.append(("mrz_crop_tesseract", candidate))
                if _passing_checks(candidate) == 4:
                    return _with_attempt_history(
                        candidate,
                        attempts,
                        selected_method="mrz_crop_tesseract",
                    )

        # Last local fallback: re-read the crop using the configured OCR engine.
        crop_lines = self.ocr_engine.read(prepared, allowlist=MRZ_ALLOWLIST)
        reconstructed = self._reconstruct_lines(crop_lines)
        candidate = self._best_candidate(reconstructed)
        if candidate:
            candidate.extraction_method = "mrz_crop_ocr"
            attempts.append(("mrz_crop_ocr", candidate))

        if attempts:
            method, best = max(attempts, key=lambda item: _passing_checks(item[1]))
            best.extraction_method = method
            return _with_attempt_history(best, attempts, selected_method=method)

        return MRZData(
            raw_lines=reconstructed,
            extraction_method="mrz_crop_ocr",
            attempts=[],
            error=(
                "A valid three-line TD1 MRZ could not be reconstructed from OCR output."
                if tesseract_available()
                else "A valid three-line TD1 MRZ could not be reconstructed. Tesseract is "
                "not installed, so only the general OCR engine was available for the MRZ."
            ),
        )

    @staticmethod
    def _looks_like_mrz(value: str) -> bool:
        """Decide whether a normalised line could be part of a TD1 MRZ.

        The previous predicate required a `<` in every line. TD1 line 1 on an Emirates ID
        carries the 15-digit national number in optional data 1, which fills the field
        exactly and leaves no filler, so a correctly read line 1 contains no `<` at all
        and was discarded before parsing. With line 1 gone the window never reached three
        lines and no MRZ could ever be assembled, regardless of OCR quality.

        Membership of the ICAO character set plus a plausible length is the real signal.
        """

        if len(value) < 18:
            return False
        return bool(re.fullmatch(r"[A-Z0-9<]+", value))

    @staticmethod
    def _candidates_from_text(text: str) -> list[str]:
        candidates: list[str] = []
        for raw in text.splitlines():
            cleaned = re.sub(r"\s+", "", raw.upper())
            if MRZExtractor._looks_like_mrz(cleaned):
                candidates.append(cleaned)
        return candidates

    @staticmethod
    def _reconstruct_lines(lines: list[OCRLine]) -> list[str]:
        # OCR engines can split one MRZ row into several boxes. Group boxes with close
        # vertical centers and then join them left-to-right before attempting TD1 parse.
        if not lines:
            return []

        grouped: list[list[OCRLine]] = []
        for line in sorted(lines, key=_vertical_center):
            center = _vertical_center(line)
            if not grouped or abs(center - _vertical_center(grouped[-1][0])) > 18:
                grouped.append([line])
            else:
                grouped[-1].append(line)

        output: list[str] = []
        for group in grouped:
            group.sort(key=_left_edge)
            value = "".join(item.text for item in group)
            value = re.sub(r"\s+", "", value.upper())
            if MRZExtractor._looks_like_mrz(value):
                output.append(value)
        return output

    @staticmethod
    def _best_candidate(lines: list[str]) -> MRZData | None:
        """Parse the strongest TD1 candidate, repairing OCR damage if a direct parse fails.

        A direct parse is attempted first so a clean read costs nothing extra. Repair is
        the fallback, and it is a fallback rather than the default because a
        reconstruction that changed characters is weaker evidence than one that did not,
        even when both satisfy the same check digits.
        """

        if len(lines) < 3:
            return None

        best: MRZData | None = None
        best_score = -1
        for start in range(len(lines) - 2):
            parsed = parse_td1(lines[start : start + 3])
            if not parsed.valid_structure:
                continue
            checks = [
                parsed.checks.document_number,
                parsed.checks.date_of_birth,
                parsed.checks.expiry_date,
                parsed.checks.composite,
            ]
            score = sum(value is True for value in checks)
            if score > best_score:
                best = parsed
                best_score = score

        if best is not None and best_score >= 2:
            return best

        repaired = repair_td1(lines)
        if repaired is None:
            return best

        candidate = parse_td1(repaired.lines)
        if not candidate.valid_structure:
            return best

        repaired_score = sum(
            value is True
            for value in (
                candidate.checks.document_number,
                candidate.checks.date_of_birth,
                candidate.checks.expiry_date,
                candidate.checks.composite,
            )
        )
        if repaired_score <= best_score:
            return best

        # Record that characters were altered so the reviewer sees reconstructed values
        # as reconstructed rather than as a clean read.
        candidate.extraction_method = "repaired"
        candidate.error = (
            None
            if repaired.repairs_applied == 0
            else f"MRZ reconstructed with {repaired.repairs_applied} character repair(s); "
            f"{repaired_score}/{repaired.checks_available} check digits pass."
        )
        return candidate


def _with_attempt_history(
    selected: MRZData,
    attempts: list[tuple[str, MRZData]],
    *,
    selected_method: str,
) -> MRZData:
    """Attach a compact audit trail of MRZ reads to the selected result."""

    selected.attempts = [
        MRZAttempt(
            method=method,
            raw_lines=list(data.raw_lines),
            valid_structure=data.valid_structure,
            checks=data.checks.model_copy(deep=True),
            passing_checks=_passing_checks(data),
            selected=method == selected_method,
            error=data.error,
        )
        for method, data in attempts
    ]
    return selected


def _passing_checks(data: MRZData) -> int:
    """Number of ICAO check digits that verify, used to choose between readers."""

    return sum(
        value is True
        for value in (
            data.checks.document_number,
            data.checks.date_of_birth,
            data.checks.expiry_date,
            data.checks.composite,
        )
    )


def _vertical_center(line: OCRLine) -> float:
    if not line.bbox:
        return 0.0
    ys = [point[1] for point in line.bbox]
    return sum(ys) / len(ys)


def _left_edge(line: OCRLine) -> float:
    if not line.bbox:
        return 0.0
    return min(point[0] for point in line.bbox)
