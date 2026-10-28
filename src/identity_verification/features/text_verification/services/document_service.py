import base64
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter

import numpy as np
from PIL import Image

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.domain.models import (
    DocumentExtraction,
    MRZData,
    OCRLine,
    PageSummary,
)
from identity_verification.features.text_verification.extraction.base import OCREngine
from identity_verification.features.text_verification.extraction.field_extractor import (
    VisibleFieldExtractor,
)
from identity_verification.features.text_verification.input.loader import (
    DocumentInput,
    DocumentLoader,
    PreparedPage,
)
from identity_verification.features.text_verification.input.preprocess import cap_resolution
from identity_verification.features.text_verification.mrz.extractor import MRZExtractor
from identity_verification.features.text_verification.normalization.values import (
    normalize_english_name,
    normalize_nationality,
    normalize_sex,
)
from identity_verification.features.text_verification.services.mrz_backfill import (
    backfill_from_mrz,
)


@dataclass(slots=True)
class _ProcessedPage:
    prepared: PreparedPage
    lines: list[OCRLine]
    ocr_source: str
    ocr_width: int | None
    ocr_height: int | None


class DocumentService:
    """Extract and validate one logical Emirates ID document.

    One logical document may arrive as a two-page PDF or as two image files. The service
    deliberately treats extraction, MRZ validation and old/current matching as separate
    concerns so OCR engines can be benchmarked without rewriting verification rules.
    """

    def __init__(self, settings: Settings, ocr_engine: OCREngine) -> None:
        self.settings = settings
        self.ocr_engine = ocr_engine
        self.loader = DocumentLoader(settings)
        self.field_extractor = VisibleFieldExtractor()
        self.mrz_extractor = MRZExtractor(
            ocr_engine,
            settings.mrz_crop_top_ratio,
        )

    def extract(self, document: DocumentInput) -> DocumentExtraction:
        total_start = perf_counter()
        timings: dict[str, float] = {}

        stage_start = perf_counter()
        prepared_pages = self.loader.prepare_document(document)
        timings["document_preparation"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        processed_pages = [self._process_visible_page(page) for page in prepared_pages]
        timings["visible_text_extraction"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()

        # Fields can legitimately appear on either side, so both pages
        # contribute to the visible record.
        visible_lines = [line for page in processed_pages for line in page.lines]

        source = "+".join(f"{page.prepared.label}:{page.ocr_source}" for page in processed_pages)

        record = self.field_extractor.extract(
            visible_lines,
            source=f"visible:{source}",
        )

        timings["field_mapping_and_normalization"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()

        back_page, mrz = self._resolve_back_and_mrz(processed_pages)

        timings["mrz_processing"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()

        # Checksum-protected MRZ fields can independently corroborate
        # visible OCR. Backfill remains limited to missing fields.
        backfilled = backfill_from_mrz(
            record,
            mrz,
        )

        internal_checks = self._cross_check(
            record,
            mrz,
        )

        internal_checks["fields_sourced_from_mrz"] = backfilled

        timings["internal_validation"] = _elapsed_ms(stage_start)

        warnings: list[str] = []
        page_summaries: list[PageSummary] = []

        raw_text: dict[str, str] | None = {} if self.settings.debug_include_raw_text else None

        for page in processed_pages:
            warnings.extend(page.prepared.warnings)

            if page.prepared.diagnostics:
                warnings.extend(page.prepared.diagnostics.warnings)

            page_summaries.append(
                PageSummary(
                    label=page.prepared.label,
                    source_type=page.prepared.source_type,
                    ocr_source=page.ocr_source,
                    diagnostics=page.prepared.diagnostics,
                    ocr_width=(page.ocr_width if self.settings.debug_include_raw_text else None),
                    ocr_height=(page.ocr_height if self.settings.debug_include_raw_text else None),
                    ocr_lines=(page.lines if self.settings.debug_include_raw_text else None),
                    preview_data_url=None,
                )
            )

            if raw_text is not None:
                raw_text[page.prepared.label] = "\n".join(line.text for line in page.lines)

        if len(processed_pages) == 1:
            warnings.append("Only one side/page was provided. Back-side MRZ validation may be unavailable.")

        if mrz.error:
            warnings.append(f"MRZ: {mrz.error}")

        if back_page is None:
            warnings.append("A back side could not be confidently identified from MRZ evidence.")

        # Capture verification-processing time BEFORE JPEG preview encoding.
        #
        # The preview is only a debugging/UI feature, so it should not inflate
        # the OCR/model timing being compared between engines.
        timings["total"] = _elapsed_ms(total_start)

        if self.settings.debug_include_raw_text:
            for summary, page in zip(
                page_summaries,
                processed_pages,
                strict=True,
            ):
                summary.preview_data_url = _preview_data_url(page.prepared.image_rgb)

        return DocumentExtraction(
            filenames=[item.filename for item in document.files],
            page_count=len(processed_pages),
            record=record,
            mrz=mrz,
            internal_checks=internal_checks,
            warnings=_deduplicate(warnings),
            pages=page_summaries,
            timing_ms={key: round(value, 2) for key, value in timings.items()},
            ocr_engine=self.ocr_engine.name,
            raw_text=raw_text,
        )

    def _process_visible_page(
        self,
        prepared: PreparedPage,
    ) -> _ProcessedPage:

        if prepared.text_layer:
            # A real PDF text layer is already machine-readable.
            # The rendered image still remains available for preview/MRZ.
            lines = [
                OCRLine(
                    text=line.strip(),
                    confidence=1.0,
                )
                for line in prepared.text_layer.splitlines()
                if line.strip()
            ]

            width = int(prepared.image_rgb.shape[1]) if prepared.image_rgb is not None else None

            height = int(prepared.image_rgb.shape[0]) if prepared.image_rgb is not None else None

            return _ProcessedPage(
                prepared,
                lines,
                "pdf_text",
                width,
                height,
            )

        if prepared.image_rgb is None:
            return _ProcessedPage(
                prepared,
                [],
                "none",
                None,
                None,
            )

        # This is the exact image used for OCR.
        working_image = cap_resolution(prepared.image_rgb)

        lines = self.ocr_engine.read(working_image)

        return _ProcessedPage(
            prepared,
            lines,
            self.ocr_engine.name,
            int(working_image.shape[1]),
            int(working_image.shape[0]),
        )

    def _resolve_back_and_mrz(
        self,
        pages: list[_ProcessedPage],
    ) -> tuple[_ProcessedPage | None, MRZData]:

        # Reuse full-page OCR first.
        for page in pages:
            existing = self.mrz_extractor.parse_existing(
                text_layer=page.prepared.text_layer,
                ocr_lines=page.lines,
            )

            if existing and existing.valid_structure and existing.checks.all_available_pass is True:
                return page, existing

        if not pages:
            return (
                None,
                MRZData(error="No pages were available for MRZ processing."),
            )

        # Expected order is front first, back second.
        candidate = pages[1] if len(pages) >= 2 else pages[0]

        mrz = self.mrz_extractor.extract_with_fallback(
            candidate.prepared.image_rgb,
            text_layer=candidate.prepared.text_layer,
            ocr_lines=candidate.lines,
        )

        return (
            candidate if mrz.valid_structure else None,
            mrz,
        )

    @staticmethod
    def _cross_check(
        record,
        mrz,
    ) -> dict[str, object]:

        checks: dict[str, object] = {
            "mrz_structure_valid": mrz.valid_structure,
            "mrz_check_digits_pass": mrz.checks.all_available_pass,
        }

        def compare(
            name: str,
            visible,
            mrz_value,
            *,
            transform=lambda value: value,
            severity: str = "critical",
        ) -> None:

            if not visible or not mrz_value:
                return

            expected = transform(mrz_value)

            checks[name] = {
                "pass": visible == expected,
                "visible": visible,
                "mrz": expected,
                "severity": severity,
            }

        if mrz.checks.composite is True:
            compare(
                "id_number_visible_vs_mrz",
                record.id_number.normalized,
                mrz.emirates_id_candidate,
            )

        if mrz.checks.date_of_birth is True and record.date_of_birth.normalized:
            compare(
                "date_of_birth_visible_vs_mrz",
                record.date_of_birth.normalized.replace("-", "")[2:],
                mrz.date_of_birth_raw,
            )

        if mrz.checks.expiry_date is True and record.expiry_date.normalized:
            compare(
                "expiry_date_visible_vs_mrz",
                record.expiry_date.normalized.replace("-", "")[2:],
                mrz.expiry_date_raw,
            )

        compare(
            "nationality_visible_vs_mrz",
            record.nationality.normalized,
            mrz.nationality,
            transform=normalize_nationality,
            severity="supporting",
        )

        compare(
            "sex_visible_vs_mrz",
            record.sex.normalized,
            mrz.sex,
            transform=normalize_sex,
            severity="supporting",
        )

        if record.name_en.normalized and mrz.name:
            visible_tokens = set((normalize_english_name(record.name_en.normalized) or "").split())

            mrz_tokens = set((normalize_english_name(mrz.name) or "").split())

            checks["name_visible_vs_mrz"] = {
                "pass": (bool(visible_tokens) and visible_tokens == mrz_tokens),
                "visible": " ".join(sorted(visible_tokens)),
                "mrz": " ".join(sorted(mrz_tokens)),
                "severity": "supporting",
            }

        return checks


def _preview_data_url(
    image_rgb: np.ndarray | None,
) -> str | None:
    """Encode a small debug-only preview from the already-rendered page."""

    if image_rgb is None:
        return None

    image = Image.fromarray(
        image_rgb.astype(np.uint8),
        mode="RGB",
    )

    # Preserve aspect ratio while limiting response size.
    image.thumbnail((1200, 1200))

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=78,
    )

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def _deduplicate(
    values: list[str],
) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
