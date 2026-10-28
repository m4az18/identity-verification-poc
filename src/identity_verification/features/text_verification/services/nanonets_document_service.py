from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from identity_verification.features.text_verification.config import Settings
from identity_verification.features.text_verification.domain.models import (
    DocumentExtraction,
    FieldValue,
    IdentityRecord,
    MRZAttempt,
    MRZData,
    OCRLine,
    PageSummary,
)
from identity_verification.features.text_verification.extraction.field_extractor import (
    VisibleFieldExtractor,
)
from identity_verification.features.text_verification.extraction.nanonets_ocr2 import (
    NanonetsOCR2Extractor,
    NanonetsPageExtraction,
)
from identity_verification.features.text_verification.input.loader import (
    DocumentInput,
    DocumentLoader,
    PreparedPage,
)
from identity_verification.features.text_verification.mrz.parser import parse_td1
from identity_verification.features.text_verification.mrz.repair import repair_td1
from identity_verification.features.text_verification.normalization.values import (
    normalize_arabic_name,
    normalize_date,
    normalize_english_name,
    normalize_id_number,
    normalize_nationality,
    normalize_sex,
)
from identity_verification.features.text_verification.services.document_service import (
    DocumentService,
)
from identity_verification.features.text_verification.services.mrz_backfill import backfill_from_mrz

_NORMALIZERS = {
    "id_number": normalize_id_number,
    "name_en": normalize_english_name,
    "name_ar": normalize_arabic_name,
    "date_of_birth": normalize_date,
    "nationality": normalize_nationality,
    "sex": normalize_sex,
    "expiry_date": normalize_date,
}


@dataclass(slots=True)
class _ProcessedNanonetsPage:
    prepared: PreparedPage
    page_result: NanonetsPageExtraction | None
    text_record: IdentityRecord | None
    mrz_lines: list[str]
    raw_output: str
    source: str
    warnings: list[str]


class NanonetsDocumentService:
    """Document-level adapter for the Nanonets OCR2 document VLM.

    This backend intentionally does not masquerade as a traditional ``OCREngine``.
    Nanonets performs structured document extraction natively; its output is mapped to
    the application's canonical ``IdentityRecord`` and then passed through the exact
    same deterministic normalization, TD1 checksum validation, historical/current
    matching and decision logic used by the local OCR pipeline.
    """

    def __init__(self, settings: Settings, extractor: NanonetsOCR2Extractor) -> None:
        self.settings = settings
        self.extractor = extractor
        # Compatibility with VerificationService/lifecycle logging. The object exposes
        # the same name/warm_up surface as an OCR engine, but extraction happens here at
        # the document level rather than through OCRLine.
        self.ocr_engine = extractor
        self.loader = DocumentLoader(settings)
        self.field_extractor = VisibleFieldExtractor()

    def extract(self, document: DocumentInput) -> DocumentExtraction:
        total_start = perf_counter()
        timings: dict[str, float] = {}

        stage_start = perf_counter()
        prepared_pages = self.loader.prepare_document(document)
        timings["document_preparation"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        processed_pages = [self._process_page(page) for page in prepared_pages]
        timings["visible_text_extraction"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        record, merge_warnings = self._merge_records(processed_pages)
        timings["field_mapping_and_normalization"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        mrz = self._select_mrz(processed_pages)
        timings["mrz_processing"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        backfilled = backfill_from_mrz(record, mrz)
        internal_checks = DocumentService._cross_check(record, mrz)
        internal_checks["fields_sourced_from_mrz"] = backfilled
        timings["internal_validation"] = _elapsed_ms(stage_start)

        warnings = list(merge_warnings)
        page_summaries: list[PageSummary] = []
        raw_text: dict[str, str] | None = {} if self.settings.debug_include_raw_text else None

        for page in processed_pages:
            warnings.extend(page.prepared.warnings)
            warnings.extend(page.warnings)
            if page.prepared.diagnostics:
                warnings.extend(page.prepared.diagnostics.warnings)
            page_summaries.append(
                PageSummary(
                    label=page.prepared.label,
                    source_type=page.prepared.source_type,
                    ocr_source=page.source,
                    diagnostics=page.prepared.diagnostics,
                )
            )
            if raw_text is not None:
                raw_text[page.prepared.label] = page.raw_output

        if len(processed_pages) == 1:
            warnings.append("Only one side/page was provided. Back-side MRZ validation may be unavailable.")
        if mrz.error:
            warnings.append(f"MRZ: {mrz.error}")

        timings["total"] = _elapsed_ms(total_start)
        return DocumentExtraction(
            filenames=[item.filename for item in document.files],
            page_count=len(processed_pages),
            record=record,
            mrz=mrz,
            internal_checks=internal_checks,
            warnings=_deduplicate(warnings),
            pages=page_summaries,
            timing_ms={key: round(value, 2) for key, value in timings.items()},
            ocr_engine=self.extractor.name,
            raw_text=raw_text,
        )

    def _process_page(self, prepared: PreparedPage) -> _ProcessedNanonetsPage:
        if prepared.text_layer:
            lines = [
                OCRLine(text=line.strip(), confidence=1.0) for line in prepared.text_layer.splitlines() if line.strip()
            ]
            record = self.field_extractor.extract(
                lines,
                source=f"visible:{prepared.label}:pdf_text",
            )
            return _ProcessedNanonetsPage(
                prepared=prepared,
                page_result=None,
                text_record=record,
                mrz_lines=[line.text for line in lines if "<" in line.text],
                raw_output=prepared.text_layer,
                source="pdf_text",
                warnings=[],
            )

        if prepared.image_rgb is None:
            return _ProcessedNanonetsPage(
                prepared=prepared,
                page_result=None,
                text_record=IdentityRecord(),
                mrz_lines=[],
                raw_output="",
                source="none",
                warnings=["No image data was available for Nanonets OCR2 extraction."],
            )

        page_result = self.extractor.extract_page(prepared.image_rgb)
        return _ProcessedNanonetsPage(
            prepared=prepared,
            page_result=page_result,
            text_record=None,
            mrz_lines=page_result.mrz_lines,
            raw_output=page_result.raw_output,
            source=self.extractor.name,
            warnings=page_result.warnings,
        )

    @staticmethod
    def _merge_records(
        pages: list[_ProcessedNanonetsPage],
    ) -> tuple[IdentityRecord, list[str]]:
        merged = IdentityRecord()
        warnings: list[str] = []
        page_records: list[IdentityRecord] = []
        for page in pages:
            if page.text_record is not None:
                page_records.append(page.text_record)
            elif page.page_result is not None:
                page_records.append(
                    _record_from_nanonets(
                        page.page_result,
                        page.prepared.label,
                        page.source,
                    )
                )

        for field_name in _NORMALIZERS:
            candidates: list[FieldValue] = []
            for page_record in page_records:
                value: FieldValue = getattr(page_record, field_name)
                if value.raw or value.normalized:
                    candidates.append(value)

            valid = [value for value in candidates if value.normalized]
            if valid:
                chosen = valid[0].model_copy(deep=True)
                for candidate in valid[1:]:
                    if candidate.normalized == chosen.normalized:
                        for source in candidate.sources:
                            if source not in chosen.sources:
                                chosen.sources.append(source)
                    else:
                        warnings.append(
                            f"Nanonets OCR2 returned conflicting {field_name.replace('_', ' ')} "
                            "values across pages; the first valid page value was retained for review."
                        )
                setattr(merged, field_name, chosen)
            elif candidates:
                # Keep the first raw unresolved value for traceability while ensuring it
                # cannot be treated as a canonical match downstream.
                setattr(merged, field_name, candidates[0].model_copy(deep=True))

        return merged, warnings

    @staticmethod
    def _select_mrz(pages: list[_ProcessedNanonetsPage]) -> MRZData:
        candidates: list[MRZData] = []

        for page in pages:
            if not page.mrz_lines:
                continue

            direct = parse_td1(page.mrz_lines)
            direct.extraction_method = "nanonets_ocr2"
            direct_attempt = _attempt("nanonets_ocr2", direct, selected=False)
            direct.attempts = [direct_attempt]
            candidates.append(direct)

            if direct.valid_structure and direct.checks.all_available_pass is True:
                direct.attempts[0].selected = True
                return direct

            repaired = repair_td1(page.mrz_lines)
            if repaired:
                repaired_data = parse_td1(repaired.lines)
                repaired_data.extraction_method = "repaired"
                repaired_data.attempts = [
                    direct_attempt.model_copy(deep=True),
                    _attempt("repaired", repaired_data, selected=False),
                ]
                candidates.append(repaired_data)
                if repaired_data.valid_structure and repaired_data.checks.all_available_pass is True:
                    repaired_data.attempts[-1].selected = True
                    return repaired_data

        if not candidates:
            return MRZData(error="Nanonets OCR2 did not extract a usable TD1 MRZ candidate.")

        selected = max(
            candidates,
            key=lambda data: (
                int(data.valid_structure),
                _passing_checks(data),
            ),
        )
        if selected.attempts:
            for attempt in selected.attempts:
                attempt.selected = attempt.method == selected.extraction_method
        if not selected.error and selected.checks.all_available_pass is False:
            selected.error = (
                "Nanonets OCR2 extracted an MRZ candidate, but at least one available TD1 check digit did not validate."
            )
        return selected


def _record_from_nanonets(
    page: NanonetsPageExtraction,
    page_label: str,
    backend_name: str,
) -> IdentityRecord:
    record = IdentityRecord()
    source = f"structured:{page_label}:{backend_name}"

    for field_name, normalizer in _NORMALIZERS.items():
        raw = page.fields.get(field_name)
        if not raw:
            continue
        setattr(
            record,
            field_name,
            FieldValue(
                raw=raw,
                normalized=normalizer(raw),
                # OCR2 does not expose a calibrated field-level confidence comparable
                # to Paddle/EasyOCR recognition confidence, so do not invent one.
                confidence=None,
                sources=[source],
            ),
        )
    return record


def _attempt(method: str, data: MRZData, *, selected: bool) -> MRZAttempt:
    return MRZAttempt(
        method=method,
        raw_lines=list(data.raw_lines),
        valid_structure=data.valid_structure,
        checks=data.checks,
        passing_checks=_passing_checks(data),
        selected=selected,
        error=data.error,
    )


def _passing_checks(data: MRZData) -> int:
    return sum(
        value is True
        for value in (
            data.checks.document_number,
            data.checks.date_of_birth,
            data.checks.expiry_date,
            data.checks.composite,
        )
    )


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
