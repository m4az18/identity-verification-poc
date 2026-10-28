import re
from collections.abc import Callable

from identity_verification.features.text_verification.domain.models import (
    FieldValue,
    IdentityRecord,
    OCRLine,
)
from identity_verification.features.text_verification.normalization.values import (
    normalize_arabic_name,
    normalize_date,
    normalize_english_name,
    normalize_id_number,
    normalize_nationality,
    normalize_sex,
)

ID_PATTERN = re.compile(r"(?<!\d)784(?:[-\s]?\d){12}(?!\d)")

DATE_PATTERN = re.compile(
    r"(?<!\d)(?:"
    r"\d{4}[/.-]\d{1,2}[/.-]\d{1,2}"
    r"|"
    r"\d{1,2}[/.-]\d{1,2}[/.-]\d{4}"
    r")(?!\d)"
)

ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")

ARABIC_WORD_PATTERN = re.compile(r"[\u0600-\u06FF]+")


# ---------------------------------------------------------------------------
# Known visible-field labels
# ---------------------------------------------------------------------------
#
# Labels are kept in one shared table because some Arabic labels overlap.
#
# Example:
#
#     الجنس    = sex
#     الجنسية  = nationality
#
# Because "الجنس" is a prefix of "الجنسية", independent substring searches
# could incorrectly classify a nationality label as a sex label.
#
# `_ALL_LABELS` is therefore sorted longest-first so the most specific label
# claims an OCR line first.
#
FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "date_of_birth": (
        "date of birth",
        "birth date",
        "dob",
        "تاريخ الميلاد",
        "تاريخ الولادة",
    ),
    "expiry_date": (
        "expiry date",
        "expiration date",
        "date of expiry",
        "expires",
        "تاريخ الانتهاء",
        "تاريخ انتهاء الصلاحية",
    ),
    "issue_date": (
        "issue date",
        "date of issue",
        "تاريخ الإصدار",
        "تاريخ الاصدار",
    ),
    "name_en": (
        "name",
        "full name",
        "holder name",
    ),
    "name_ar": (
        "الاسم",
        "الاسم الكامل",
    ),
    "nationality": (
        "nationality",
        "الجنسية",
    ),
    "sex": (
        "sex",
        "gender",
        "الجنس",
    ),
    "card_number": (
        "card number",
        "رقم البطاقة",
    ),
    "id_number": (
        "id number",
        "identity number",
        "رقم الهوية",
    ),
}


# These labels are recognised so they cannot accidentally be borrowed as values,
# even though the corresponding fields are not part of the identity comparison.
NON_EXTRACTED_FIELDS = frozenset(
    {
        "issue_date",
        "card_number",
    }
)


_ALL_LABELS: list[tuple[str, str]] = sorted(
    ((label, field) for field, labels in FIELD_LABELS.items() for label in labels),
    key=lambda item: len(item[0]),
    reverse=True,
)


class VisibleFieldExtractor:
    """Convert OCR detections into the canonical visible identity record.

    The extractor deliberately separates OCR from field interpretation.

    PaddleOCR provides:
        * detected text,
        * model confidence,
        * and optional geometry.

    This class determines which OCR result corresponds to which Emirates ID
    field.

    The implementation is intentionally conservative: unresolved values remain
    missing rather than being guessed. This prevents the same OCR mistake on the
    historical and current documents from being incorrectly treated as a match.
    """

    def extract(
        self,
        lines: list[OCRLine],
        source: str,
    ) -> IdentityRecord:
        """Map OCR/PDF text into the canonical identity schema."""

        record = IdentityRecord()

        full_text = "\n".join(line.text for line in lines)

        # ------------------------------------------------------------------
        # Emirates ID number
        # ------------------------------------------------------------------
        #
        # The Emirates ID has a strong structural pattern:
        #
        #     784 + twelve further digits
        #
        # Therefore it can safely be found across the complete OCR output
        # without relying on label ordering.
        #
        id_match = ID_PATTERN.search(full_text)

        if id_match:
            record.id_number = self._field(
                id_match.group(0),
                normalize_id_number,
                source,
                confidence=self._confidence_for_text(
                    lines,
                    id_match.group(0),
                ),
            )

        # Determine which OCR detections contain known field labels.
        claims = self._claim_lines(lines)

        extractors: dict[
            str,
            tuple[
                Callable[[str], str | None],
                Callable[[str | None], str | None],
            ],
        ] = {
            "date_of_birth": (
                self._extract_date,
                normalize_date,
            ),
            "expiry_date": (
                self._extract_date,
                normalize_date,
            ),
            "name_en": (
                self._extract_english_text,
                normalize_english_name,
            ),
            "name_ar": (
                self._extract_arabic_text,
                normalize_arabic_name,
            ),
            "nationality": (
                self._extract_general_text,
                normalize_nationality,
            ),
            "sex": (
                self._extract_sex_token,
                normalize_sex,
            ),
        }

        for field, (
            value_extractor,
            normalizer,
        ) in extractors.items():
            found = self._value_for_field(
                lines,
                claims,
                field,
                value_extractor,
                normalizer,
            )

            if found:
                setattr(
                    record,
                    field,
                    self._field(
                        found[0],
                        normalizer,
                        source,
                        found[1],
                    ),
                )

        # ------------------------------------------------------------------
        # Arabic-name fallback
        # ------------------------------------------------------------------
        #
        # PaddleOCR does not guarantee human reading order, especially with
        # mixed Arabic/English text.
        #
        # On a real Emirates ID it may return:
        #
        #     احمد زهير احمد
        #     Name: Samir Example Person
        #
        # rather than:
        #
        #     الاسم
        #     احمد زهير احمد
        #
        # Therefore a label-only Arabic-name extractor misses a value that OCR
        # actually recognised correctly.
        #
        # If no Arabic name was resolved through the normal labelled path, look
        # for a plausible Arabic person-name close to the English-name region.
        #
        if not record.name_ar.normalized:
            arabic_name = self._find_arabic_name_near_english_name(
                lines,
                claims,
            )

            if arabic_name:
                record.name_ar = self._field(
                    arabic_name[0],
                    normalize_arabic_name,
                    source,
                    arabic_name[1],
                )

        return record

    # ======================================================================
    # LABEL CLAIMING
    # ======================================================================

    @staticmethod
    def _claim_lines(
        lines: list[OCRLine],
    ) -> dict[int, tuple[str, str]]:
        """Assign each OCR line to at most one known field label.

        Returns:

            {
                line_index: (field_name, matched_label)
            }

        Labels are checked longest-first so overlapping Arabic labels cannot
        claim the same OCR line incorrectly.
        """

        claims: dict[
            int,
            tuple[str, str],
        ] = {}

        for index, line in enumerate(lines):
            haystack = VisibleFieldExtractor._normalize_label_text(line.text)

            for label, field in _ALL_LABELS:
                normalized_label = VisibleFieldExtractor._normalize_label_text(label)

                if normalized_label in haystack:
                    claims[index] = (
                        field,
                        label,
                    )
                    break

        return claims

    # ======================================================================
    # GENERIC FIELD EXTRACTION
    # ======================================================================

    def _value_for_field(
        self,
        lines: list[OCRLine],
        claims: dict[int, tuple[str, str]],
        field: str,
        extractor: Callable[[str], str | None],
        normalizer: Callable[[str | None], str | None],
    ) -> tuple[str, float | None] | None:
        """Resolve one visible field from OCR detections.

        Most labelled Emirates ID fields are extracted conservatively:

            label
              ↓
            inline value, or
            one of the next few unlabelled OCR detections

        Sex is handled slightly differently because PaddleOCR frequently returns
        the single-character value immediately *before* the ``Sex`` label.

        Example observed from the actual Emirates ID OCR:

            M
            Sex

        Since the sex domain is strictly limited to M/F (or supported Arabic
        equivalents), looking immediately around the label remains conservative.
        """

        for index, (
            claimed_field,
            _label,
        ) in claims.items():
            if claimed_field != field:
                continue

            line = lines[index]

            # --------------------------------------------------------------
            # 1. Try a value contained in the same OCR detection as the label.
            # --------------------------------------------------------------
            #
            # Examples:
            #
            #     Nationality الجنسية SYRIAN
            #     Sex الجنس M
            #
            inline = self._strip_field_labels(
                line.text,
                field,
            ).strip(" :|-")

            value = extractor(inline)

            if value and normalizer(value):
                return (
                    value,
                    line.confidence,
                )

            # --------------------------------------------------------------
            # 2. Sex-specific nearby search
            # --------------------------------------------------------------
            #
            # A single M/F may appear directly before the label in Paddle's
            # reading order. Searching both directions is safe here because
            # normalize_sex() only accepts the very small legitimate sex domain.
            #
            if field == "sex":
                sex_candidate = self._find_sex_near_label(
                    lines,
                    claims,
                    index,
                    extractor,
                    normalizer,
                )

                if sex_candidate:
                    return sex_candidate

                continue

            # --------------------------------------------------------------
            # 3. Standard forward label/value relationship
            # --------------------------------------------------------------
            #
            # For dates, nationality and names we remain conservative and keep
            # the original forward-only search. We intentionally stop at another
            # labelled line so, for example, DOB cannot accidentally borrow an
            # issue/expiry date belonging to another field.
            #
            for offset in range(1, 4):
                candidate_index = index + offset

                if candidate_index >= len(lines):
                    break

                if candidate_index in claims:
                    break

                candidate_line = lines[candidate_index]

                value = extractor(candidate_line.text)

                if value and normalizer(value):
                    return (
                        value,
                        candidate_line.confidence,
                    )

        return None

    # ======================================================================
    # SEX FALLBACK
    # ======================================================================

    @staticmethod
    def _find_sex_near_label(
        lines: list[OCRLine],
        claims: dict[int, tuple[str, str]],
        label_index: int,
        extractor: Callable[[str], str | None],
        normalizer: Callable[[str | None], str | None],
    ) -> tuple[str, float | None] | None:
        """Find a valid sex token immediately around the Sex/Gender label.

        PaddleOCR may order the card as::

            M
            Sex

        rather than::

            Sex
            M

        Search nearest detections first:

            -1, +1, -2, +2, -3, +3

        A candidate must successfully normalize to M or F, so unrelated nearby
        OCR text is never accepted as sex.
        """

        offsets = (
            -1,
            1,
            -2,
            2,
            -3,
            3,
        )

        for offset in offsets:
            candidate_index = label_index + offset

            if not (0 <= candidate_index < len(lines)):
                continue

            # Never treat another recognised field label itself as the value.
            if candidate_index in claims:
                continue

            candidate_line = lines[candidate_index]

            value = extractor(candidate_line.text)

            if not value:
                continue

            if normalizer(value):
                return (
                    value,
                    candidate_line.confidence,
                )

        return None

    # ======================================================================
    # ARABIC-NAME FALLBACK
    # ======================================================================

    def _find_arabic_name_near_english_name(
        self,
        lines: list[OCRLine],
        claims: dict[int, tuple[str, str]],
    ) -> tuple[str, float | None] | None:
        """Find an Arabic name near the English name region.

        Mixed RTL/LTR OCR does not guarantee that an Arabic ``الاسم`` label will
        be detected or that the Arabic name will appear after it.

        The actual Paddle result for the test card contains::

            احمد زهير احمد
            Name: Samir Example Person

        This fallback therefore examines a small neighbourhood around the
        English-name detection and accepts only Arabic text that looks like a
        multi-token person name.

        It does not search the whole page indiscriminately; this prevents Arabic
        authority headings, nationality text and other labels from being treated
        as the holder's name.
        """

        english_name_indices = [
            index
            for index, (
                claimed_field,
                _label,
            ) in claims.items()
            if claimed_field == "name_en"
        ]

        if not english_name_indices:
            return None

        for name_index in english_name_indices:
            # The Arabic name is commonly next to or immediately above/below
            # the English holder name. Check the closest OCR detections first.
            offsets = (
                -1,
                1,
                -2,
                2,
                -3,
                3,
            )

            for offset in offsets:
                candidate_index = name_index + offset

                if not (0 <= candidate_index < len(lines)):
                    continue

                # Skip known labels. We want an actual value.
                if candidate_index in claims:
                    continue

                candidate_line = lines[candidate_index]

                candidate = self._extract_arabic_name_candidate(candidate_line.text)

                if not candidate:
                    continue

                # Do not misclassify an Arabic nationality or sex value as a name.
                if normalize_nationality(candidate):
                    continue

                if normalize_sex(candidate):
                    continue

                normalized = normalize_arabic_name(candidate)

                if normalized:
                    return (
                        candidate,
                        candidate_line.confidence,
                    )

        return None

    # ======================================================================
    # LABEL CLEANING
    # ======================================================================

    @staticmethod
    def _strip_field_labels(
        value: str,
        field: str,
    ) -> str:
        """Remove all English/Arabic aliases belonging to a field.

        Emirates IDs often print both language labels together:

            Nationality الجنسية SYRIAN

        Removing only ``Nationality`` would leave ``الجنسية SYRIAN`` behind.
        Removing every known alias for the already-selected field preserves only
        the candidate value.
        """

        cleaned = value

        for label in sorted(
            FIELD_LABELS.get(
                field,
                (),
            ),
            key=len,
            reverse=True,
        ):
            cleaned = re.sub(
                re.escape(label),
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        return cleaned

    # ======================================================================
    # FIELD CONSTRUCTION
    # ======================================================================

    @staticmethod
    def _field(
        raw: str,
        normalizer: Callable[
            [str | None],
            str | None,
        ],
        source: str,
        confidence: float | None = None,
    ) -> FieldValue:
        """Build one auditable FieldValue."""

        return FieldValue(
            raw=raw.strip(),
            normalized=normalizer(raw),
            confidence=confidence,
            sources=[source],
        )

    # ======================================================================
    # TEXT HELPERS
    # ======================================================================

    @staticmethod
    def _normalize_label_text(
        value: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            value.casefold(),
        ).strip()

    @staticmethod
    def _extract_date(
        value: str,
    ) -> str | None:

        match = DATE_PATTERN.search(value)

        return match.group(0) if match else None

    @staticmethod
    def _extract_english_text(
        value: str,
    ) -> str | None:

        cleaned = value.strip(" :|-")

        if len(cleaned) < 3:
            return None

        if ARABIC_PATTERN.search(cleaned):
            return None

        if not any(char.isalpha() for char in cleaned):
            return None

        return cleaned

    @staticmethod
    def _extract_arabic_text(
        value: str,
    ) -> str | None:

        cleaned = value.strip(" :|-")

        return cleaned if ARABIC_PATTERN.search(cleaned) else None

    @staticmethod
    def _extract_arabic_name_candidate(
        value: str,
    ) -> str | None:
        """Return Arabic text only when it plausibly resembles a person's name.

        A name should contain at least two Arabic word tokens and no digits.

        This intentionally rejects:
            * ``M``
            * dates,
            * ``الجنس``,
            * short one-word field values,
            * MRZ text.
        """

        cleaned = value.strip(" :|-")

        if not cleaned:
            return None

        if any(char.isdigit() for char in cleaned):
            return None

        arabic_words = ARABIC_WORD_PATTERN.findall(cleaned)

        if len(arabic_words) < 2:
            return None

        # Require the candidate to be predominantly Arabic rather than an
        # English/Arabic mixed field label.
        arabic_characters = sum(len(word) for word in arabic_words)

        letters = sum(char.isalpha() for char in cleaned)

        if not letters:
            return None

        if arabic_characters / letters < 0.70:
            return None

        return cleaned

    @staticmethod
    def _extract_general_text(
        value: str,
    ) -> str | None:

        cleaned = value.strip(" :|-")

        return cleaned if cleaned else None

    @staticmethod
    def _extract_sex_token(
        value: str,
    ) -> str | None:
        """Return only a token that could plausibly represent sex.

        Actual acceptance is performed by ``normalize_sex``. This helper only
        isolates short candidate tokens before normalization.
        """

        cleaned = value.strip(" :|-")

        for token in re.split(
            r"[\s/|,]+",
            cleaned,
        ):
            token = token.strip()

            if not token:
                continue

            if len(token) <= 6 and (token.isalpha() or ARABIC_PATTERN.search(token)):
                return token

        return None

    @staticmethod
    def _confidence_for_text(
        lines: list[OCRLine],
        needle: str,
    ) -> float | None:

        compact_needle = re.sub(
            r"\s+",
            "",
            needle,
        )

        for line in lines:
            compact_line = re.sub(
                r"\s+",
                "",
                line.text,
            )

            if compact_needle in compact_line:
                return line.confidence

        return None
