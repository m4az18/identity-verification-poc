"""Conservative canonicalization helpers for identity-field comparison.

Normalization removes presentation differences; it must not invent identity data. The
raw OCR value is retained separately in `FieldValue` for traceability.

Two rules govern every function here:

1. A value that cannot be resolved into the field's known domain returns `None`.
   Returning the unresolved string instead lets OCR noise reach the matcher, where two
   identically-misread values compare equal and are reported as a match. On an identity
   system that is a false accept, which is the most expensive error this POC can make.
2. Nothing is guessed. `None` routes the field to `MISSING`, which the matcher already
   treats as insufficient evidence rather than as agreement.
"""

import re
import unicodedata
from datetime import date, datetime

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

EMIRATES_ID_LENGTH = 15
EMIRATES_ID_PREFIX = "784"

# Plausibility bounds for a date read off an identity document. These reject OCR
# artefacts such as a year of 1002, not unusual-but-real people.
MIN_PLAUSIBLE_YEAR = 1900
MAX_PLAUSIBLE_YEAR = 2100


def normalize_id_number(value: str | None) -> str | None:
    """Return the canonical 15-digit Emirates ID, or `None` if the value is not one.

    Arabic-Indic digits are folded first because Emirates IDs print both digit sets and
    OCR may return either. Length and the `784` issuer prefix are structural facts about
    the number, so a value failing them was misread and must not be compared.
    """

    if not value:
        return None
    digits = re.sub(r"\D", "", value.translate(ARABIC_INDIC_DIGITS))
    if len(digits) != EMIRATES_ID_LENGTH or not digits.startswith(EMIRATES_ID_PREFIX):
        return None
    return digits


def emirates_id_check_digit_passes(value: str | None) -> bool | None:
    """Advisory Luhn check over the Emirates ID number.

    The UAE does not publish the check-digit specification. Community implementations
    apply Luhn over the first fourteen digits and match most real cards, but documented
    counter-examples exist where a number confirmed valid by ICP fails the check. This
    is therefore returned as a signal for review routing and must never be used to
    reject a document on its own. Returns `None` when the input is not a well-formed
    Emirates ID number.
    """

    canonical = normalize_id_number(value)
    if not canonical:
        return None

    payload = canonical[:-1]
    expected = int(canonical[-1])
    total = 0
    for index, character in enumerate(reversed(payload)):
        digit = int(character)
        if index % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (total + expected) % 10 == 0


def normalize_date(value: str | None) -> str | None:
    """Convert supported visible date formats to ISO `YYYY-MM-DD`.

    A parsed date must also be calendar-plausible. `strptime` happily accepts a year of
    0412 produced by a dropped leading digit, and an implausible date silently poisons
    both the comparison and the MRZ cross-check.
    """

    if not value:
        return None
    cleaned = value.strip().translate(ARABIC_INDIC_DIGITS)
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y.%m.%d"):
        try:
            parsed = datetime.strptime(cleaned, pattern).date()
        except ValueError:
            continue
        if MIN_PLAUSIBLE_YEAR <= parsed.year <= MAX_PLAUSIBLE_YEAR:
            return parsed.isoformat()
        return None
    return None


def normalize_mrz_date(value: str | None, *, reference: date | None = None) -> str | None:
    """Expand a six-digit MRZ `YYMMDD` into an ISO date.

    ICAO 9303 does not carry a century, so the window has to be chosen. A date of birth
    cannot be in the future and an expiry date is not decades past, so `reference`
    (today, injected for testability) resolves the ambiguity: a two-digit year that
    would land in the future is pushed back one century.
    """

    if not value or not re.fullmatch(r"\d{6}", value):
        return None

    today = reference or date.today()
    year_short, month, day = int(value[0:2]), int(value[2:4]), int(value[4:6])
    century = today.year - (today.year % 100)
    year = century + year_short
    if year > today.year + 20:
        year -= 100

    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_english_name(value: str | None) -> str | None:
    """Normalize casing, punctuation and spacing without reordering name tokens."""

    if not value:
        return None
    text = unicodedata.normalize("NFKC", value).upper()
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def comparison_key_english_name(value: str | None) -> str | None:
    normalized = normalize_english_name(value)
    if not normalized:
        return None
    return re.sub(r"\s+", "", normalized)


def normalize_arabic_name(value: str | None) -> str | None:
    """Apply limited Arabic orthographic normalization while retaining name content."""

    if not value:
        return None
    text = unicodedata.normalize("NFKC", value)
    text = ARABIC_DIACRITICS.sub("", text).replace("ـ", "")
    text = re.sub("[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def comparison_key_arabic_name(value: str | None) -> str | None:
    normalized = normalize_arabic_name(value)
    return normalized.replace(" ", "") if normalized else None


# ISO 3166-1 alpha-3 is the interchange format because the MRZ nationality field already
# uses it. Mapping the visible zone into the same space is what makes the visible/MRZ
# cross-check meaningful instead of a guaranteed mismatch between a printed country name
# and a three-letter code. The table covers the nationalities expected in this POC and is
# extended from observed data, not guessed at.
NATIONALITY_ALIASES = {
    "UNITED ARAB EMIRATES": "ARE",
    "U.A.E": "ARE",
    "U.A.E.": "ARE",
    "UAE": "ARE",
    "EMIRATI": "ARE",
    "ARE": "ARE",
    "الإمارات العربية المتحدة": "ARE",
    "الامارات العربية المتحدة": "ARE",
    "SYRIA": "SYR",
    "SYRIAN ARAB REPUBLIC": "SYR",
    "SYRIAN": "SYR",
    "SYR": "SYR",
    "الجمهورية العربية السورية": "SYR",
    "سوريا": "SYR",
    "سورية": "SYR",
    "سوري": "SYR",
    "EGYPT": "EGY",
    "EGYPTIAN": "EGY",
    "EGY": "EGY",
    "مصر": "EGY",
    "جمهورية مصر العربية": "EGY",
    "INDIA": "IND",
    "INDIAN": "IND",
    "IND": "IND",
    "الهند": "IND",
    "PAKISTAN": "PAK",
    "PAKISTANI": "PAK",
    "PAK": "PAK",
    "باكستان": "PAK",
    "JORDAN": "JOR",
    "JORDANIAN": "JOR",
    "JOR": "JOR",
    "الأردن": "JOR",
    "الاردن": "JOR",
    "LEBANON": "LBN",
    "LEBANESE": "LBN",
    "LBN": "LBN",
    "لبنان": "LBN",
    "PHILIPPINES": "PHL",
    "FILIPINO": "PHL",
    "PHL": "PHL",
    "الفلبين": "PHL",
    "BANGLADESH": "BGD",
    "BGD": "BGD",
    "بنغلاديش": "BGD",
    "SUDAN": "SDN",
    "SDN": "SDN",
    "السودان": "SDN",
    "YEMEN": "YEM",
    "YEM": "YEM",
    "اليمن": "YEM",
}


def normalize_nationality(value: str | None) -> str | None:
    """Resolve a printed or MRZ nationality to ISO 3166-1 alpha-3, else `None`.

    Returning `None` for an unrecognised value is deliberate. An unmapped string would
    still compare equal to the same unmapped string on the other document, producing a
    match that reflects consistent OCR noise rather than consistent identity.
    """

    if not value:
        return None

    text = re.sub(r"\s+", " ", value.strip())
    direct = NATIONALITY_ALIASES.get(text) or NATIONALITY_ALIASES.get(text.upper())
    if direct:
        return direct

    stripped = normalize_arabic_name(text)
    if stripped:
        arabic = NATIONALITY_ALIASES.get(stripped) or NATIONALITY_ALIASES.get(stripped.upper())
        if arabic:
            return arabic
        for alias, code in NATIONALITY_ALIASES.items():
            if normalize_arabic_name(alias) == stripped:
                return code

    return None


SEX_VALUES = {
    "M": "M",
    "MALE": "M",
    "ذكر": "M",
    "ذكر/M": "M",
    "M/ذكر": "M",
    "F": "F",
    "FEMALE": "F",
    "أنثى": "F",
    "انثى": "F",
    "أنثي": "F",
    "انثي": "F",
    "أنثى/F": "F",
    "انثى/F": "F",
    "F/أنثى": "F",
    "F/انثى": "F",
}


def normalize_sex(value: str | None) -> str | None:
    """Resolve a sex value to `M` or `F`, else `None`.

    The previous fall-through returned the uppercased raw string for anything
    unrecognised. Because the historical and current documents are captured under
    similar conditions, OCR tends to make the *same* mistake on both, so two identical
    unrecognised fragments reached the matcher and were reported as a match. Restricting
    the output to the field's actual domain removes that path entirely.
    """

    if not value:
        return None

    text = re.sub(r"\s+", "", value.strip())
    direct = SEX_VALUES.get(text) or SEX_VALUES.get(text.upper())
    if direct:
        return direct

    stripped = normalize_arabic_name(text)
    if stripped:
        for alias, code in SEX_VALUES.items():
            if normalize_arabic_name(alias) == stripped:
                return code
    return None
