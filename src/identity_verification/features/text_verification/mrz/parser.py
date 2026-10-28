import re

from identity_verification.features.text_verification.domain.models import MRZChecks, MRZData
from identity_verification.features.text_verification.mrz.checksums import check_digit_matches

MRZ_ALLOWED = re.compile(r"^[A-Z0-9<]+$")
EMIRATES_ID_PATTERN = re.compile(r"784\d{12}")


def normalize_mrz_lines(lines: list[str]) -> list[str]:
    normalized: list[str] = []
    for line in lines:
        cleaned = re.sub(r"\s+", "", line.upper())
        cleaned = cleaned.replace("«", "<").replace("‹", "<")
        if cleaned:
            normalized.append(cleaned)
    return normalized


def parse_td1(lines: list[str]) -> MRZData:
    """Parse a three-line ICAO TD1 MRZ and recompute available check digits."""

    normalized = normalize_mrz_lines(lines)
    result = MRZData(raw_lines=normalized)

    if len(normalized) != 3:
        result.error = f"TD1 MRZ requires 3 lines; received {len(normalized)}."
        return result
    if any(len(line) != 30 for line in normalized):
        lengths = ", ".join(str(len(line)) for line in normalized)
        result.error = f"TD1 MRZ lines must be 30 characters each; received lengths {lengths}."
        return result
    if any(not MRZ_ALLOWED.fullmatch(line) for line in normalized):
        result.error = "MRZ contains characters outside the ICAO machine-readable character set."
        return result

    line1, line2, line3 = normalized

    document_number = line1[5:14]
    document_number_check = line1[14]
    optional_data_1 = line1[15:30]
    date_of_birth = line2[0:6]
    birth_check = line2[6]
    sex = line2[7]
    expiry = line2[8:14]
    expiry_check = line2[14]
    nationality = line2[15:18]
    optional_data_2 = line2[18:29]
    composite_check = line2[29]

    composite_value = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]

    optional_digits = re.sub(r"[^0-9]", "", optional_data_1 + optional_data_2)
    emirates_id_match = EMIRATES_ID_PATTERN.search(optional_digits)

    result.document_code = line1[0:2].replace("<", "").strip() or None
    result.issuing_state = line1[2:5].replace("<", "").strip() or None
    result.document_number = document_number.replace("<", "").strip() or None
    result.optional_data_1 = optional_data_1
    result.date_of_birth_raw = date_of_birth
    result.sex = None if sex == "<" else sex
    result.expiry_date_raw = expiry
    result.nationality = nationality.replace("<", "").strip() or None
    result.optional_data_2 = optional_data_2
    result.name = _parse_name(line3)
    result.emirates_id_candidate = emirates_id_match.group(0) if emirates_id_match else None
    result.checks = MRZChecks(
        document_number=check_digit_matches(document_number, document_number_check),
        date_of_birth=check_digit_matches(date_of_birth, birth_check),
        expiry_date=check_digit_matches(expiry, expiry_check),
        composite=check_digit_matches(composite_value, composite_check),
    )
    result.valid_structure = True
    return result


def _parse_name(line: str) -> str | None:
    parts = line.rstrip("<").split("<<", maxsplit=1)
    surname = parts[0].replace("<", " ").strip()
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    name = " ".join(part for part in (given, surname) if part).strip()
    return name or None
