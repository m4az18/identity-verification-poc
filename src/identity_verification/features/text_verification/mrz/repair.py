"""Repair OCR-damaged TD1 MRZ lines before parsing.

General-purpose OCR models are trained on natural scene text, not on OCR-B with `<`
filler. Three failure modes account for almost every unparseable Emirates ID MRZ:

1. filler characters are dropped entirely, so a line arrives shorter than 30;
2. filler characters are recognised as visually similar glyphs (`K`, `C`, `«`, `≪`);
3. characters are confused across the letter/digit boundary (`O`/`0`, `I`/`1`, `S`/`5`).

None of these are ambiguous once the ICAO TD1 layout is applied, because the layout
fixes which positions must be numeric, which must be alphabetic, and which are check
digits over a known span. This module uses the layout as a constraint, then uses the
check digits as an objective function to score competing reconstructions.

Repair never invents identity data: every candidate is scored against the check digits
already present in the document, and a reconstruction that cannot satisfy them is
reported as unverified rather than returned as fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product

from identity_verification.features.text_verification.mrz.checksums import calculate_check_digit

MRZ_LINE_LENGTH = 30
TD1_LINE_COUNT = 3

# Glyphs that OCR commonly emits in place of the ICAO filler character.
FILLER_GLYPHS = str.maketrans(
    {
        "«": "<",
        "‹": "<",
        "≪": "<",
        "＜": "<",
        "(": "<",
        "[": "<",
        "{": "<",
        "^": "<",
        "*": "<",
    }
)

# Confusions are directional: applied only where the TD1 layout already tells us which
# character class the position must belong to.
TO_DIGIT = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "U": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "J": "1",
    "Z": "2",
    "R": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "C": "6",
    "B": "8",
    "P": "9",
    "Y": "9",
    "<": "0",
}
TO_ALPHA = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}

# Position classes for the two structured TD1 lines (0-indexed, end-exclusive).
NUMERIC_SPANS = {
    0: [(14, 15)],  # document-number check digit
    1: [(0, 7), (8, 15), (29, 30)],  # DOB + check, expiry + check, composite check
}
ALPHA_SPANS = {
    0: [(0, 2), (2, 5)],  # document code, issuing state
    1: [(15, 18)],  # nationality
}
SEX_POSITION = (1, 7)
SEX_VALUES = ("M", "F", "<")

# Bounded search: only positions whose check digit already fails are re-examined, and
# only a handful of alternatives per position. This keeps repair deterministic and fast
# instead of turning into an unbounded brute force over 90 characters.
MAX_SUBSTITUTION_POSITIONS = 3


@dataclass(slots=True)
class RepairedMRZ:
    """One reconstructed TD1 candidate and the evidence supporting it."""

    lines: list[str]
    check_score: int
    checks_available: int
    repairs_applied: int

    @property
    def fully_verified(self) -> bool:
        return self.checks_available > 0 and self.check_score == self.checks_available


def normalize_candidate(line: str) -> str:
    """Uppercase a raw OCR line and coerce lookalike filler glyphs to `<`."""

    cleaned = re.sub(r"\s+", "", line.upper()).translate(FILLER_GLYPHS)
    return re.sub(r"[^A-Z0-9<]", "", cleaned)


MAX_MISSING_CHARACTERS = 16


def fit_to_length(line: str) -> list[str]:
    """Return plausible 30-character forms of one OCR line.

    Long lines yield every 30-character window, because the surplus is usually printed
    card text bleeding into the MRZ crop rather than MRZ content.

    Short lines are the more interesting case. OCR does not lose one filler at a time;
    it loses a whole `<` run, because a run of identical low-contrast glyphs is exactly
    what a scene-text recogniser collapses. The run can sit at the end of the line (name
    padding on line 3) or in the middle (optional data on line 2, which is followed by
    the composite check digit). Right-padding alone therefore fixes line 3 but silently
    misaligns line 2's check digit, so this returns one variant per plausible insertion
    point instead: the end, and immediately before or after each existing filler run.
    Every variant is scored by check digit afterwards, so a wrong guess loses to a right
    one rather than being adopted.
    """

    if len(line) == MRZ_LINE_LENGTH:
        return [line]

    if len(line) > MRZ_LINE_LENGTH:
        return [line[start : start + MRZ_LINE_LENGTH] for start in range(len(line) - MRZ_LINE_LENGTH + 1)]

    deficit = MRZ_LINE_LENGTH - len(line)
    if deficit > MAX_MISSING_CHARACTERS:
        # Too much is missing for reconstruction to be honest about what was read.
        return []

    filler = "<" * deficit
    insertion_points = {len(line)}
    for match in re.finditer(r"<+", line):
        insertion_points.add(match.start())
        insertion_points.add(match.end())

    return list(dict.fromkeys(line[:point] + filler + line[point:] for point in sorted(insertion_points)))


def coerce_to_layout(lines: list[str]) -> tuple[list[str], int]:
    """Force each position into the character class the TD1 layout requires.

    Returns the coerced lines and the number of characters that were changed, so the
    caller can prefer reconstructions that needed less intervention.
    """

    repaired = [list(line) for line in lines]
    changes = 0

    for line_index, spans in NUMERIC_SPANS.items():
        for start, end in spans:
            for position in range(start, end):
                character = repaired[line_index][position]
                if character.isdigit():
                    continue
                replacement = TO_DIGIT.get(character)
                if replacement:
                    repaired[line_index][position] = replacement
                    changes += 1

    for line_index, spans in ALPHA_SPANS.items():
        for start, end in spans:
            for position in range(start, end):
                character = repaired[line_index][position]
                if character.isalpha() or character == "<":
                    continue
                replacement = TO_ALPHA.get(character)
                if replacement:
                    repaired[line_index][position] = replacement
                    changes += 1

    sex_line, sex_position = SEX_POSITION
    if repaired[sex_line][sex_position] not in SEX_VALUES:
        # `F` is frequently read as `E` or `P`; `M` as `H` or `N`.
        mapped = {"E": "F", "P": "F", "R": "F", "H": "M", "N": "M", "W": "M"}.get(repaired[sex_line][sex_position])
        if mapped:
            repaired[sex_line][sex_position] = mapped
            changes += 1

    return ["".join(line) for line in repaired], changes


def _check_groups(lines: list[str]) -> list[tuple[str, str, list[int]]]:
    """Return `(value, expected_digit, absolute_positions)` for every TD1 check digit.

    Positions are absolute indices into the concatenated three lines, which lets the
    substitution search address any character that feeds a failing check.
    """

    line1, line2, _ = lines
    return [
        ("document_number", line1[5:14], line1[14], list(range(5, 14))),
        ("date_of_birth", line2[0:6], line2[6], list(range(30, 36))),
        ("expiry_date", line2[8:14], line2[14], list(range(38, 44))),
        (
            "composite",
            line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29],
            line2[29],
            list(range(5, 30)) + list(range(30, 37)) + list(range(38, 45)) + list(range(48, 59)),
        ),
    ]


def _score(lines: list[str]) -> tuple[int, int]:
    """Count passing and evaluable check digits for a candidate."""

    passing = 0
    available = 0
    for _, value, expected, _ in [
        (name, value, expected, positions) for name, value, expected, positions in _check_groups(lines)
    ]:
        if expected == "<":
            continue
        available += 1
        try:
            if calculate_check_digit(value) == expected:
                passing += 1
        except ValueError:
            continue
    return passing, available


def _alternatives(character: str, numeric: bool) -> list[str]:
    if numeric:
        mapped = TO_DIGIT.get(character)
        return [mapped] if mapped and mapped != character else []
    mapped = TO_ALPHA.get(character)
    return [mapped] if mapped and mapped != character else []


def _search_substitutions(lines: list[str]) -> tuple[list[str], int]:
    """Try small character substitutions on the spans feeding failing check digits.

    Only the document-number and date groups are searched, because the composite check
    is derived from them: repairing the parts repairs the whole. The search is capped at
    `MAX_SUBSTITUTION_POSITIONS` simultaneous edits so runtime stays bounded and the
    result stays explainable.
    """

    flat = list("".join(lines))
    best_flat = list(flat)
    best_score, available = _score(lines)
    if available and best_score == available:
        return lines, 0

    best_edits = 0
    for name, value, expected, positions in _check_groups(lines):
        if name == "composite" or expected == "<":
            continue
        try:
            if calculate_check_digit(value) == expected:
                continue
        except ValueError:
            pass

        numeric = name != "document_number"
        editable = [
            (
                position,
                _alternatives(flat[position], numeric=True) if numeric or flat[position].isalpha() else [],
            )
            for position in positions
        ]
        editable = [(position, options) for position, options in editable if options]
        if not editable:
            continue

        for edit_count in range(1, min(MAX_SUBSTITUTION_POSITIONS, len(editable)) + 1):
            improved = False
            for combination in _combinations(editable, edit_count):
                trial = list(best_flat)
                for position, replacement in combination:
                    trial[position] = replacement
                trial_lines = _split(trial)
                score, trial_available = _score(trial_lines)
                if score > best_score:
                    best_score = score
                    best_flat = trial
                    best_edits += edit_count
                    improved = True
                    if trial_available and score == trial_available:
                        return _split(best_flat), best_edits
            if improved:
                break

    return _split(best_flat), best_edits


def _combinations(editable, edit_count):
    from itertools import combinations

    for chosen in combinations(editable, edit_count):
        option_lists = [[(position, option) for option in options] for position, options in chosen]
        yield from product(*option_lists)


def _split(flat: list[str]) -> list[str]:
    joined = "".join(flat)
    return [joined[index : index + MRZ_LINE_LENGTH] for index in range(0, 90, MRZ_LINE_LENGTH)]


def repair_td1(candidates: list[str]) -> RepairedMRZ | None:
    """Return the best-scoring TD1 reconstruction from raw OCR line candidates.

    The caller keeps full control of the decision: `check_score` versus
    `checks_available` says how much of the reconstruction the document itself
    corroborates, and `repairs_applied` says how much was changed to get there.
    """

    normalized = [normalize_candidate(line) for line in candidates]
    normalized = [line for line in normalized if len(line) >= MRZ_LINE_LENGTH - MAX_MISSING_CHARACTERS]
    if len(normalized) < TD1_LINE_COUNT:
        return None

    best: RepairedMRZ | None = None
    for start in range(len(normalized) - TD1_LINE_COUNT + 1):
        window = normalized[start : start + TD1_LINE_COUNT]
        fitted = [fit_to_length(line) for line in window]
        if any(not options for options in fitted):
            continue

        for combination in product(*fitted):
            lines = list(combination)
            coerced, coercions = coerce_to_layout(lines)
            searched, edits = _search_substitutions(coerced)
            score, available = _score(searched)
            candidate = RepairedMRZ(
                lines=searched,
                check_score=score,
                checks_available=available,
                repairs_applied=coercions + edits,
            )
            if best is None or (candidate.check_score, -candidate.repairs_applied) > (
                best.check_score,
                -best.repairs_applied,
            ):
                best = candidate
            if best.fully_verified:
                return best

    return best
