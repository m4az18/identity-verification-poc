MRZ_WEIGHTS = (7, 3, 1)


def character_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    if character == "<":
        return 0
    raise ValueError(f"Unsupported MRZ character: {character!r}")


def calculate_check_digit(value: str) -> str:
    """Calculate an ICAO 9303 MRZ check digit using the repeating 7-3-1 weights."""

    total = sum(
        character_value(character) * MRZ_WEIGHTS[index % len(MRZ_WEIGHTS)] for index, character in enumerate(value)
    )
    return str(total % 10)


def check_digit_matches(value: str, expected: str) -> bool | None:
    if expected == "<":
        return None
    if not expected.isdigit():
        return False
    try:
        return calculate_check_digit(value) == expected
    except ValueError:
        return False
