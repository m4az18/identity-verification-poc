import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src/identity_verification/features/text_verification"
MANIFEST = Path(__file__).resolve().parents[2] / "docs/integration/TEXT_SOURCE_SHA256.txt"


def test_text_algorithms_match_publication_manifest():
    """UI/router integration may evolve; the existing text algorithms stay intact."""

    expected: dict[str, str] = {}
    for line in MANIFEST.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("./")
        # The feature branch already added document normalization in the loader.
        # This integration adds only request-scoped page reuse and startup settings.
        if relative in {"router.py", "config.py", "input/loader.py", "extraction/factory.py"} or relative.startswith(
            "web/"
        ):
            continue
        expected[relative] = digest

    actual = {
        relative: hashlib.sha256((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for relative in expected
    }
    assert actual == expected
