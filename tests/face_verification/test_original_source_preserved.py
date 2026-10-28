import hashlib
from pathlib import Path

ORIGINAL = Path(__file__).resolve().parents[2] / "src/identity_verification/features/face_verification/original"
EXPECTED = {
    "app.py": "3f1f4f4ffd419a5c7efe1d95ba62b8587b299512c21eaee98ff1c6852fa6f9e2",
    "config.py": "9049ccaa311d60c8809aed180428ab2cc928b179ae813196bca6e451a98d780f",
    "evaluate.py": "029d20d8bb6cb54361cd4fafd0cda67990a8e04bdbb347171c0caf703bfe0ea7",
    "prepare_dataset.py": "86cd5151af9af345fe3cddb1839b37fe114e408657a1afdd74115f320a470fde",
    "quality.py": "0dfc16a2ee4843464566b8eb4261211ead490e35878442f081ad1939877a345f",
    "verification.py": "612e6f871cb698f34296843142f00684d29fa27a6affdf73cb234de569568d86",
    "webcam.py": "610221f539744e2df79b3b79f7dbc815cc49a23f77500fa02025e6be517a5132",
    "__init__.py": "c61b489be7f20fb5e399ffa764d7aab1cb3a29e9f52cb8663d944e9b73fda28f",
}


def test_face_sources_match_publication_baseline():
    # Baseline follows the documented package rename, rather than the private source archive.
    actual = {
        name: hashlib.sha256((ORIGINAL / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in EXPECTED
    }
    assert actual == EXPECTED
