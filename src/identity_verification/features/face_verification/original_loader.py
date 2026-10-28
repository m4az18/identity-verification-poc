"""Compatibility loader for the preserved face-verification source.

The face feature arrived as a flat standalone Python application whose modules use
imports such as ``from config import ...``. The files under ``original/`` are kept
byte-for-byte unchanged. This loader creates a narrow compatibility boundary so the
original implementation can run inside the package without rewriting its code.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import ModuleType

_ORIGINAL_DIR = Path(__file__).resolve().parent / "original"
_IMPORT_LOCK = Lock()
_MISSING = object()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load preserved face module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_original_face_modules() -> tuple[ModuleType, ModuleType]:
    """Return the preserved ``verification`` and ``config`` modules.

    Temporary top-level aliases are installed only while importing because the
    original standalone files intentionally remain unchanged. Existing aliases are
    restored immediately afterward to avoid leaking compatibility names globally.
    """

    with _IMPORT_LOCK:
        previous_config = sys.modules.get("config", _MISSING)
        previous_quality = sys.modules.get("quality", _MISSING)
        try:
            config = _load_module("config", _ORIGINAL_DIR / "config.py")
            _load_module("quality", _ORIGINAL_DIR / "quality.py")
            verification = _load_module(
                "_saal_preserved_face_verification",
                _ORIGINAL_DIR / "verification.py",
            )
        finally:
            if previous_config is _MISSING:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = previous_config
            if previous_quality is _MISSING:
                sys.modules.pop("quality", None)
            else:
                sys.modules["quality"] = previous_quality

    return verification, config
