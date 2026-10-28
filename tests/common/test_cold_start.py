from fastapi.testclient import TestClient

import identity_verification.features.text_verification.lifecycle as lifecycle
from identity_verification.api.main import app
from identity_verification.features.text_verification.config import Settings


def test_default_startup_does_not_initialize_ocr(monkeypatch):
    settings = Settings(_env_file=None)
    assert settings.warm_up_ocr_on_startup is False
    monkeypatch.setattr(lifecycle, "get_settings", lambda: settings)

    def unexpected_load():
        raise AssertionError("Cold start must not load the OCR service")

    monkeypatch.setattr(lifecycle, "get_verification_service", unexpected_load)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
