from fastapi.testclient import TestClient

from identity_verification.api.main import app

client = TestClient(app)


def test_unified_ui_is_default_product_path() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Verify your identity" in response.text
    assert "Take photo &amp; verify" in response.text
    assert "not a certified liveness test" in response.text
    assert "Documents + face" in response.text
    assert 'id="historical-document"' in response.text
    assert 'id="id-photo"' in response.text
    assert "ocr-results.js" in response.text


def test_document_ui_is_simple_product_view() -> None:
    response = client.get("/text")
    assert response.status_code == 200
    assert "Document Verification" in response.text
    assert "Compare a previous Emirates ID with a current one" in response.text
    assert "View compared fields" in response.text
    assert "Technical details" in response.text
    assert "English name" in response.text
    assert "Arabic name" in response.text
    assert "Nationality" in response.text
    assert "Sex" in response.text
    assert "Extraction model" in response.text
    assert 'value="easyocr"' in response.text
    assert "Open engineering diagnostics" not in response.text
    assert "Exact verification workflow used by this POC" not in response.text
    assert 'href="/"' in response.text
    assert 'href="/face"' in response.text


def test_engineering_diagnostics_remain_available_but_are_clearly_separate() -> None:
    response = client.get("/text/diagnostics")
    assert response.status_code == 200
    assert "Document Diagnostics" in response.text
    assert "Engineering evidence view" in response.text
    assert "Exact verification workflow used by this POC" in response.text
    assert 'href="/"' in response.text
    assert 'href="/text"' in response.text
    assert 'href="/face"' in response.text


def test_face_ui_has_consistent_navigation() -> None:
    response = client.get("/face")
    assert response.status_code == 200
    assert "Face Verification" in response.text
    assert 'href="/"' in response.text
    assert 'href="/text"' in response.text
    assert 'aria-current="page" href="/face"' in response.text


def test_feature_health_routes_do_not_require_face_model_loading() -> None:
    assert client.get("/api/v1/identity/health").json()["status"] == "ok"
    face = client.get("/api/v1/face/health").json()
    assert face["status"] == "ok"
    assert face["model_loading"] == "lazy"
