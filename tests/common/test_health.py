from fastapi.testclient import TestClient

from identity_verification.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "identity-verification-poc",
        "version": "0.5.0",
        "decision_policy": "poc-evidence-policy-v2",
    }
