from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "vehicle-detection"
    assert data["model_name"] == "fasterrcnn_resnet50_fpn"
    assert data["num_classes"] > 0