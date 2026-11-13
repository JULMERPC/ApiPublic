"""API tests using an isolated temporary data file and model."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

SETOSA = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}
VIRGINICA = {
    "sepal_length": 6.7,
    "sepal_width": 3.0,
    "petal_length": 5.2,
    "petal_width": 2.3,
}


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(
        data_file=tmp_path / "predictions.json",
        model_file=tmp_path / "iris_model.joblib",
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_creates_record(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json=SETOSA)
    assert response.status_code == 201

    body = response.json()
    assert body["id"] == 1
    assert body["predicted_label"] == "setosa"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["sepal_length"] == SETOSA["sepal_length"]


def test_list_predictions_newest_first(client: TestClient) -> None:
    first = client.post("/api/v1/predict", json=SETOSA).json()
    second = client.post("/api/v1/predict", json=VIRGINICA).json()

    response = client.get("/api/v1/predictions")
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [second["id"], first["id"]]


def test_get_prediction_and_404(client: TestClient) -> None:
    created = client.post("/api/v1/predict", json=SETOSA).json()

    response = client.get(f"/api/v1/predictions/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]

    missing = client.get("/api/v1/predictions/9999")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Predicción 9999 no encontrada"


def test_update_repredicts(client: TestClient) -> None:
    created = client.post("/api/v1/predict", json=SETOSA).json()

    response = client.put(f"/api/v1/predictions/{created['id']}", json=VIRGINICA)
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_label"] == "virginica"
    assert body["petal_length"] == VIRGINICA["petal_length"]
    assert body["created_at"] <= body["updated_at"]


def test_delete(client: TestClient) -> None:
    created = client.post("/api/v1/predict", json=SETOSA).json()

    response = client.delete(f"/api/v1/predictions/{created['id']}")
    assert response.status_code == 204
    assert client.get(f"/api/v1/predictions/{created['id']}").status_code == 404
    assert client.delete("/api/v1/predictions/9999").status_code == 404


def test_validation_errors(client: TestClient) -> None:
    missing = client.post("/api/v1/predict", json={"sepal_length": 5.1})
    assert missing.status_code == 422

    negative = client.post("/api/v1/predict", json={**SETOSA, "petal_width": -1})
    assert negative.status_code == 422


def test_openapi_exposes_documented_routes(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/api/v1/predict" in schema["paths"]
    assert "/api/v1/predictions" in schema["paths"]
    assert "/api/v1/predictions/{prediction_id}" in schema["paths"]
    assert {tag["name"] for tag in schema["tags"]} >= {"predictions", "health"}
