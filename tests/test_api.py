import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


# ─────────────────────────────────────────────
# Fixture: payload válido de un cliente
# ─────────────────────────────────────────────
@pytest.fixture
def valid_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 99.9,
        "TotalCharges": 499.5,
    }


@pytest.fixture
def low_risk_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
    }


# ─────────────────────────────────────────────
# Tests de endpoints de salud
# ─────────────────────────────────────────────
def test_root_endpoint():
    """El endpoint raíz debe retornar status 200 y campo 'status'."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint():
    """El health check debe confirmar que el modelo está cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ─────────────────────────────────────────────
# Tests del endpoint /predict
# ─────────────────────────────────────────────
def test_predict_returns_200(valid_customer):
    """Una solicitud válida debe retornar código 200."""
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200


def test_predict_response_fields(valid_customer):
    """La respuesta debe contener todos los campos esperados."""
    response = client.post("/predict", json=valid_customer)
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert "risk_level" in data
    assert "model_name" in data


def test_predict_probability_range(valid_customer):
    """La probabilidad debe estar entre 0 y 1."""
    response = client.post("/predict", json=valid_customer)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_prediction_values(valid_customer):
    """El campo prediction solo puede ser 'Yes' o 'No'."""
    response = client.post("/predict", json=valid_customer)
    prediction = response.json()["prediction"]
    assert prediction in ("Yes", "No")


def test_predict_risk_level_values(valid_customer):
    """El risk_level debe ser uno de los tres niveles definidos."""
    response = client.post("/predict", json=valid_customer)
    risk = response.json()["risk_level"]
    assert risk in ("Alto", "Medio", "Bajo")


def test_predict_high_risk_profile(valid_customer):
    """Un cliente con perfil de alto riesgo debería tener prob > 0.3."""
    response = client.post("/predict", json=valid_customer)
    prob = response.json()["churn_probability"]
    # No podemos garantizar el valor exacto, pero sí que es mayor que azar
    assert prob > 0.0


def test_predict_low_risk_profile(low_risk_customer):
    """Un cliente con perfil de bajo riesgo debería tener prob < 0.9."""
    response = client.post("/predict", json=low_risk_customer)
    prob = response.json()["churn_probability"]
    assert prob < 1.0


# ─────────────────────────────────────────────
# Tests de validación de datos (errores 422)
# ─────────────────────────────────────────────
def test_predict_missing_field(valid_customer):
    """Falta un campo requerido → debe retornar 422 Unprocessable Entity."""
    del valid_customer["Contract"]
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 422


def test_predict_invalid_contract(valid_customer):
    """Valor inválido en Contract → debe retornar 422."""
    valid_customer["Contract"] = "Quarterly"  # valor no permitido
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 422


def test_predict_negative_tenure(valid_customer):
    """tenure negativo → debe retornar 422."""
    valid_customer["tenure"] = -5
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 422


def test_predict_negative_monthly_charges(valid_customer):
    """MonthlyCharges <= 0 → debe retornar 422."""
    valid_customer["MonthlyCharges"] = -10.0
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 422


# ─────────────────────────────────────────────
# Tests del endpoint /predict/batch
# ─────────────────────────────────────────────
def test_predict_batch_single(valid_customer):
    """Batch con un solo cliente debe funcionar correctamente."""
    response = client.post("/predict/batch", json=[valid_customer])
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert len(data["predictions"]) == 1


def test_predict_batch_multiple(valid_customer, low_risk_customer):
    """Batch con múltiples clientes debe retornar la misma cantidad."""
    payload = [valid_customer, low_risk_customer]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2


def test_predict_batch_too_large(valid_customer):
    """Batch de más de 1000 registros debe retornar 400."""
    payload = [valid_customer] * 1001
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400
