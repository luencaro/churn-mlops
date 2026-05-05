"""
api.py — Servicio de inferencia FastAPI para predicción de Churn
Telco Customer Churn — Proyecto MLOps
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
# Inicialización de la aplicación
# ─────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "API REST para predicción de abandono de clientes (churn) "
        "en una empresa de telecomunicaciones. Utiliza un pipeline de "
        "Machine Learning entrenado con el dataset Telco Customer Churn."
    ),
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Carga del modelo (una sola vez al iniciar)
# ─────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "app/model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Modelo cargado desde: {MODEL_PATH}")
    print(f"[INFO] Tipo de clasificador: {type(model.named_steps['clf']).__name__}")
except FileNotFoundError:
    raise RuntimeError(
        f"No se encontró el modelo en '{MODEL_PATH}'. "
        "Asegúrate de haber ejecutado el Notebook 2 para generar app/model.joblib."
    )


# ─────────────────────────────────────────────
# Esquema de datos de entrada (Pydantic)
# ─────────────────────────────────────────────
class CustomerData(BaseModel):
    """
    Esquema de entrada: características del cliente para la predicción.
    Todos los campos reflejan las columnas del dataset Telco Customer Churn.
    """
    # Demográficas
    gender: Literal["Male", "Female"] = Field(..., example="Male")
    SeniorCitizen: Literal[0, 1] = Field(..., example=0, description="1=adulto mayor, 0=no")
    Partner: Literal["Yes", "No"] = Field(..., example="No")
    Dependents: Literal["Yes", "No"] = Field(..., example="No")

    # Servicios de telecomunicaciones
    tenure: int = Field(..., ge=0, le=72, example=5, description="Meses como cliente")
    PhoneService: Literal["Yes", "No"] = Field(..., example="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., example="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., example="Fiber optic")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")

    # Facturación
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., example="Month-to-month"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., gt=0, example=99.9, description="Cargo mensual en USD")
    TotalCharges: float = Field(..., ge=0, example=499.5, description="Cargo total acumulado en USD")

    class Config:
        schema_extra = {
            "example": {
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
        }


# ─────────────────────────────────────────────
# Esquema de respuesta
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probabilidad de churn (0.0 – 1.0)")
    prediction: str = Field(..., description="'Yes' si prob >= 0.5, 'No' en caso contrario")
    risk_level: str = Field(..., description="Nivel de riesgo: Alto / Medio / Bajo")
    model_name: str = Field(..., description="Tipo de clasificador utilizado")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Endpoint raíz — verificación de que la API está activa."""
    return {
        "status": "online",
        "service": "Telco Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check — verifica que el modelo está cargado y listo."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classifier": type(model.named_steps["clf"]).__name__,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: CustomerData):
    """
    Recibe las características de un cliente y retorna:
    - `churn_probability`: probabilidad de abandono (0.0 – 1.0)
    - `prediction`: 'Yes' (churn) o 'No' (no churn)
    - `risk_level`: categorización del riesgo (Alto / Medio / Bajo)
    - `model_name`: nombre del clasificador usado
    """
    try:
        # Convertir a DataFrame para que el Pipeline aplique el preprocesador correctamente
        input_df = pd.DataFrame([data.dict()])

        # Inferencia
        churn_prob = float(model.predict_proba(input_df)[0][1])
        prediction = "Yes" if churn_prob >= 0.5 else "No"

        # Categorización del riesgo
        if churn_prob >= 0.70:
            risk_level = "Alto"
        elif churn_prob >= 0.40:
            risk_level = "Medio"
        else:
            risk_level = "Bajo"

        classifier_name = type(model.named_steps["clf"]).__name__

        return PredictionResponse(
            churn_probability=round(churn_prob, 4),
            prediction=prediction,
            risk_level=risk_level,
            model_name=classifier_name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerData]):
    """
    Recibe una lista de clientes y retorna predicciones para todos.
    Útil para scoring masivo desde sistemas externos.
    """
    if len(customers) > 1000:
        raise HTTPException(
            status_code=400,
            detail="El batch no puede superar 1000 registros por solicitud."
        )
    try:
        input_df = pd.DataFrame([c.dict() for c in customers])
        probs = model.predict_proba(input_df)[:, 1]
        results = []
        for prob in probs:
            prob = float(prob)
            results.append({
                "churn_probability": round(prob, 4),
                "prediction": "Yes" if prob >= 0.5 else "No",
                "risk_level": "Alto" if prob >= 0.70 else ("Medio" if prob >= 0.40 else "Bajo"),
            })
        return {"count": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
