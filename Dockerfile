# ─────────────────────────────────────────────
# Dockerfile — Telco Churn Prediction API
# ─────────────────────────────────────────────

# Imagen base: Python 3.10 slim para minimizar el tamaño del contenedor
FROM python:3.10-slim

# Metadatos
LABEL maintainer="tu_email@ejemplo.com"
LABEL description="API de predicción de churn — Telco Customer Churn MLOps"
LABEL version="1.0.0"

# Evitar prompts interactivos durante la instalación de paquetes del sistema
ENV DEBIAN_FRONTEND=noninteractive

# Variables de entorno para Python
# PYTHONDONTWRITEBYTECODE: evita archivos .pyc innecesarios
# PYTHONUNBUFFERED: fuerza salida de logs en tiempo real (importante en Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=app/model.joblib

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# ── Instalar dependencias del sistema ──────────────────────────────────────
# libgomp1 es necesaria para LightGBM/XGBoost en contenedores slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Copiar e instalar dependencias Python ──────────────────────────────────
# Se copia requirements.txt primero para aprovechar la caché de capas Docker:
# si requirements.txt no cambia, esta capa no se re-ejecuta al reconstruir.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copiar el código fuente y el modelo ───────────────────────────────────
COPY app/ ./app/

# ── Exponer el puerto de la aplicación ────────────────────────────────────
EXPOSE 8000

# ── Health check del contenedor ───────────────────────────────────────────
# Docker verifica cada 30s que la API responde correctamente
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Comando de inicio ─────────────────────────────────────────────────────
# uvicorn sirve la aplicación FastAPI
# --host 0.0.0.0: acepta conexiones desde cualquier red (necesario en Docker)
# --port 8000:    puerto expuesto
# --workers 2:    dos workers para manejar concurrencia básica
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
