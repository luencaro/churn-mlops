# Predicción de Churn en Telecomunicaciones
## Proyecto Integrador — Machine Learning con MLOps

Proyecto de clasificación supervisada para predecir el abandono de clientes (*churn*) en una empresa de telecomunicaciones. Implementa cuatro modelos de ensemble (Random Forest, XGBoost, CatBoost, LightGBM), explicabilidad con LIME, y una arquitectura MLOps completa con FastAPI, Docker y CI/CD.

---

## Estructura del proyecto

```
telco-churn-mlops/
├── data/
│   └── telco_churn.csv             # Dataset original de Kaggle
├── notebooks/
│   ├── Notebook1_EDA.ipynb         # Análisis exploratorio y preprocesamiento
│   ├── Notebook2_Models.ipynb      # Entrenamiento, GridSearchCV y evaluación
│   └── Notebook3_LIME.ipynb        # Interpretabilidad con LIME
├── app/
│   ├── api.py                      # Servicio FastAPI para inferencia
│   └── model.joblib                # Modelo entrenado (generado por Notebook 2)
├── tests/
│   └── test_api.py                 # Pruebas unitarias de la API
├── .github/
│   └── workflows/
│       └── ci.yml                  # Pipeline CI/CD con GitHub Actions
├── Dockerfile                      # Imagen Docker del servicio
├── requirements.txt                # Dependencias del proyecto
└── README.md
```

---

## Dataset

- **Fuente:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Tamaño:** ~7 000 registros, 21 columnas
- **Variable objetivo:** `Churn` (Yes/No) — desbalance ~73%/27%

---

## Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/telco-churn-mlops.git
cd telco-churn-mlops
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Ejecutar los notebooks en orden

```
Notebook1_EDA.ipynb        → genera data/preprocessor.joblib y data/train_test_splits.joblib
Notebook2_Models.ipynb     → genera app/model.joblib
Notebook3_LIME.ipynb       → genera explicaciones individuales
```

### 4. Iniciar la API localmente

```bash
uvicorn app.api:app --reload --port 8001
```

La documentación interactiva estará disponible en: [http://localhost:8001/docs](http://localhost:8001/docs)

---

## Uso de la API

### Endpoint: `POST /predict`

**Request:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 499.5
  }'
```

**Response:**
```json
{
  "churn_probability": 0.9600,
  "prediction": "Yes",
  "risk_level": "Alto",
  "model_name": "XGBClassifier"
}
```

### Endpoint: `POST /predict/batch`

Acepta una lista de hasta 1000 clientes y retorna predicciones para todos:

```bash
curl -X POST http://localhost:8001/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...cliente1...}, {...cliente2...}]'
```

---

## Docker

### Construir la imagen

```bash
docker build -t telco-churn-api .
```

### Ejecutar el contenedor

```bash
docker run -d \
  --name telco-churn-container \
  -p 8000:8000 \
  telco-churn-api
```

### Verificar que está corriendo

```bash
curl http://localhost:8000/health
```

### Administrar el contenedor

```bash
docker stop telco-churn-container    # Detener
docker start telco-churn-container   # Iniciar de nuevo
docker ps -a | grep telco-churn      # Ver estado
docker logs telco-churn-container    # Ver logs
```

---

## Pruebas unitarias

```bash
# Ejecutar todas las pruebas
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ --cov=app --cov-report=term-missing
```

---

**Secrets necesarios en GitHub** (para el paso Docker):
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## Modelos evaluados

| Modelo | Ventajas | Hiperparámetros clave |
|--------|----------|----------------------|
| Random Forest | Robusto, estable, fácil de interpretar | `max_depth`, `min_samples_split`, `criterion` |
| XGBoost | Alta precisión, manejo de desbalance | `subsample`, `colsample_bytree`, `scale_pos_weight` |
| CatBoost | Excelente con categóricas, pocas iteraciones | `l2_leaf_reg`, `bagging_temperature` |
| LightGBM | Muy rápido, eficiente con tabulares | `min_child_samples`, `max_bin`, `num_leaves` |

---

## Herramientas utilizadas

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `catboost` · `lightgbm` · `lime` · `matplotlib` · `seaborn` · `fastapi` · `uvicorn` · `joblib` · `docker` · `GitHub Actions`
