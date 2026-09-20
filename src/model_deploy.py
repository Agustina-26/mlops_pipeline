"""API FastAPI para predicciones batch de pago a tiempo."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ft_engineering import DATE_COLUMN, TARGET, create_features


# En local se encuentra junto a este script; en Docker ambos se copian a /app.
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).with_name("random_forest_v1.joblib"))))
REQUIRED_RAW_COLUMNS = [
    "tipo_credito", "fecha_prestamo", "capital_prestado", "plazo_meses", "edad_cliente",
    "tipo_laboral", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
    "puntaje_datacredito", "cant_creditosvigentes", "huella_consulta", "saldo_mora",
    "saldo_total", "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal", "promedio_ingresos_datacredito",
    "tendencia_ingresos",
]


class BatchRequest(BaseModel):
    """Cuerpo JSON del endpoint /predict."""

    records: list[dict[str, Any]] = Field(..., min_length=1, description="Registros a predecir")


app = FastAPI(
    title="API de predicción crediticia",
    version="1.0.1",
    description="Servicio batch para estimar la probabilidad de pago a tiempo.",
)


@lru_cache
def load_model():
    """Carga el artefacto una sola vez durante la vida del servidor."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en '{MODEL_PATH}'. Ejecuta: python train_model.py")
    return joblib.load(MODEL_PATH)


def prepare_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Valida y aplica la misma ingeniería de características del entrenamiento."""
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in raw_data.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Faltan columnas requeridas.", "columns": missing})

    data = raw_data.copy()
    if TARGET not in data.columns:
        data[TARGET] = np.nan
    engineered = create_features(data).drop(columns=[TARGET, DATE_COLUMN], errors="ignore")
    model = load_model()
    return engineered.reindex(columns=list(model.feature_names_in_))


def predict_batch(raw_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Genera predicción, probabilidad de no pago y probabilidad de pago para cada registro."""
    if raw_data.empty:
        raise HTTPException(status_code=422, detail="El lote no contiene registros.")
    model = load_model()
    features = prepare_features(raw_data)
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    no_payment_probability = probabilities[:, classes.index(0)]
    on_time_probability = probabilities[:, classes.index(1)]
    predictions = model.predict(features)

    return [
        {
            "id_registro": index + 1,
            "prediccion": int(prediction),
            "etiqueta": "Pago a tiempo" if prediction == 1 else "No pago a tiempo",
            "probabilidad_no_pago": round(float(no_payment_probability[index]), 6),
            "probabilidad_pago_a_tiempo": round(float(on_time_probability[index]), 6),
        }
        for index, prediction in enumerate(predictions)
    ]


@app.get("/health")
def health() -> dict[str, str]:
    """Indica si el artefacto requerido por el servicio está disponible."""
    return {"status": "ok" if MODEL_PATH.exists() else "model_not_found", "model_path": str(MODEL_PATH)}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Expone información mínima para clientes de la API."""
    return {"model": "Random Forest V1.0.1", "required_columns": REQUIRED_RAW_COLUMNS, "batch_supported": True}


@app.post("/predict")
def predict_json(payload: BatchRequest) -> dict[str, Any]:
    """Recibe un objeto JSON con records y devuelve predicciones batch."""
    predictions = predict_batch(pd.DataFrame(payload.records))
    return {"count": len(predictions), "predictions": predictions}


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """Recibe un CSV y devuelve las predicciones de todos sus registros."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="El archivo debe tener extensión .csv.")
    try:
        content = await file.read()
        raw_data = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise HTTPException(status_code=422, detail=f"No se pudo leer el CSV: {error}") from error
    predictions = predict_batch(raw_data)
    return {"filename": file.filename, "count": len(predictions), "predictions": predictions}
