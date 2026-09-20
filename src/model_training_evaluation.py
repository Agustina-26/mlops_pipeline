"""Entrena y guarda el Random Forest seleccionado para su despliegue."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ft_engineering import build_model, get_feature_groups, split_train_test


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATHS = [PROJECT_ROOT / "Base_de_datos.csv"]
MODEL_PATH = Path(__file__).with_name("random_forest_v1.joblib")
METADATA_PATH = Path(__file__).with_name("random_forest_v1_metadata.json")


def find_data_path() -> Path:
    """Localiza la base en el proyecto o en la ubicación original."""
    path = next((candidate for candidate in DEFAULT_DATA_PATHS if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No se encontró base_datos.csv. Copiala al proyecto o ajusta DEFAULT_DATA_PATHS.")
    return path


def train_and_save(data_path: Path, model_path: Path = MODEL_PATH) -> None:
    data = pd.read_csv(data_path, na_values=["", " ", "NA", "N/A", "null", "NULL", "None", "-", "?"])
    X_train, _, y_train, _ = split_train_test(data)
    numeric_features, categorical_features = get_feature_groups(X_train)

    model = build_model(
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        numeric_features,
        categorical_features,
    )
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "model_name": "Random Forest V1.0.1",
        "target": "Pago_atiempo",
        "training_rows": int(len(X_train)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "source_data": str(data_path),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo guardado en: {model_path}")
    print(f"Registros de entrenamiento: {len(X_train):,}")


if __name__ == "__main__":
    train_and_save(find_data_path())
