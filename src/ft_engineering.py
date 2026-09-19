"""Ingeniería de características para predecir Pago_atiempo.

El módulo concentra las transformaciones para que el entrenamiento y la
evaluación usen exactamente el mismo proceso de preparación de datos.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


TARGET = "Pago_atiempo"
DATE_COLUMN = "fecha_prestamo"

# Se excluye por defecto: el EDA indica que puede haber sido calculada después
# de conocer el comportamiento de pago. Solo debe incorporarse si negocio
# confirma que estaba disponible al otorgar el crédito.
POSSIBLE_LEAKAGE_COLUMNS = ["puntaje"]

MONEY_COLUMNS = [
    "capital_prestado",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "promedio_ingresos_datacredito",
]


def _available(data: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Devuelve únicamente las columnas presentes en el DataFrame."""
    return [column for column in columns if column in data.columns]


def create_features(data: pd.DataFrame, include_possible_leakage: bool = False) -> pd.DataFrame:
    """Limpia tipos y crea atributos listos para el preprocesamiento.

    No imputa ni escala variables: esas operaciones ocurren dentro del
    pipeline y se ajustan solamente con los datos de entrenamiento.
    """
    if TARGET not in data.columns:
        raise ValueError(f"No se encontró la variable objetivo '{TARGET}'.")

    df = data.copy()
    df = df.replace({"": np.nan, " ": np.nan, "NA": np.nan, "N/A": np.nan, "null": np.nan})

    numeric_columns = _available(
        df,
        [
            "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
            "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
            "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
            "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
            "creditos_sectorCooperativo", "creditos_sectorReal",
            "promedio_ingresos_datacredito", TARGET,
        ],
    )
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if DATE_COLUMN in df.columns:
        loan_date = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df["anio_prestamo"] = loan_date.dt.year
        df["mes_prestamo"] = loan_date.dt.month
        df["dia_semana_prestamo"] = loan_date.dt.dayofweek
        # Se conserva solo para ordenar el split temporal; no es predictor.
        df[DATE_COLUMN] = loan_date

    for column in _available(df, ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]):
        df[column] = df[column].astype("string").str.strip()

    if "tendencia_ingresos" in df.columns:
        valid_values = {"Creciente", "Decreciente", "Estable"}
        invalid = df["tendencia_ingresos"].notna() & ~df["tendencia_ingresos"].isin(valid_values)
        df.loc[invalid, "tendencia_ingresos"] = pd.NA

    for column in _available(df, ["puntaje_datacredito", "saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor", "promedio_ingresos_datacredito"]):
        df[f"{column}_faltante"] = df[column].isna().astype(int)

    if {"cuota_pactada", "salario_cliente"}.issubset(df.columns):
        df["ratio_cuota_ingreso"] = np.where(
            df["salario_cliente"] > 0,
            df["cuota_pactada"] / df["salario_cliente"],
            np.nan,
        )
    if {"total_otros_prestamos", "salario_cliente"}.issubset(df.columns):
        df["ratio_deuda_ingreso"] = np.where(
            df["salario_cliente"] > 0,
            df["total_otros_prestamos"] / df["salario_cliente"],
            np.nan,
        )

    sector_columns = _available(df, ["creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal"])
    if sector_columns:
        df["creditos_totales_sector"] = df[sector_columns].sum(axis=1, min_count=1)
    if "saldo_mora" in df.columns:
        df["tiene_mora"] = (df["saldo_mora"].fillna(0) > 0).astype(int)

    # log1p controla asimetrías; se mantienen las variables originales para
    # que el modelo pueda aprovechar relaciones no lineales simples.
    for column in _available(df, MONEY_COLUMNS):
        safe_values = df[column].where(df[column] >= 0)
        df[f"log_{column}"] = np.log1p(safe_values)

    # SimpleImputer reconoce np.nan, pero no pd.NA dentro de columnas object.
    # Esta conversión se hace antes del pipeline sin alterar los valores válidos.
    text_columns = df.select_dtypes(include=["string", "object", "category"]).columns
    for column in text_columns:
        df[column] = df[column].astype(object).where(df[column].notna(), np.nan)

    if not include_possible_leakage:
        df = df.drop(columns=_available(df, POSSIBLE_LEAKAGE_COLUMNS))
    return df


def split_train_test(data: pd.DataFrame, test_size: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide cronológicamente, dejando los créditos más recientes para evaluación."""
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    df = create_features(data)
    df = df.dropna(subset=[TARGET]).sort_values(DATE_COLUMN, na_position="first").reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_size))
    if cutoff == 0 or cutoff == len(df):
        raise ValueError("No hay suficientes registros para crear train y test.")

    feature_data = df.drop(columns=[TARGET, DATE_COLUMN], errors="ignore")
    target = df[TARGET].astype(int)
    return feature_data.iloc[:cutoff], feature_data.iloc[cutoff:], target.iloc[:cutoff], target.iloc[cutoff:]


def get_feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identifica columnas numéricas y categóricas después de crear features."""
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]
    return numeric_features, categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Crea un preprocesador compatible con los modelos comparados."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numericas", numeric_pipeline, numeric_features),
        ("categoricas", categorical_pipeline, categorical_features),
    ], remainder="drop")


def build_model(estimator, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Une el preprocesamiento y un estimador en un único pipeline."""
    return Pipeline([
        ("preprocesamiento", make_preprocessor(numeric_features, categorical_features)),
        ("modelo", estimator),
    ])
