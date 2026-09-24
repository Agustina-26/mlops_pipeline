"""Monitoreo de data drift para el modelo de pago a tiempo."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from ft_engineering import DATE_COLUMN, TARGET, create_features

DEFAULT_THRESHOLDS = {"alpha": 0.05, "psi_warning": 0.10, "psi_alert": 0.25, "js_alert": 0.10}


def _proportions(values: pd.Series, categories: pd.Index) -> np.ndarray:
    """Proporciones con suavizado para evitar ceros en divergencias."""
    counts = values.value_counts(dropna=False).reindex(categories, fill_value=0).astype(float)
    counts = counts + 0.5
    return (counts / counts.sum()).to_numpy()


def _numeric_distribution(reference: pd.Series, current: pd.Series, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Distribuciones basadas en deciles de referencia."""
    quantiles = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(quantiles) == 1:
        # Una referencia constante también puede cambiar: conservar una caja central.
        value = quantiles[0]
        epsilon = max(abs(value) * 1e-9, 1e-9)
        edges = np.array([-np.inf, value-epsilon, value+epsilon, np.inf])
    else:
        # Midpoints preserva dos cajas para variables binarias; no colapsarlas en una.
        edges = np.r_[-np.inf, (quantiles[:-1]+quantiles[1:])/2, np.inf]
    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]
    ref_probs = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_probs = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))
    return ref_probs, cur_probs


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calcula el PSI usando cortes definidos solamente con la referencia."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if len(reference) < 2 or len(current) < 2:
        return np.nan
    ref_probs, cur_probs = _numeric_distribution(reference, current, bins)
    return float(np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs)))


def _drift_status(psi: float, js_distance: float, p_value: float, thresholds: dict) -> str:
    if (pd.notna(psi) and psi >= thresholds["psi_alert"]) or (pd.notna(js_distance) and js_distance >= thresholds["js_alert"]):
        return "Alerta"
    if (pd.notna(psi) and psi >= thresholds["psi_warning"]) or (pd.notna(p_value) and p_value < thresholds["alpha"]):
        return "Revisar"
    if pd.isna(psi) and pd.isna(js_distance) and pd.isna(p_value):
        return "Sin datos"
    return "Estable"


def _numeric_metrics(reference: pd.Series, current: pd.Series) -> dict:
    ref_valid = pd.to_numeric(reference, errors="coerce").dropna()
    cur_valid = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref_valid) < 2 or len(cur_valid) < 2:
        return {"ks_statistic": np.nan, "p_value": np.nan, "psi": np.nan, "js_distance": np.nan}
    ks = ks_2samp(ref_valid, cur_valid)
    ref_probs, cur_probs = _numeric_distribution(ref_valid, cur_valid)
    return {"ks_statistic": float(ks.statistic), "p_value": float(ks.pvalue), "psi": population_stability_index(ref_valid, cur_valid), "js_distance": float(jensenshannon(ref_probs, cur_probs))}


def _categorical_metrics(reference: pd.Series, current: pd.Series) -> dict:
    ref_values = reference.fillna("Sin dato").astype(str)
    cur_values = current.fillna("Sin dato").astype(str)
    categories = pd.Index(sorted(set(ref_values).union(set(cur_values))))
    ref_counts = ref_values.value_counts().reindex(categories, fill_value=0)
    cur_counts = cur_values.value_counts().reindex(categories, fill_value=0)
    try:
        chi2, p_value, _, _ = chi2_contingency(np.vstack([ref_counts.to_numpy(), cur_counts.to_numpy()]))
    except ValueError:
        chi2, p_value = np.nan, np.nan
    ref_probs, cur_probs = _proportions(ref_values, categories), _proportions(cur_values, categories)
    psi = float(np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs)))
    return {"chi2_statistic": chi2, "p_value": p_value, "psi": psi, "js_distance": float(jensenshannon(ref_probs, cur_probs))}


def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, numeric_columns: Iterable[str] | None = None, categorical_columns: Iterable[str] | None = None, thresholds: dict | None = None) -> pd.DataFrame:
    """Calcula KS, PSI y Jensen-Shannon para numéricas; chi-cuadrado para categorías."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    excluded = {TARGET, DATE_COLUMN, "prediccion", "probabilidad_no_pago", "periodo_monitoreo", "id_registro"}
    common = [col for col in reference_data.columns if col in current_data.columns and col not in excluded]
    if numeric_columns is None:
        numeric_columns = [col for col in common if pd.api.types.is_numeric_dtype(reference_data[col])]
    else:
        numeric_columns = [col for col in numeric_columns if col in common]
    if categorical_columns is None:
        categorical_columns = [col for col in common if col not in numeric_columns]
    else:
        categorical_columns = [col for col in categorical_columns if col in common]

    rows = []
    for col in numeric_columns:
        metric = _numeric_metrics(reference_data[col], current_data[col])
        rows.append({"variable": col, "tipo": "Numérica", "ks_statistic": metric["ks_statistic"], "chi2_statistic": np.nan, "p_value": metric["p_value"], "psi": metric["psi"], "js_distance": metric["js_distance"], "estado": _drift_status(metric["psi"], metric["js_distance"], metric["p_value"], thresholds)})
    for col in categorical_columns:
        metric = _categorical_metrics(reference_data[col], current_data[col])
        rows.append({"variable": col, "tipo": "Categórica", "ks_statistic": np.nan, "chi2_statistic": metric["chi2_statistic"], "p_value": metric["p_value"], "psi": metric["psi"], "js_distance": metric["js_distance"], "estado": _drift_status(metric["psi"], metric["js_distance"], metric["p_value"], thresholds)})
    return pd.DataFrame(rows).sort_values(["estado", "psi"], ascending=[True, False], na_position="last").reset_index(drop=True)


def periodic_sample(data: pd.DataFrame, sample_size: int = 1000, date_column: str = DATE_COLUMN, frequency: str = "M") -> pd.DataFrame:
    """Muestrea en cada período para que el monitoreo sea repetible y acotado."""
    sampled = data.copy()
    sampled[date_column] = pd.to_datetime(sampled[date_column], errors="coerce")
    sampled = sampled.dropna(subset=[date_column])
    sampled["periodo_monitoreo"] = sampled[date_column].dt.to_period(frequency).astype(str)
    samples = [
        group.sample(n=min(sample_size, len(group)), random_state=42)
        for _, group in sampled.groupby("periodo_monitoreo")
    ]
    # Se evita groupby.apply: en versiones recientes de pandas puede excluir la
    # columna de agrupación y luego impedir el reporte por período.
    return pd.concat(samples, ignore_index=True) if samples else sampled.iloc[0:0].copy()


def monitor_by_period(reference_data: pd.DataFrame, incoming_data: pd.DataFrame, sample_size: int = 1000, frequency: str = "M", thresholds: dict | None = None) -> pd.DataFrame:
    """Genera un reporte de drift para cada período de los datos entrantes."""
    reports = []
    for period, period_data in periodic_sample(incoming_data, sample_size, frequency=frequency).groupby("periodo_monitoreo"):
        report = detect_data_drift(reference_data, period_data, thresholds=thresholds)
        report.insert(0, "periodo_monitoreo", period)
        reports.append(report)
    return pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()


def build_prediction_table(model, raw_data: pd.DataFrame) -> pd.DataFrame:
    """Une los datos monitoreados con la predicción y probabilidad de no pago."""
    data_for_features = raw_data.copy()
    if TARGET not in data_for_features:
        data_for_features[TARGET] = np.nan
    features = create_features(data_for_features).drop(columns=[TARGET, DATE_COLUMN], errors="ignore")
    features = features.reindex(columns=list(model.feature_names_in_))
    probabilities = model.predict_proba(features)
    result = raw_data.copy().reset_index(drop=True)
    result.insert(0, "id_registro", np.arange(1, len(result) + 1))
    result["prediccion"] = np.where(probabilities[:, list(model.classes_).index(0)] >= model.threshold_no_pago_, 0, 1)
    result["probabilidad_no_pago"] = probabilities[:, list(model.classes_).index(0)]
    return result



