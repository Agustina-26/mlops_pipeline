"""Aplicación Streamlit para monitorear data drift del modelo crediticio."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from ft_engineering import DATE_COLUMN
from model_deploy import REQUIRED_RAW_COLUMNS
from model_monitoring import DEFAULT_THRESHOLDS, build_prediction_table, detect_data_drift, monitor_by_period, periodic_sample


st.set_page_config(page_title="Monitoreo de modelo", page_icon="📊", layout="wide")


def find_base_data() -> Path | None:
    candidates = [
        Path.cwd() / "Base_de_datos.csv",
        Path(__file__).resolve().parent.parent / "Base_de_datos.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["", " ", "NA", "N/A", "null", "NULL", "None", "-", "?"])


@st.cache_resource
def load_deployed_model():
    import joblib
    return joblib.load(Path(__file__).with_name("model.joblib"))

st.title("📊 Monitoreo de data drift")
st.caption("Comparación entre la población de entrenamiento y datos recientes del modelo de pago a tiempo.")

base_path = find_base_data()
if base_path is None:
    st.error("No se encontró base_datos.csv. Colócala junto a app.py o actualiza la ruta en la aplicación.")
    st.stop()

base_data = load_csv(str(base_path))
base_data[DATE_COLUMN] = pd.to_datetime(base_data[DATE_COLUMN], errors="coerce")
base_data = base_data.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)
metadata = json.loads(Path(__file__).with_name("model_metadata.json").read_text(encoding="utf-8"))
reference_data = base_data[base_data[DATE_COLUMN] <= pd.Timestamp(metadata["partitions"]["train"]["end"])].copy()
default_current_data = base_data[base_data[DATE_COLUMN] >= pd.Timestamp(metadata["partitions"]["test"]["start"])].copy()

with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("CSV de datos recientes (opcional)", type="csv")
    sample_size = st.slider("Muestra máxima por período", 100, 2000, 1000, 100)
    psi_alert = st.slider("Umbral de alerta PSI", 0.10, 0.50, 0.25, 0.05)
    js_alert = st.slider("Umbral de alerta Jensen-Shannon", 0.05, 0.30, 0.10, 0.01)

if uploaded_file is not None:
    current_data = pd.read_csv(uploaded_file, na_values=["", " ", "NA", "N/A", "null", "NULL", "None", "-", "?"])
    st.info("Se compara el CSV cargado con la población histórica de referencia.")
else:
    current_data = default_current_data
    st.info("Se usa el 20% más reciente de la base como período de monitoreo. Podés cargar un CSV para simular datos nuevos.")

thresholds = {**DEFAULT_THRESHOLDS, "psi_alert": psi_alert, "js_alert": js_alert}
model = load_deployed_model()
missing = [c for c in REQUIRED_RAW_COLUMNS if c not in current_data.columns]
if missing or current_data.empty:
    st.error(f"CSV vacío o faltan columnas: {missing}")
    st.stop()
current_data[DATE_COLUMN] = pd.to_datetime(current_data[DATE_COLUMN], errors="coerce")
if current_data[DATE_COLUMN].isna().all():
    st.error("El CSV no contiene fechas válidas.")
    st.stop()
prediction_table = build_prediction_table(model, current_data)
sampled_current = periodic_sample(current_data, sample_size=sample_size)
drift_report = detect_data_drift(reference_data, sampled_current, thresholds=thresholds)

alert_count = int((drift_report["estado"] == "Alerta").sum())
review_count = int((drift_report["estado"] == "Revisar").sum())
no_payment_rate = (prediction_table["prediccion"] == 0).mean()
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Variables monitoreadas", len(drift_report))
metric_2.metric("Alertas de drift", alert_count)
metric_3.metric("Variables a revisar", review_count)
metric_4.metric("Predicciones de no pago", f"{no_payment_rate:.1%}")

st.subheader("Resumen de data drift")
st.caption("Umbrales configurados en la barra lateral. Jensen-Shannon se expresa como distancia. Un p-valor menor a 0,05 requiere revisión.")
st.dataframe(drift_report.style.format({"ks_statistic": "{:.3f}", "chi2_statistic": "{:.2f}", "p_value": "{:.4f}", "psi": "{:.3f}", "js_distance": "{:.3f}"}), width="stretch", hide_index=True)

st.subheader("Comparación visual")
selected_variable = st.selectbox("Variable", drift_report["variable"].tolist())
left, right = st.columns(2)
if pd.api.types.is_numeric_dtype(reference_data[selected_variable]):
    ref = pd.to_numeric(reference_data[selected_variable], errors="coerce").dropna()
    cur = pd.to_numeric(current_data[selected_variable], errors="coerce").dropna()
    values = pd.concat([ref, cur]).replace([np.inf, -np.inf], np.nan).dropna()
    if not values.empty:
        edges = np.histogram_bin_edges(values, bins=25)
        labels = [f"{a:.3g} a {b:.3g}" for a,b in zip(edges[:-1],edges[1:])]
        for panel, series, title in [(left,ref,"Referencia"),(right,cur,"Datos recientes")]:
            counts = np.histogram(series, bins=edges)[0]
            panel.caption(f"{title}: proporción por intervalo (mismos cortes)")
            panel.bar_chart(pd.Series(counts/max(counts.sum(),1), index=labels, name="Proporción"))
else:
    left.caption("Frecuencias de referencia")
    left.bar_chart(reference_data[selected_variable].fillna("Sin dato").astype(str).value_counts())
    right.caption("Frecuencias de datos recientes")
    right.bar_chart(current_data[selected_variable].fillna("Sin dato").astype(str).value_counts())

st.subheader("Datos recientes y pronósticos")
priority_columns = [column for column in ["id_registro", DATE_COLUMN, "prediccion", "probabilidad_no_pago"] if column in prediction_table.columns]
other_columns = [column for column in prediction_table.columns if column not in priority_columns]
st.dataframe(prediction_table[priority_columns + other_columns].head(500), width="stretch", hide_index=True)
st.download_button("Descargar tabla de monitoreo y predicciones", prediction_table.to_csv(index=False).encode("utf-8"), file_name="monitoreo_predicciones.csv", mime="text/csv")

with st.expander("Reporte por período"):
    periodic_report = monitor_by_period(reference_data, current_data, sample_size=sample_size, thresholds=thresholds)
    if periodic_report.empty:
        st.warning("No hay fechas válidas para construir el reporte periódico.")
    else:
        st.dataframe(periodic_report, width="stretch", hide_index=True)

