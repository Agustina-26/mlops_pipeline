"""Aplicación Streamlit para monitorear data drift del modelo crediticio."""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from ft_engineering import DATE_COLUMN, TARGET, build_model, create_features, get_feature_groups
from model_monitoring import DEFAULT_THRESHOLDS, build_prediction_table, detect_data_drift, monitor_by_period, periodic_sample

import importlib
import model_monitoring


st.set_page_config(page_title="Monitoreo de modelo", page_icon="📊", layout="wide")


def find_base_data() -> Path | None:
    candidates = [
        Path.cwd() / "base_datos.csv",
        Path(r"C:/Users/Usuario/OneDrive/Desktop/ProyectoM5/mlops_pipeline/src/base_datos.csv"),
    ]
    return next((path for path in candidates if path.exists()), None)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["", " ", "NA", "N/A", "null", "NULL", "None", "-", "?"])


@st.cache_resource
def train_reference_model(reference_data: pd.DataFrame):
    transformed = create_features(reference_data).dropna(subset=[TARGET])
    X_reference = transformed.drop(columns=[TARGET, DATE_COLUMN], errors="ignore")
    y_reference = transformed[TARGET].astype(int)
    numeric_features, categorical_features = get_feature_groups(X_reference)
    estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1)
    return build_model(estimator, numeric_features, categorical_features).fit(X_reference, y_reference)


st.title("📊 Monitoreo de data drift")
st.caption("Comparación entre la población de entrenamiento y datos recientes del modelo de pago a tiempo.")

base_path = find_base_data()
if base_path is None:
    st.error("No se encontró base_datos.csv. Colócala junto a app.py o actualiza la ruta en la aplicación.")
    st.stop()

base_data = load_csv(str(base_path))
base_data[DATE_COLUMN] = pd.to_datetime(base_data[DATE_COLUMN], errors="coerce")
base_data = base_data.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)
cutoff = int(len(base_data) * 0.80)
reference_data = base_data.iloc[:cutoff].copy()
default_current_data = base_data.iloc[cutoff:].copy()

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
model = train_reference_model(reference_data)
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
st.caption("PSI ≥ 0,25 o Jensen-Shannon ≥ 0,10 generan alerta. Un p-valor menor a 0,05 requiere revisión.")
st.dataframe(drift_report.style.format({"ks_statistic": "{:.3f}", "chi2_statistic": "{:.2f}", "p_value": "{:.4f}", "psi": "{:.3f}", "js_divergence": "{:.3f}"}), use_container_width=True, hide_index=True)

st.subheader("Comparación visual")
selected_variable = st.selectbox("Variable", drift_report["variable"].tolist())
left, right = st.columns(2)
if pd.api.types.is_numeric_dtype(reference_data[selected_variable]):
    left.caption("Distribución de referencia")
    left.bar_chart(pd.to_numeric(reference_data[selected_variable], errors="coerce").dropna().value_counts(bins=25, sort=False))
    right.caption("Distribución de datos recientes")
    right.bar_chart(pd.to_numeric(current_data[selected_variable], errors="coerce").dropna().value_counts(bins=25, sort=False))
else:
    left.caption("Frecuencias de referencia")
    left.bar_chart(reference_data[selected_variable].fillna("Sin dato").astype(str).value_counts())
    right.caption("Frecuencias de datos recientes")
    right.bar_chart(current_data[selected_variable].fillna("Sin dato").astype(str).value_counts())

st.subheader("Datos recientes y pronósticos")
priority_columns = [column for column in ["id_registro", DATE_COLUMN, "prediccion", "probabilidad_no_pago"] if column in prediction_table.columns]
other_columns = [column for column in prediction_table.columns if column not in priority_columns]
st.dataframe(prediction_table[priority_columns + other_columns].head(500), use_container_width=True, hide_index=True)
st.download_button("Descargar tabla de monitoreo y predicciones", prediction_table.to_csv(index=False).encode("utf-8"), file_name="monitoreo_predicciones.csv", mime="text/csv")

with st.expander("Reporte por período"):
    periodic_report = monitor_by_period(reference_data, current_data, sample_size=sample_size, thresholds=thresholds)
    if periodic_report.empty:
        st.warning("No hay fechas válidas para construir el reporte periódico.")
    else:
        st.dataframe(periodic_report, use_container_width=True, hide_index=True)

