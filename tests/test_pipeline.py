from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from model_deploy import app, REQUIRED_RAW_COLUMNS, load_model
from model_monitoring import detect_data_drift, build_prediction_table
from model_training_evaluation import temporal_partitions
from ft_engineering import create_features, DATE_COLUMN

ROOT = Path(__file__).resolve().parent.parent
data = pd.read_csv(ROOT/'Base_de_datos.csv')
client = TestClient(app)

def test_temporal_separation():
    train, val, test = temporal_partitions(data)
    assert train[DATE_COLUMN].max() < val[DATE_COLUMN].min()
    assert val[DATE_COLUMN].max() < test[DATE_COLUMN].min()
    assert 'puntaje' not in train

def test_zero_income_no_infinity():
    row = data.head(1).copy()
    row['salario_cliente'] = 0
    features = create_features(row)
    assert features['ratio_cuota_ingreso'].isna().all()
    assert not np.isinf(features.select_dtypes('number')).any().any()

def test_api_and_dashboard_same_predictions():
    raw = data.tail(8)[REQUIRED_RAW_COLUMNS]
    records = __import__('json').loads(raw.to_json(orient='records'))
    response = client.post('/predict', json={'records': records})
    assert response.status_code == 200, response.text
    predictions = response.json()['predictions']
    dashboard = build_prediction_table(load_model(), raw)
    assert [r['prediccion'] for r in predictions] == dashboard.prediccion.tolist()
    for row in predictions:
        assert abs(row['probabilidad_no_pago']+row['probabilidad_pago_a_tiempo']-1) < 1e-5
    csv_response = client.post('/predict/csv', files={'file': ('batch.csv',raw.to_csv(index=False),'text/csv')})
    assert csv_response.status_code == 200
    assert csv_response.json()['predictions'] == predictions

def test_invalid_requests():
    assert client.post('/predict', json={'records': []}).status_code == 422
    assert client.post('/predict', json={'records': [{'x': 1}]}).status_code == 422
    assert client.post('/predict/csv', files={'file': ('empty.csv',b'', 'text/csv')}).status_code == 422
    assert client.get('/health').status_code == 200

def test_drift_detects_shift_and_missing():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({'x': rng.normal(size=1000)})
    assert detect_data_drift(ref, ref).iloc[0]['estado'] == 'Estable'
    assert detect_data_drift(ref, ref+10).iloc[0]['estado'] == 'Alerta'
    empty = pd.DataFrame({'x': [np.nan]*10})
    assert detect_data_drift(ref, empty).iloc[0]['estado'] == 'Sin datos'

def test_drift_constant_and_binary_variables():
    constant = pd.DataFrame({'x': [0.0]*200})
    assert detect_data_drift(constant, constant+1).iloc[0]['estado'] == 'Alerta'
    ref = pd.DataFrame({'x': [0.0]*100+[1.0]*100})
    current = pd.DataFrame({'x': [0.0]*190+[1.0]*10})
    assert detect_data_drift(ref, current).iloc[0]['psi'] > .25
