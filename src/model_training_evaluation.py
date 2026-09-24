"""Compara modelos en validación temporal; evalúa el elegido una vez en test."""
import hashlib
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, precision_recall_curve,
    PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ft_engineering import TARGET, DATE_COLUMN, create_features, get_feature_groups, build_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'src/model.joblib'
REPORTS = ROOT / 'reports'

def temporal_partitions(data):
    """60/20/20 por fechas; nunca divide una misma fecha entre grupos."""
    df = create_features(data)
    valid = df[TARGET].isin([0, 1]) & df[DATE_COLUMN].notna()
    df = df.loc[valid].sort_values(DATE_COLUMN).reset_index(drop=True)
    dates = df[DATE_COLUMN].drop_duplicates().sort_values().to_numpy()
    if len(dates) < 5:
        raise ValueError('Se necesitan al menos cinco fechas distintas.')
    first, second = dates[int(len(dates)*.6)], dates[int(len(dates)*.8)]
    parts = [df[df[DATE_COLUMN] < first],
             df[(df[DATE_COLUMN] >= first) & (df[DATE_COLUMN] < second)],
             df[df[DATE_COLUMN] >= second]]
    for part in parts:
        if set(part[TARGET].unique()) != {0, 1}:
            raise ValueError('Cada partición necesita ambas clases para evaluar.')
    return parts

def xy(part):
    return part.drop(columns=[TARGET, DATE_COLUMN]), part[TARGET].astype(int)

def probabilities(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(0)]

def metrics(y, p, threshold):
    """Clase positiva de evaluación: NO pago (etiqueta original 0)."""
    truth = np.asarray(y) == 0
    pred = p >= threshold
    return {'roc_auc': float(roc_auc_score(truth, p)),
            'average_precision_no_pago': float(average_precision_score(truth, p)),
            'precision_no_pago': float(precision_score(truth, pred, zero_division=0)),
            'recall_no_pago': float(recall_score(truth, pred, zero_division=0)),
            'f1_no_pago': float(f1_score(truth, pred, zero_division=0)),
            'prevalencia_no_pago': float(truth.mean()),
            'matriz_confusion_no_pago': confusion_matrix(truth, pred, labels=[False, True]).tolist()}

def train_and_evaluate():
    REPORTS.mkdir(exist_ok=True)
    path = ROOT / 'Base_de_datos.csv'
    data = pd.read_csv(path)
    train, validation, test = temporal_partitions(data)
    X_train, y_train = xy(train)
    X_val, y_val = xy(validation)
    X_test, y_test = xy(test)
    numeric, categorical = get_feature_groups(X_train)
    candidates = {
        'Baseline': DummyClassifier(strategy='prior'),
        'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=3000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, min_samples_leaf=8,
                          class_weight='balanced', random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150,
                          class_weight='balanced', random_state=42),
    }
    fitted, rows = {}, []
    for name, estimator in candidates.items():
        print('Entrenando:', name, flush=True)
        model = build_model(estimator, numeric, categorical)
        model.fit(X_train, y_train)
        fitted[name] = model
        row = {'modelo': name, **metrics(y_val, probabilities(model, X_val), .5)}
        row.pop('matriz_confusion_no_pago')
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values('average_precision_no_pago', ascending=False)
    comparison.to_csv(REPORTS/'comparacion_validacion.csv', index=False)
    # Regla fijada antes de consultar test: mayor AP en validación, baseline elegible.
    name = comparison.iloc[0]['modelo']
    model = fitted[name]
    val_p = probabilities(model, X_val)
    precision, recall, thresholds = precision_recall_curve(y_val == 0, val_p)
    f1 = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-12)
    threshold = float(thresholds[int(np.argmax(f1))])
    # No se reajusta con validación: así el umbral conserva el modelo que lo generó.
    p = probabilities(model, X_test)
    results = metrics(y_test, p, threshold)
    metadata = {
        'model_name': name, 'version': '2.0.0', 'threshold_no_pago': threshold,
        'selection': 'Máxima average precision de no pago en validación temporal',
        'threshold_selection': 'Máximo F1 de no pago en validación; no optimizado en test',
        'test': results, 'validation_at_selected_threshold': metrics(y_val, val_p, threshold),
        'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'excluded_invalid_rows': int(len(data)-sum(map(len, [train, validation, test]))),
        'partitions': {key: {'rows': len(part), 'start': str(part[DATE_COLUMN].min()),
                             'end': str(part[DATE_COLUMN].max()),
                             'no_payment': int((part[TARGET]==0).sum())}
                       for key, part in zip(['train','validation','test'],[train,validation,test])},
        'limitations': ['Evaluación retrospectiva; requiere validación prospectiva.',
                       'El EDA original explora toda la base: test tiene exposición exploratoria previa.',
                       'Disponibilidad de todas las variables al otorgar crédito debe confirmarse.',
                       'F1 no representa costos económicos ni calibra probabilidades.']}
    model.threshold_no_pago_ = threshold
    model.model_name_ = name
    joblib.dump(model, MODEL_PATH)
    metadata['artifact_sha256'] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    (REPORTS/'metricas.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    (ROOT/'src/model_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(16,4))
    PrecisionRecallDisplay.from_predictions(y_test==0, p, ax=axes[0], name=name)
    axes[0].axhline((y_test==0).mean(), linestyle='--', label='Prevalencia')
    axes[0].legend()
    RocCurveDisplay.from_predictions(y_test==0, p, ax=axes[1], name=name)
    ConfusionMatrixDisplay.from_predictions(y_test==0, p>=threshold, ax=axes[2],
                                            display_labels=['Pago','No pago'], colorbar=False)
    fig.tight_layout(); fig.savefig(REPORTS/'evaluacion_test.png', dpi=140); plt.close(fig)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
