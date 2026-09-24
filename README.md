# Predicción de pago a tiempo — Proyecto MLOps

## Caso de negocio

El proyecto estima si un crédito será pagado a tiempo (`Pago_atiempo`). El objetivo es apoyar la evaluación de riesgo crediticio e identificar operaciones con mayor probabilidad de no pago.

Se entrenó un **Random Forest**, seleccionado al comparar modelos supervisados. La evaluación prioriza métricas de la clase minoritaria (no pago), especialmente PR-AUC y F1.

## Flujo del proyecto

| Etapa | Archivo principal |
|---|---|
| Carga y exploración | `src/Cargar_Datos.ipynb` y `src/Comprensión_eda.ipynb` |
| Ingeniería de características | `src/ft_engineering.py` |
| Entrenamiento y evaluación | `src/model_training_evaluation.py` |
| Monitoreo de data drift | `src/model_monitoring.py` |
| Aplicación de monitoreo | `src/app.py` |
| API de predicción batch | `src/model_deploy.py` |

El modelo almacenado en `src/random_forest_v1.joblib` excluye `puntaje` por posible fuga de información detectada durante el EDA.

## Ejecución local

Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

Entrenar o regenerar el modelo:

```powershell
python src/model_training_evaluation.py
```

Iniciar la aplicación de monitoreo:

```powershell
python -m streamlit run src/app.py
```

Iniciar la API:

```powershell
Set-Location src
python -m uvicorn model_deploy:app --reload
```

La documentación interactiva queda disponible en `http://127.0.0.1:8000/docs`.

## API batch

- `GET /health`: estado del servicio y disponibilidad del modelo.
- `POST /predict`: recibe JSON con una lista de registros.
- `POST /predict/csv`: recibe un archivo CSV con múltiples registros.

Cada predicción devuelve la clase estimada y las probabilidades de pago a tiempo y no pago.

## Docker

La imagen utiliza `requirements-api.txt`, un conjunto reducido de dependencias para la API. `requirements.txt` se conserva para ejecutar el análisis, Streamlit y los notebooks localmente.

```powershell
docker build -t api-pago-tiempo .
docker run --name api-pago-tiempo-container -p 8000:8000 api-pago-tiempo
```

Luego se puede consultar `http://127.0.0.1:8000/docs`.
