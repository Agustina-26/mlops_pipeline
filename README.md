# Predicción de pago a tiempo — Proyecto MLOps

Proyecto académico para estimar `Pago_atiempo`. La clase de interés para evaluación es **no pago (0)**. Se conserva la etiqueta original en las respuestas de la API.

## Preparación

Python **3.12**. Desde la raíz del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En Linux/macOS activar con `source .venv/bin/activate`. Las dependencias directas están fijadas a las versiones usadas en la comprobación local. No se incluyen credenciales.

## Archivos y ejecución

| Etapa | Archivo |
|---|---|
| Carga portable Excel a CSV | `src/Cargar_Datos.ipynb` |
| EDA | `src/Comprensión_eda.ipynb` — original recuperado y ejecutado |
| Transformaciones | `src/ft_engineering.py` |
| Comparación y entrenamiento | `src/model_training_evaluation.py` |
| Interpretación de resultados | `src/Evaluacion_modelos.ipynb` |
| Métricas y curvas | `reports/` |
| Modelo seleccionado | `src/model.joblib` y `src/model_metadata.json` |
| API | `src/model_deploy.py` |
| Monitoreo | `src/model_monitoring.py` y `src/app.py` |

```powershell
python src/model_training_evaluation.py
python -m pytest -q
python scripts/validate_notebooks.py
python -m uvicorn model_deploy:app --app-dir src --reload
```

La API ofrece documentación interactiva en http://127.0.0.1:8000/docs. Solo cargar artefactos joblib de origen confiable.

## Evaluación reproducible

Comparación temporal 60/20/20, sin compartir timestamps entre particiones. Baseline, regresión logística, Random Forest e HistGradientBoosting. Selección por AP de no pago en validación; umbral por F1 en validación. No se reajusta el modelo elegido después de fijar el umbral. Test no participa en selección. El EDA original explora la base completa: por esa exposición previa, esta es una evaluación retrospectiva, no un test externo completamente inédito. Imputación, escalado y encoding se ajustan solo en entrenamiento.

Resultado ejecutado: **RandomForest**, ROC-AUC **0.650**, AP **0.063**, F1 **0.116**, precision **6.5%**, recall **52.2%** para no pago. Prevalencia de referencia en test: **3.2%**.

Se detectan 36 de 69 casos de no pago, con 517 falsas alarmas. Es un rendimiento limitado: no se afirma que esté listo para decisiones crediticias reales. `puntaje` se excluye preventivamente por posible fuga; debe confirmarse con el origen de los datos la disponibilidad de todas las variables al otorgar el crédito. El umbral optimiza F1, no costos monetarios. El modelo no está calibrado.

El script guarda hashes de datos/modelo, versiones, particiones, métricas y figuras. Ver `reports/comparacion_validacion.csv`, `reports/metricas.json` y el notebook de evaluación. Cambios posteriores deben validarse sin optimizar contra este test ya observado.

## API batch

- `GET /health`: verifica carga del modelo (503 si falla).
- `GET /model-info`: lista columnas necesarias y umbral.
- `POST /predict`: JSON `{"records": [...]}`, con una lista no vacía de registros.
- `POST /predict/csv`: archivo CSV mediante el campo `file`.

Cada resultado incluye etiqueta, predicción, probabilidad de pago y no pago. El esquema de columnas se consulta en `/model-info`. La API y el dashboard usan **el mismo artefacto y umbral**.

Para generar una solicitud de ejemplo a partir de los datos del proyecto:

```powershell
python scripts/create_example.py
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/request.json"
```

## Docker

```powershell
docker build -t api-pago-tiempo .
docker run --rm -p 8000:8000 api-pago-tiempo
```

Docker no está instalado en el entorno de esta corrección: el build local no fue comprobado. El workflow de GitHub está preparado para construir la imagen y comprobar `/health` cuando se suban los cambios.

## Streamlit y drift

```powershell
python -m streamlit run src/app.py
```

Referencia: período de entrenamiento registrado en metadata; período actual por defecto: test. Se puede cargar un CSV con las columnas originales. Se calculan KS/PSI para numéricas, chi-cuadrado para categorías y distancia Jensen-Shannon. Los umbrales son configurables. Los p-valores se usan como señales exploratorias sin corrección por comparaciones múltiples; una alerta no demuestra por sí sola pérdida de rendimiento. El reporte por período agrupa fechas, pero no constituye una tarea programada.

## Git y automatización

Trabajar en una rama nueva y abrir un PR hacia `main`. `.github/workflows/ci.yml` ejecuta pruebas, valida notebooks y comprueba Docker. La ejecución en GitHub está pendiente de subir esta versión. SonarCloud no está configurado; no se atribuye ese crédito extra.

El archivo antiguo `src/random_forest_v1.joblib` se conserva como versión histórica; la API y Streamlit usan exclusivamente `src/model.joblib`.
