# Correcciones y cómo incorporarlas

Esta carpeta contiene una copia corregida del proyecto. **No se modificó el repositorio de GitHub ni tu archivo original de OneDrive.**

## Qué cambió

- Se recuperó tu EDA original y se guardó en `src/Comprensión_eda.ipynb`, el mismo nombre que estaba vacío en GitHub. Conserva tu análisis y agrega hallazgos cuantificados.
- Se corrigieron las rutas de carga y el gráfico de categorías que fallaba con valores faltantes. Se ejecutaron los notebooks y quedaron guardadas sus salidas.
- Se distinguieron reglas exploratorias de reglas de negocio confirmadas; no se elimina un código o plazo solo por ser poco frecuente.
- Se comparan baseline, regresión logística, Random Forest e HistGradientBoosting. Modelo y umbral se eligen con validación temporal. Se documentan métricas y limitaciones del conjunto de prueba.
- Se entrenó `src/model.joblib`. API y Streamlit usan ese mismo artefacto y umbral. El modelo anterior se conserva como histórico, sin utilizarlo en la aplicación.
- Se agregaron pruebas y un workflow de GitHub Actions. Se corrigió el monitoreo para distinguir falta de datos de estabilidad y detectar cambios de variables constantes/binarias.
- Se actualizaron README y dependencias directas con las versiones comprobadas.

## Cómo subirlo desde tu proyecto existente

1. Descomprimí el ZIP en una carpeta aparte. Conservá una copia de tu proyecto actual.
2. En VS Code, abrí tu repositorio original y su terminal. Ejecutá `git status`. Si hay cambios tuyos sin guardar en Git, revisalos y guardalos antes de continuar.
3. Con el árbol de trabajo limpio, creá una rama nueva:

```powershell
git switch main
git pull --ff-only origin main
git switch -c mejoras-evaluacion-eda
```

Si `git pull` muestra un error o conflicto, resolvelo antes de copiar archivos. Si el nombre de rama ya existe, usá otro nombre nuevo.

4. Copiá el contenido de esta carpeta dentro de tu repositorio original, combinando las carpetas y reemplazando los archivos correspondientes. No borres ni reemplaces la carpeta `.git`. El ZIP no contiene `.git` ni entornos virtuales.
5. Instalá las dependencias y comprobá el proyecto siguiendo el README. Revisá los cambios con `git diff --stat` y `git status`.
6. Agregá los archivos del proyecto corregido:

```powershell
git add -- src reports tests scripts .github README.md CAMBIOS_Y_COMO_SUBIR.md requirements.txt requirements-api.txt Dockerfile .dockerignore .gitignore pytest.ini
git diff --cached --stat
git commit -m "Completa EDA, evaluación temporal y pruebas del pipeline"
git push -u origin mejoras-evaluacion-eda
```

7. En GitHub, abrí un PR desde `mejoras-evaluacion-eda` hacia `main`. Esperá los controles de Actions y pedí una nueva revisión antes de integrar, si así lo requiere tu curso.

## Descripción sugerida del PR

> Recupera el notebook de EDA completo y corrige su ejecución y rutas. Agrega comparación de modelos con validación temporal, evaluación de no pago y documentación de limitaciones. Unifica modelo y umbral de la API y Streamlit e incorpora pruebas y un workflow de validación.
>
> Verificación local: notebooks ejecutados, pruebas de API/preprocesamiento/drift y prueba del dashboard. Docker pendiente de comprobar en Actions o en una computadora con Docker instalado. SonarCloud no está configurado.

## Límites de la verificación

El modelo tiene rendimiento modesto y muchas falsas alarmas, documentadas en el notebook de evaluación. El EDA original explora toda la base: el test es retrospectivo, con exposición exploratoria previa, y no sustituye datos nuevos. No se promete una calificación ni aptitud para decisiones reales de crédito.

GitHub Actions y Docker están preparados, pero no se afirma que hayan corrido en GitHub. SonarCloud requiere configuración adicional y queda fuera de esta entrega.
