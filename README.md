# MLOps — Predicción académica de precios históricos de vivienda

Proyecto grupal: Mateo Aguirre, Santiago García y Leandro Panesso Atehortúa.

## Alcance
Se estima la **mediana histórica del precio por área** del conjunto Boston Housing, en miles de USD. **No** se predice el precio de una vivienda individual ni su valor actual. Los datos históricos tienen limitaciones sociales y de representatividad; la columna `b`, derivada de información racial, se excluye deliberadamente de las variables del modelo.

**Fuente pública**: https://github.com/selva86/datasets/blob/master/BostonHousing.csv (506 registros). El CSV normalizado `data/housing.csv` queda versionado para reproducibilidad.

## Ejecutar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,tracking]'
python -m housing_mlops.training
pytest -q
uvicorn housing_mlops.api:app --reload
```

La dependencia `mlflow` es opcional para el entrenamiento en entornos sin conexión. Si está instalada, `training.py` registra los tres experimentos y sus métricas en una base SQLite `mlflow.db` (o en la URI indicada por `MLFLOW_TRACKING_URI`); el registro portátil `artifacts/experiments.csv` se genera siempre. En GitHub Actions, la base SQLite también se conserva como artefacto descargable.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## API

- `GET /health`: estado y versión del modelo.
- `POST /predict`: recibe `crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, lstat`.
- `GET /metrics`: métricas Prometheus del servicio.

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"crim":0.00632,"zn":18,"indus":2.31,"chas":0,"nox":0.538,"rm":6.575,"age":65.2,"dis":4.09,"rad":1,"tax":296,"ptratio":15.3,"lstat":4.98}'
```

## Contenedor y CI

```bash
docker build -t mlops-housing .
docker run --rm -p 8000:8000 mlops-housing
```

`.github/workflows/ci.yml` entrena, comprueba tres ejecuciones registradas en MLflow y ejecuta `pytest` en cada push y pull request. El pipeline publica los experimentos y la base SQLite como artefactos. Consultar la pestaña Actions para verificar el resultado de cada ejecución.

## Experimentos y umbral
Tres algoritmos: baseline mediana, Ridge, Random Forest. Partición fija 80/20 (`random_state=42`); seleccionar por MAE menor. Umbral mínimo propuesto: MAE ≤ 4 (miles de USD históricos) y al menos 25 % de mejora sobre la línea base.

## Monitoreo y reentrenamiento
Datos: completitud, rangos e indicador PSI para `rm` (número medio de habitaciones). Rendimiento: MAE cuando existan etiquetas posteriores. Sistema: latencia p95, errores HTTP, disponibilidad. Señal de deriva: PSI > 0,2 en dos ventanas consecutivas; el disparador es revisar y reentrenar cuando además exista evidencia de degradación (MAE > 1,2 veces el MAE de validación), con nuevas etiquetas verificadas y aprobación humana.

Para analizar deriva sobre un CSV de eventos recientes con columna `rm`:

```bash
python -m housing_mlops.drift --recent data/housing.csv
```

## Estado de las evidencias
Entrenamiento, registro CSV, artefacto y pruebas: ejecutables localmente. API: probada en cliente de integración. Docker: empaquetado listo para construir. GitHub Actions: definido y publicado (estado verificable en Actions). Servicio público: no desplegado.

## Gestión ágil
Scrum con sprints **simulados** del 17 al 23 y del 24 al 30 de octubre de 2026, retrospectivas de aula. No constituyen evidencia de ejecución histórica. Entrega 31 de octubre de 2026.
