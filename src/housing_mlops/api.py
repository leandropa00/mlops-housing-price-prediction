"""API de inferencia. El proceso se inicia con un modelo versionado joblib."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from .training import BASE

PREDICTIONS=Counter("housing_predictions_total","Predicciones realizadas")
LATENCY=Histogram("housing_prediction_seconds","Duración de predicción")
app=FastAPI(title="Housing MLOps - Académico",version="0.1.0",description="Mediana histórica por área, no una tasación actual")


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid")
    crim: float = Field(ge=0)
    zn: float = Field(ge=0,le=100)
    indus: float = Field(ge=0)
    chas: int = Field(ge=0,le=1)
    nox: float = Field(gt=0)
    rm: float = Field(gt=0)
    age: float = Field(ge=0,le=100)
    dis: float = Field(gt=0)
    rad: int = Field(ge=1)
    tax: float = Field(ge=0)
    ptratio: float = Field(gt=0)
    lstat: float = Field(ge=0,le=100)


@lru_cache(maxsize=1)
def load_model():
    path=Path(os.environ.get("MODEL_PATH",str(BASE / "artifacts/model.joblib")))
    if not path.exists():
        raise FileNotFoundError(f"Entrena primero el modelo: {path}")
    return joblib.load(path)


@app.get("/health")
def health():
    try:
        model=load_model()
        return {"status":"ok","model":model["model_name"],"scope":"historical_district_median"}
    except (FileNotFoundError,ValueError) as exc:
        raise HTTPException(status_code=503,detail=str(exc)) from exc


@app.post("/predict")
def predict(features: Features):
    try:
        model=load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503,detail=str(exc)) from exc
    with LATENCY.time():
        frame=pd.DataFrame([features.model_dump()])[model["features"]]
        value=float(model["model"].predict(frame)[0])
    PREDICTIONS.inc()
    return {"median_value_thousands_usd_historical":round(value,3),"model":model["model_name"],"warning":"Referencia histórica por distrito; no sirve para tasar viviendas actuales."}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(),media_type=CONTENT_TYPE_LATEST)
