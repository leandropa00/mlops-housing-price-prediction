"""Entrenamiento reproducible de tres experimentos sobre Boston Housing."""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Se excluye 'b': variable histórica construida a partir de una medida racial.
FEATURES = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "lstat"]
TARGET = "medv"  # miles de USD históricos; NO es valor de una casa individual.
BASE = Path(__file__).resolve().parents[2]


def make_pipeline(kind: str):
    if kind == "baseline":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="median"))])
    if kind == "ridge":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=20.0))])
    if kind == "forest":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=180, max_depth=11, min_samples_leaf=2, n_jobs=1, random_state=42))])
    raise ValueError(f"Modelo desconocido: {kind}")


def train(dataset: Path | None = None, output: Path | None = None, tracking: bool = True):
    dataset = Path(dataset) if dataset else BASE / "data/housing.csv"
    output = Path(output) if output else BASE / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {sorted(missing)}")
    df = df.dropna(subset=[TARGET]); X = df[FEATURES]; y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    rows = []; models = {}
    for kind in ("baseline", "ridge", "forest"):
        model = make_pipeline(kind)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {"mae": round(float(mean_absolute_error(y_test, pred)), 5), "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 5), "r2": round(float(r2_score(y_test, pred)), 5)}
        params = {k.replace("model__", ""):str(v) for k,v in model.get_params().items() if k.startswith("model__") and k.count("__") == 1}
        row = {"experiment":kind, "dataset_rows":len(df), "train_rows":len(X_train), "test_rows":len(X_test), "random_state":42, "test_size":0.2, **metrics, "parameters":json.dumps(params,sort_keys=True)}
        rows.append(row); models[kind]=model
        if tracking:
            try:
                import mlflow
                mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{BASE / 'mlflow.db'}"))
                mlflow.set_experiment("precios-vivienda-historicos")
                with mlflow.start_run(run_name=kind):
                    mlflow.log_params({"algorithm":kind, "random_state":42,"test_size":0.2,**params})
                    mlflow.log_metrics(metrics)
                    mlflow.set_tags({"dataset":"BostonHousing","source":"selva86/datasets","use":"academic-only"})
            except ImportError:
                # El CSV sigue registrando los tres experimentos sin dependencia opcional.
                pass
    best=min(rows,key=lambda x:x["mae"])
    payload={"model":models[best["experiment"]], "features":FEATURES, "model_name":best["experiment"], "mae":best["mae"], "dataset_source":"selva86/datasets/BostonHousing.csv", "target_units":"thousands_historical_usd"}
    joblib.dump(payload, output / "model.joblib")
    X_train.describe(percentiles=[.25,.5,.75]).loc[["mean","std","25%","50%","75%"]].to_json(output / "reference_stats.json",orient="index",indent=2)
    pd.DataFrame(rows).to_csv(output / "experiments.csv",index=False)
    (output / "experiments.json").write_text(json.dumps({"generated_at":datetime.now(timezone.utc).isoformat(),"dataset":"Boston Housing","objective":"regression on historical district medians","selected_model":best["experiment"],"selected_mae":best["mae"],"runs":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    result={"model":best["experiment"],"mae":best["mae"],"run_count":len(rows),"test_rows":len(y_test)}
    print(json.dumps(result,ensure_ascii=False));return result


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path);p.add_argument("--output",type=Path);p.add_argument("--no-mlflow",action="store_true")
    a=p.parse_args();train(dataset=a.data,output=a.output,tracking=not a.no_mlflow)
