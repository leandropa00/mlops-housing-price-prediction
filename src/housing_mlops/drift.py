"""Señal de deriva: PSI de número de habitaciones; solo señal, no diagnóstico."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .training import BASE


def population_stability_index(expected, actual, bins=10):
    expected=np.asarray(expected,dtype=float);actual=np.asarray(actual,dtype=float)
    expected=expected[np.isfinite(expected)];actual=actual[np.isfinite(actual)]
    if len(expected)<20 or len(actual)<20: raise ValueError("Se requieren al menos 20 valores por muestra")
    edges=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0]=-np.inf;edges[-1]=np.inf
    e=np.histogram(expected,bins=edges)[0]/len(expected)
    a=np.histogram(actual,bins=edges)[0]/len(actual)
    e=np.maximum(e,1e-6);a=np.maximum(a,1e-6)
    return float(np.sum((a-e)*np.log(a/e)))


def report(reference_csv:Path, recent_csv:Path,threshold:float=0.2):
    reference=pd.read_csv(reference_csv); recent=pd.read_csv(recent_csv)
    psi=population_stability_index(reference["rm"],recent["rm"])
    return {"feature":"rm","psi":round(psi,4),"warning":psi>threshold,"threshold":threshold,"action":"Revisar calidad y evaluar reentrenamiento con etiquetas" if psi>threshold else "Sin alerta"}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--reference",type=Path,default=BASE/"data/housing.csv");p.add_argument("--recent",type=Path,required=True)
    a=p.parse_args();print(json.dumps(report(a.reference,a.recent),ensure_ascii=False))
