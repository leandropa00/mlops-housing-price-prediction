"""Punto de entrada de la aplicación FastAPI para Vercel.

Los módulos del proyecto viven en el layout src/ y también funcionan
con el servidor local y con Docker.
"""
from pathlib import Path
import sys

src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from housing_mlops.api import app  # noqa: E402,F401
