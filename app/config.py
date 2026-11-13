"""Application paths and environment overrides."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

DEFAULT_DATA_FILE = DATA_DIR / "predictions.json"
DEFAULT_MODEL_FILE = MODELS_DIR / "iris_model.joblib"
