from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def ensure_dirs() -> None:
    for path in ["data/processed", "reports/tables", "reports/figures", "models"]:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: str | Path) -> None:
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=convert)


def load_data(customer_path: str | Path, panel_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer = pd.read_csv(customer_path)
    panel = pd.read_csv(panel_path)
    return customer, panel
