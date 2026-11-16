"""
src/ml_models/evaluate.py
-------------------------
Metric helpers used by the ML notebooks and any programmatic arena runs.

Public API:
    evaluate_regression(y_true, y_pred, model_name)            -> dict
    evaluate_classification(y_true, y_pred, model_name)        -> dict
    plot_confusion_matrix(y_true, y_pred, model_name, output_dir,
                          class_names=None)                    -> Path
    compare_models(list_of_dicts, task="regression")           -> DataFrame

Design:
  * All metrics are plain Python floats (JSON / CSV friendly).
  * Classification metrics default to `average="weighted"` — correct for
    multi-class AQI buckets with imbalanced categories (Good vs Severe).
  * `zero_division=0` so a missing predicted class yields 0.0 instead of
    a sklearn warning + NaN.
  * Confusion-matrix plot uses deterministic class ordering when provided,
    guaranteeing "Good → Severe" axis order across runs.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
sns.set_theme(style="white", context="notebook")


# --------------------------------------------------------------------------- #
# Regression metrics
# --------------------------------------------------------------------------- #
def evaluate_regression(y_true, y_pred, model_name: str = "model") -> Dict[str, object]:
    """
    Returns:
      {
        "model": <name>,
        "r2":   float,     # coefficient of determination
        "mae":  float,     # mean absolute error
        "rmse": float,     # root mean squared error
        "n":    int,       # sample count after NaN filtering
      }
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(f"Length mismatch: y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}")

    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    yt, yp = y_true_arr[mask], y_pred_arr[mask]
    if len(yt) == 0:
        return {"model": model_name, "r2": np.nan, "mae": np.nan,
                "rmse": np.nan, "n": 0}

    return {
        "model": model_name,
        "r2":   float(r2_score(yt, yp)),
        "mae":  float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "n":    int(len(yt)),
    }


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def evaluate_classification(y_true, y_pred, model_name: str = "model") -> Dict[str, object]:
    """
    Returns:
      {
        "model":     <name>,
        "accuracy":  float,
        "precision": float,   # weighted
        "recall":    float,   # weighted
        "f1":        float,   # weighted
        "n":         int,
      }
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(f"Length mismatch: y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}")

    mask = pd.notna(y_true_arr) & pd.notna(y_pred_arr)
    yt, yp = y_true_arr[mask], y_pred_arr[mask]
    if len(yt) == 0:
        return {"model": model_name, "accuracy": np.nan, "precision": np.nan,
                "recall": np.nan, "f1": np.nan, "n": 0}

    return {
        "model":     model_name,
        "accuracy":  float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, average="weighted", zero_division=0)),
        "recall":    float(recall_score(yt, yp, average="weighted", zero_division=0)),
        "f1":        float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "n":         int(len(yt)),
    }


# --------------------------------------------------------------------------- #
# Confusion matrix plot
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "model",
    output_dir: Optional[Path] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Save a Seaborn heatmap of the confusion matrix to <output_dir>/<slug>_cm.png.
    Returns the saved path.

    `class_names` fixes label order; when omitted we use sorted(union) which
    is deterministic but not semantic (e.g., "Good" before "Moderate").
    """
    output_dir = Path(output_dir) if output_dir else ASSETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    mask = pd.notna(y_true_arr) & pd.notna(y_pred_arr)
    yt, yp = y_true_arr[mask], y_pred_arr[mask]

    if len(yt) == 0:
        raise ValueError("plot_confusion_matrix received all-NaN labels.")

    if class_names is not None:
        labels = list(class_names)
    else:
        labels = sorted({*yt.tolist(), *yp.tolist()})

    cm = confusion_matrix(yt, yp, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Count"},
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", pad=14)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name.lower())
    out_path = output_dir / f"cm_{slug}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix -> %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Convenience: build a comparison table
# --------------------------------------------------------------------------- #
def compare_models(results: List[Dict[str, object]],
                   task: str = "regression") -> pd.DataFrame:
    """
    Turn a list of metric dicts into a sorted comparison DataFrame.

    Regression   : sorted by r2 (desc), then rmse (asc).
    Classification: sorted by f1 (desc), then accuracy (desc).
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if task == "regression":
        sort_cols = [c for c in ("r2", "rmse") if c in df.columns]
        ascending = [False if c == "r2" else True for c in sort_cols]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif task == "classification":
        sort_cols = [c for c in ("f1", "accuracy") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown task: {task!r} (expected 'regression' or 'classification')")

    # Round floats for readability
    float_cols = df.select_dtypes(include=[np.floating]).columns
    df[float_cols] = df[float_cols].round(4)
    return df


if __name__ == "__main__":
    # Smoke test
    y_t = [10, 20, 30, 40, 50]
    y_p = [12, 18, 33, 39, 55]
    print(evaluate_regression(y_t, y_p, "smoke_linreg"))

    yc_t = ["Good", "Poor", "Moderate", "Good", "Severe"]
    yc_p = ["Good", "Moderate", "Moderate", "Poor", "Severe"]
    print(evaluate_classification(yc_t, yc_p, "smoke_clf"))
    print(plot_confusion_matrix(yc_t, yc_p, "smoke_clf",
                                class_names=["Good", "Moderate", "Poor", "Severe"]))