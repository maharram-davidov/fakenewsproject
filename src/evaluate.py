"""
evaluate.py
-----------
Evaluation utilities for classification models.

Provides:
    - evaluate_model()       → prints metrics and returns a results dict
    - plot_confusion_matrix() → saves + displays a heatmap
    - compare_models()       → side-by-side metric table for multiple models
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.config import MODELS_DIR, LABEL_NAMES

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a fitted classifier and print a formatted summary.

    Parameters
    ----------
    model      : fitted sklearn estimator
    X_test     : feature matrix
    y_test     : true labels
    model_name : display name

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc, predictions
    """
    pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall    = recall_score(y_test, pred, zero_division=0)
    f1        = f1_score(y_test, pred, zero_division=0)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc   = roc_auc_score(y_test, proba)
        except Exception:
            pass

    # ── Pretty-print ──────────────────────────────────────────────────────────
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {model_name} — Evaluation Results")
    print(sep)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC   : {auc:.4f}")
    print()
    print(
        classification_report(
            y_test, pred,
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]]
        )
    )

    return {
        "model_name": model_name,
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "auc":        auc,
        "predictions": pred,
    }


def plot_confusion_matrix(
    y_test,
    predictions,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """
    Plot and optionally save a confusion-matrix heatmap.

    Parameters
    ----------
    y_test      : true labels
    predictions : predicted labels
    model_name  : title string
    save        : if True, save PNG to models/ directory
    """
    cm = confusion_matrix(y_test, predictions)
    labels = [LABEL_NAMES[0], LABEL_NAMES[1]]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.ylabel("Actual Label",    fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        fname = model_name.replace(" ", "_") + "_confusion_matrix.png"
        fpath = os.path.join(MODELS_DIR, fname)
        plt.savefig(fpath, dpi=120)
        logger.info(f"Confusion matrix saved → {fpath}")

    plt.show()
    plt.close()


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Build a comparison table from a list of evaluate_model() dicts.

    Parameters
    ----------
    results_list : list[dict]   (output of evaluate_model)

    Returns
    -------
    pd.DataFrame  sorted by F1 descending
    """
    rows = []
    for r in results_list:
        rows.append({
            "Model":     r["model_name"],
            "Accuracy":  round(r["accuracy"],  4),
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1":        round(r["f1"],        4),
            "ROC-AUC":   round(r["auc"], 4) if r["auc"] is not None else "N/A",
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    df = df.reset_index(drop=True)
    print("\n" + "=" * 55)
    print("  Model Comparison")
    print("=" * 55)
    print(df.to_string(index=False))
    return df
