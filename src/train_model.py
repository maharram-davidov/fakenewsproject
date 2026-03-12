"""
train_model.py
--------------
Model training, persistence, and loading helpers.

Week 1 baseline models:
    ① Logistic Regression
    ② Multinomial Naive Bayes

Each model is trained on the TF-IDF feature matrix produced by features.py.
The best model (highest F1) is saved to disk with joblib.
"""

import os
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    VECTORIZER_FILENAME,
)
from src.evaluate import evaluate_model, plot_confusion_matrix

logger = logging.getLogger(__name__)


# ── Catalogue of Week-1 baseline models ──────────────────────────────────────
def get_baseline_models() -> dict:
    """
    Return a {name: unfitted_estimator} dict of all Week-1 models.
    Add more entries here in future weeks without touching main.py.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
    }


# ── Train + evaluate a single model ──────────────────────────────────────────
def train_and_evaluate(X, y, model_name: str = "Logistic Regression"):
    """
    Split data, train one model, evaluate it, and return results.

    Parameters
    ----------
    X          : TF-IDF feature matrix
    y          : label Series / array
    model_name : key in get_baseline_models()

    Returns
    -------
    (model, X_test, y_test, results_dict)
    """
    models = get_baseline_models()
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(models.keys())}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = models[model_name]
    logger.info(f"Training {model_name} ...")
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test, model_name)
    plot_confusion_matrix(y_test, results["predictions"], model_name)

    return model, X_test, y_test, results


# ── Train ALL baseline models ─────────────────────────────────────────────────
def train_all_models(X, y):
    """
    Train every model in get_baseline_models() on the same train/test split.

    Returns
    -------
    list[dict]   – list of results dicts (one per model)
    dict         – {model_name: fitted_model}
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    all_results = []
    fitted_models = {}

    for name, model in get_baseline_models().items():
        logger.info(f"Training {name} ...")
        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test, name)
        plot_confusion_matrix(y_test, results["predictions"], name)
        all_results.append(results)
        fitted_models[name] = model

    return all_results, fitted_models


# ── Persist model + vectorizer ────────────────────────────────────────────────
def save_model(model, vectorizer, model_name: str = "logistic_regression") -> None:
    """
    Serialise a fitted model and its TF-IDF vectoriser to the models/ folder.

    Parameters
    ----------
    model       : fitted sklearn estimator
    vectorizer  : fitted TfidfVectorizer
    model_name  : filename stem (spaces → underscores)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    safe_name    = model_name.lower().replace(" ", "_")
    model_path   = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
    vec_path     = os.path.join(MODELS_DIR, VECTORIZER_FILENAME)

    joblib.dump(model,      model_path)
    joblib.dump(vectorizer, vec_path)

    logger.info(f"Model saved     → {model_path}")
    logger.info(f"Vectorizer saved→ {vec_path}")
    print(f"\n[✓] Model saved      : {model_path}")
    print(f"[✓] Vectorizer saved : {vec_path}")


def load_model(model_name: str = "logistic_regression"):
    """
    Load a previously saved model and vectoriser.

    Returns
    -------
    (model, vectorizer)
    """
    safe_name  = model_name.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
    vec_path   = os.path.join(MODELS_DIR, VECTORIZER_FILENAME)

    model      = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    logger.info(f"Loaded model from {model_path}")
    return model, vectorizer


def predict(text: str, model, vectorizer) -> dict:
    """
    Predict fake/real for a single raw text string.

    Returns
    -------
    dict with keys: label (int), label_name (str), confidence (float)
    """
    from src.preprocess import clean_text
    from src.config import LABEL_NAMES

    cleaned = clean_text(text)
    X       = vectorizer.transform([cleaned])
    label   = int(model.predict(X)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X)[0][label])

    return {
        "label":      label,
        "label_name": LABEL_NAMES[label],
        "confidence": confidence,
    }
