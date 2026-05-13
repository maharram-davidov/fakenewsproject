"""
main.py
=======
AI-Based Fake News Detection System
Capstone Design Project — Week 1 Pipeline

Pipeline Steps
--------------
1. Load       → merge Fake.csv + True.csv, shuffle
2. Preprocess → clean text (lowercase, remove URLs/noise, stop-words, stemming)
3. Features   → TF-IDF (unigrams + bigrams, 5 000 features)
4. Train      → Logistic Regression + Naive Bayes (same 80/20 split)
5. Evaluate   → accuracy, precision, recall, F1, ROC-AUC, confusion matrix
6. Save       → persist best model + vectorizer to models/

Usage
-----
    python main.py
"""

import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.load_data    import load_data, get_data_summary
from src.preprocess   import preprocess_dataframe
from src.features     import vectorize
from src.train_model  import train_all_models, save_model
from src.evaluate     import compare_models


def run_pipeline() -> None:
    logger.info("=" * 60)
    logger.info("  Fake News Detection — Baseline ML Pipeline")
    logger.info("=" * 60)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    logger.info("[1/5] Loading dataset ...")
    df      = load_data()
    summary = get_data_summary(df)
    print(f"\n  Total articles : {summary['total_articles']:,}")
    print(f"  Fake           : {summary['fake_count']:,}")
    print(f"  Real           : {summary['real_count']:,}")
    print(f"  Missing text   : {summary['missing_text']}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    logger.info("[2/5] Preprocessing text ...")
    df = preprocess_dataframe(df)

    # ── 3. Feature Extraction ─────────────────────────────────────────────────
    logger.info("[3/5] Extracting TF-IDF features ...")
    X, vectorizer = vectorize(df["clean_text"])
    y = df["label"]

    # ── 4. Train All Baseline Models ──────────────────────────────────────────
    logger.info("[4/5] Training baseline models ...")
    all_results, fitted_models = train_all_models(X, y)

    # ── 5. Compare & Save Best ────────────────────────────────────────────────
    logger.info("[5/5] Comparing models and saving best ...")
    comparison_df = compare_models(all_results)

    best_result = max(all_results, key=lambda r: r["f1"])
    best_name   = best_result["model_name"]
    best_model  = fitted_models[best_name]

    print(f"\n  ★  Best model: {best_name}  (F1 = {best_result['f1']:.4f})")

    save_model(best_model, vectorizer, best_name)

    logger.info("Pipeline complete! Model files saved to models/")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
