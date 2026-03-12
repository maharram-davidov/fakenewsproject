"""
load_data.py
------------
Loads and merges the Fake / Real news CSV files into a single DataFrame.
Labels: Fake=1, Real=0
"""

import logging
import pandas as pd
from src.config import FAKE_CSV, TRUE_CSV, LABEL_FAKE, LABEL_REAL, RANDOM_STATE

logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """
    Read Fake.csv and True.csv, assign labels, concatenate, and shuffle.

    Returns
    -------
    pd.DataFrame
        Columns: title, text, subject, date, label
    """
    logger.info("Reading Fake.csv ...")
    fake = pd.read_csv(FAKE_CSV)
    fake["label"] = LABEL_FAKE

    logger.info("Reading True.csv ...")
    true = pd.read_csv(TRUE_CSV)
    true["label"] = LABEL_REAL

    df = pd.concat([fake, true], ignore_index=True)

    # Shuffle so Fake / Real are interleaved
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    logger.info(
        f"Dataset loaded → {df.shape[0]} rows | "
        f"Fake: {(df['label'] == LABEL_FAKE).sum()} | "
        f"Real: {(df['label'] == LABEL_REAL).sum()}"
    )
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return basic statistics about the loaded DataFrame."""
    return {
        "total_articles": len(df),
        "fake_count":     int((df["label"] == LABEL_FAKE).sum()),
        "real_count":     int((df["label"] == LABEL_REAL).sum()),
        "columns":        list(df.columns),
        "missing_text":   int(df["text"].isna().sum()),
    }
