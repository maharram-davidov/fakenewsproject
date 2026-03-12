"""
features.py
-----------
Feature engineering for the Fake News Detection project.

Week 1: TF-IDF with unigrams + bigrams.
Future weeks can add BERT embeddings, linguistic features, etc.
"""

import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

from src.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)

logger = logging.getLogger(__name__)


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create a fresh TF-IDF vectoriser with project settings.

    Settings (from config.py):
        - max_features : 5 000
        - ngram_range  : (1, 2)  → unigrams + bigrams
        - sublinear_tf : True    → log-scaled term frequency
    """
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        strip_accents="unicode",
        analyzer="word",
    )


def vectorize(
    text_series: pd.Series,
    vectorizer: TfidfVectorizer = None,
    fit: bool = True,
) -> tuple:
    """
    Convert a text Series into a TF-IDF sparse matrix.

    Parameters
    ----------
    text_series : pd.Series[str]
    vectorizer  : TfidfVectorizer or None
        If None a new one is created. Pass an existing fitted vectorizer
        with fit=False for transform-only (e.g. inference).
    fit         : bool
        True  → fit_transform (training)
        False → transform only (inference / test set)

    Returns
    -------
    (X, vectorizer)
        X           : scipy sparse matrix  (n_samples × n_features)
        vectorizer  : fitted TfidfVectorizer
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()

    if fit:
        logger.info(
            f"Fitting TF-IDF vectoriser "
            f"(max_features={TFIDF_MAX_FEATURES}, "
            f"ngram_range={TFIDF_NGRAM_RANGE}) ..."
        )
        X = vectorizer.fit_transform(text_series)
    else:
        logger.info("Transforming text with existing TF-IDF vectoriser ...")
        X = vectorizer.transform(text_series)

    logger.info(f"Feature matrix shape: {X.shape}")
    return X, vectorizer
