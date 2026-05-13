"""
tests/test_features.py
Unit tests for src.features (TF-IDF vectoriser).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.features import build_tfidf_vectorizer, vectorize


DOCS = pd.Series([
    "government election vote president senator",
    "scientist discovered cancer cure treatment",
    "stock market wall street economy crash",
])


def test_build_tfidf_vectorizer():
    vec = build_tfidf_vectorizer()
    assert vec is not None


def test_vectorize_shape():
    X, vec = vectorize(DOCS)
    assert X.shape[0] == len(DOCS)
    assert X.shape[1] > 0


def test_vectorize_transform_only():
    X_fit, vec = vectorize(DOCS, fit=True)
    new_doc = pd.Series(["election president vote"])
    X_new, _ = vectorize(new_doc, vectorizer=vec, fit=False)
    assert X_new.shape[0] == 1
    assert X_new.shape[1] == X_fit.shape[1]


def test_vectorize_sparse():
    from scipy.sparse import issparse
    X, _ = vectorize(DOCS)
    assert issparse(X)
