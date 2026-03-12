"""
config.py
---------
Central configuration for the Fake News Detection project.
All paths, hyperparameters, and constants live here.
"""

import os

# ── Directory Paths ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# ── Data Files ────────────────────────────────────────────────────────────────
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_FAKE = 1   # Fake news
LABEL_REAL = 0   # Real news
LABEL_NAMES = {LABEL_REAL: "Real", LABEL_FAKE: "Fake"}

# ── Preprocessing ─────────────────────────────────────────────────────────────
MIN_WORD_LENGTH  = 2      # Drop words shorter than this
USE_STEMMING     = True   # Apply Porter stemming

# ── TF-IDF Vectorizer ─────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams
TFIDF_SUBLINEAR_TF = True

# ── Model Training ────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Saved Model Names ─────────────────────────────────────────────────────────
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
