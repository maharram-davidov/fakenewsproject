"""
preprocess.py
-------------
Text cleaning and normalisation pipeline for news articles.

Steps applied per article:
    1. Lowercase
    2. Remove URLs
    3. Remove non-alphabetic characters
    4. Collapse extra whitespace
    5. Tokenise & remove stopwords (NLTK English)
    6. Optional Porter stemming
    7. Drop very short tokens (< MIN_WORD_LENGTH chars)
"""

import re
import logging
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import MIN_WORD_LENGTH, USE_STEMMING

logger = logging.getLogger(__name__)

# Download NLTK data silently on first use
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_STEMMER    = PorterStemmer()

# Pre-compiled regex patterns for speed
_RE_URL       = re.compile(r"https?://\S+|www\.\S+")
_RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean a single news article string.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned, space-joined token string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1 – lowercase
    text = text.lower()

    # 2 – remove URLs
    text = _RE_URL.sub(" ", text)

    # 3 – remove non-alphabetic characters
    text = _RE_NON_ALPHA.sub(" ", text)

    # 4 – collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # 5 – tokenise, remove stopwords and short tokens
    tokens = [
        w for w in text.split()
        if w not in _STOP_WORDS and len(w) > MIN_WORD_LENGTH
    ]

    # 6 – optional stemming
    if USE_STEMMING:
        tokens = [_STEMMER.stem(w) for w in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text to the 'text' column of a DataFrame.

    Adds a new column 'clean_text' with the processed result.

    Parameters
    ----------
    df : pd.DataFrame  (must contain a 'text' column)

    Returns
    -------
    pd.DataFrame  (original df + 'clean_text' column, copy)
    """
    df = df.copy()
    df["text"] = df["text"].fillna("")
    logger.info("Cleaning text column ...")
    df["clean_text"] = df["text"].apply(clean_text)
    empty_count = (df["clean_text"] == "").sum()
    if empty_count:
        logger.warning(f"{empty_count} articles resulted in empty clean_text.")
    logger.info("Text preprocessing complete.")
    return df
