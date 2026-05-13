"""
src/ocr_helper.py
-----------------
Week 11 — OCR helper: extract text from a news article image.

Primary engine : pytesseract (wraps Tesseract, must be installed via brew)
Fallback engine: easyocr   (pure Python, downloads models on first use)

Install Tesseract on macOS:
    brew install tesseract tesseract-lang

Usage:
    from src.ocr_helper import extract_text_from_image, ocr_available
    text, engine, error = extract_text_from_image(image_bytes_or_path)
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# Minimum extracted characters to treat the result as usable
MIN_TEXT_CHARS = 30


def _try_pytesseract(image_input: Union[bytes, str, Path]) -> str:
    """Extract text with pytesseract. Raises ImportError or pytesseract.TesseractNotFoundError."""
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps

    if isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    else:
        img = Image.open(image_input)

    # Pre-process: greyscale → mild sharpening → auto-contrast → resize if tiny
    img = img.convert("L")
    if min(img.size) < 600:
        scale = 600 / min(img.size)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)

    config = "--oem 3 --psm 3"
    text = pytesseract.image_to_string(img, config=config, lang="eng")
    return text.strip()


_easyocr_reader = None  # module-level cache


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _try_easyocr(image_input: Union[bytes, str, Path]) -> str:
    """Extract text with easyocr (downloads ~80 MB models on first run)."""
    import numpy as np
    from PIL import Image

    reader = _get_easyocr_reader()

    if isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
        arr = np.array(img)
    else:
        arr = str(image_input)

    results = reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join(results).strip()


def ocr_available() -> tuple[bool, str]:
    """
    Return (available, engine_name).
    Checks pytesseract first, then easyocr.
    """
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True, "pytesseract"
    except Exception:
        pass
    try:
        import easyocr  # noqa: F401
        return True, "easyocr"
    except ImportError:
        pass
    return False, "none"


def extract_text_from_image(
    image_input: Union[bytes, str, Path],
) -> tuple[str, str, str | None]:
    """
    Extract text from an image.

    Parameters
    ----------
    image_input : bytes | str | Path
        Raw image bytes (from Telegram download) or a file path.

    Returns
    -------
    text    : str            — extracted text (empty string on failure)
    engine  : str            — "pytesseract" | "easyocr" | "none"
    error   : str | None     — human-readable error message, or None on success
    """
    # Try pytesseract first
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        text = _try_pytesseract(image_input)
        engine = "pytesseract"
        if len(text) >= MIN_TEXT_CHARS:
            logger.info("OCR (pytesseract) extracted %d chars", len(text))
            return text, engine, None
        # too short — try easyocr before giving up
    except ImportError:
        logger.debug("pytesseract not installed, trying easyocr")
    except Exception as e:
        logger.warning("pytesseract failed: %s — trying easyocr", e)

    # Fallback: easyocr
    try:
        text = _try_easyocr(image_input)
        engine = "easyocr"
        if len(text) >= MIN_TEXT_CHARS:
            logger.info("OCR (easyocr) extracted %d chars", len(text))
            return text, engine, None
        return "", engine, "OCR ran but could not extract enough text from this image."
    except ImportError:
        pass
    except Exception as e:
        logger.warning("easyocr failed: %s", e)

    return (
        "",
        "none",
        "No OCR engine available. Install Tesseract: brew install tesseract",
    )
