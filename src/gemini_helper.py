"""
gemini_helper.py
----------------
Week 9–10 — Google Gemini API helpers (explainability, summaries, Q&A).

Uses env GEMINI_API_KEY or GOOGLE_API_KEY. Optional GEMINI_MODEL (default:
gemini-2.0-flash); fall back to gemini-1.5-flash if the model is unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ("gemini-2.0-flash", "gemini-1.5-flash")
MAX_ARTICLE_CHARS = int(os.getenv("GEMINI_MAX_CHARS", "28000"))


def truncate_article(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "…"


def build_explain_prompt(article: str, ml_label: str, ml_confidence_pct: float) -> str:
    body = truncate_article(article)
    return (
        "You are an expert media-literacy assistant. A separate TF-IDF + Logistic "
        "Regression classifier (trained on English news) predicted this label.\n\n"
        f"Classifier output: **{ml_label.upper()}** with confidence **{ml_confidence_pct:.1f}%**.\n\n"
        "Article text:\n---\n"
        f"{body}\n---\n\n"
        "Give a balanced explanation (120–220 words):\n"
        "- Mention that the ML score reflects vocabulary/style patterns, not ground-truth.\n"
        "- Note 2–4 concrete cues from the text (headline tone, sensational language, "
        "named sources, dates, hedging, emotional appeals).\n"
        "- Suggest how a reader could verify claims (primary sources, reputable outlets).\n"
        "Write in clear English with short paragraphs."
    )


def build_summarise_prompt(article: str) -> str:
    body = truncate_article(article)
    return (
        "Summarise the following news-style article for a busy reader.\n"
        "- One-line headline-style takeaway.\n"
        "- 4–6 bullet points with the main facts.\n"
        "- End with one sentence on what is still uncertain or missing.\n\n"
        f"Article:\n---\n{body}\n---"
    )


def build_chat_prompt(article: str, user_message: str) -> str:
    body = truncate_article(article)
    return (
        "You answer questions ONLY using the article below plus general reasoning. "
        "If the article does not contain the answer, say so.\n"
        "Stay neutral; do not invent quotes or sources.\n\n"
        f"Article:\n---\n{body}\n---\n\n"
        f"User question: {user_message.strip()}"
    )


def generate_gemini_text(
    api_key: str,
    prompt: str,
    model_name: Optional[str] = None,
) -> str:
    """Call Gemini and return plain text, or raise with a clear message."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    candidates = []
    if model_name:
        candidates.append(model_name.strip())
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        candidates.append(env_model)
    for m in DEFAULT_MODELS:
        if m not in candidates:
            candidates.append(m)

    last_err: Exception | None = None
    for mid in candidates:
        try:
            model = genai.GenerativeModel(mid)
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 2048,
                },
            )
            text = None
            try:
                text = resp.text
            except ValueError:
                text = None
            if text and text.strip():
                return text.strip()
            block = getattr(resp, "prompt_feedback", None)
            if block and getattr(block, "block_reason", None):
                return (
                    f"Gemini blocked this request ({block.block_reason}). "
                    "Try shorter article text or a different question."
                )
            last_err = RuntimeError("Empty response from Gemini.")
        except Exception as e:
            logger.warning("Gemini model %s failed: %s", mid, e)
            last_err = e
            continue

    if last_err:
        raise RuntimeError(f"Gemini request failed: {last_err}") from last_err
    return "No response from Gemini."
