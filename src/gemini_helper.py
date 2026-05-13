"""
gemini_helper.py
----------------
Week 9–10 — AI text generation helpers (explainability, summaries, Q&A).

Provider priority:
  1. Google Gemini  (GEMINI_API_KEY)  — gemini-2.0-flash → gemini-2.0-flash-lite
  2. Groq           (GROQ_API_KEY)    — llama-3.3-70b-versatile → llama3-8b-8192
     Groq is used automatically when Gemini quota is exceeded (HTTP 429).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

MAX_ARTICLE_CHARS = int(os.getenv("GEMINI_MAX_CHARS", "28000"))
_REQUEST_TIMEOUT  = int(os.getenv("GEMINI_REQUEST_TIMEOUT", "30"))

_GEMINI_MODELS = ("gemini-2.0-flash", "gemini-2.0-flash-lite")
_GROQ_MODELS   = ("llama-3.3-70b-versatile", "llama3-8b-8192")

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def truncate_article(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "…"


def build_explain_prompt(article: str, ml_label: str, ml_confidence_pct: float) -> str:
    body = truncate_article(article)
    return (
        "You are a media-literacy assistant. Reply ONLY in the exact format below, no extra text.\n\n"
        f"ML verdict: {ml_label.upper()} ({ml_confidence_pct:.1f}% confidence)\n\n"
        "Article:\n---\n"
        f"{body}\n---\n\n"
        "Reply in this exact format (keep each section to 1-2 short sentences):\n\n"
        "🤖 Verdict: <one sentence: what the model detected and why the score is high/low>\n\n"
        "🔍 Key signals:\n"
        "• <signal 1>\n"
        "• <signal 2>\n"
        "• <signal 3>\n\n"
        "✅ How to verify: <one sentence with a practical tip>"
    )


def build_summarise_prompt(article: str) -> str:
    body = truncate_article(article)
    return (
        "Summarise the article below. Reply ONLY in this exact format, no extra text:\n\n"
        "📌 <one-line headline takeaway>\n\n"
        "• <key fact 1>\n"
        "• <key fact 2>\n"
        "• <key fact 3>\n"
        "• <key fact 4>\n\n"
        "❓ <one sentence on what is uncertain or missing>\n\n"
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


# ─────────────────────────────────────────────────────────────────────────────
# Provider implementations
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(api_key: str, prompt: str, model_name: Optional[str] = None) -> str:
    """Try Gemini models in order. Returns text or raises RuntimeError."""
    import requests as _req

    candidates: list[str] = []
    if model_name:
        candidates.append(model_name.strip())
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model and env_model not in candidates:
        candidates.append(env_model)
    for m in _GEMINI_MODELS:
        if m not in candidates:
            candidates.append(m)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
    }

    last_err: Exception | None = None
    for mid in candidates:
        url = _GEMINI_URL.format(model=mid, key=api_key)
        try:
            resp = _req.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise _QuotaError(f"Gemini quota exceeded (model={mid})")
            if resp.status_code != 200:
                msg = resp.json().get("error", {}).get("message", resp.text[:200])
                raise RuntimeError(f"Gemini HTTP {resp.status_code}: {msg}")
            data = resp.json()
            if "promptFeedback" in data:
                reason = data["promptFeedback"].get("blockReason")
                if reason:
                    return (
                        f"Gemini blocked this request ({reason}). "
                        "Try shorter article text or a different question."
                    )
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            text = " ".join(p.get("text", "") for p in parts).strip()
            if text:
                return text
            last_err = RuntimeError("Empty response from Gemini.")
        except _QuotaError:
            raise
        except _req.Timeout:
            last_err = RuntimeError(f"Gemini model {mid} timed out.")
            logger.warning("Gemini model %s timed out", mid)
        except Exception as e:
            last_err = e
            logger.warning("Gemini model %s failed: %s", mid, e)

    raise RuntimeError(f"Gemini failed: {last_err}") from last_err


def _call_groq(api_key: str, prompt: str) -> str:
    """Call Groq (OpenAI-compatible) API. Returns text or raises RuntimeError."""
    import requests as _req

    models = [os.getenv("GROQ_MODEL", "").strip()] if os.getenv("GROQ_MODEL") else []
    for m in _GROQ_MODELS:
        if m not in models:
            models.append(m)

    last_err: Exception | None = None
    for mid in models:
        try:
            resp = _req.post(
                _GROQ_URL,
                json={
                    "model": mid,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": 2048,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                raise _QuotaError(f"Groq quota exceeded (model={mid})")
            if resp.status_code != 200:
                msg = resp.json().get("error", {}).get("message", resp.text[:200])
                raise RuntimeError(f"Groq HTTP {resp.status_code}: {msg}")
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if text:
                return text
            last_err = RuntimeError("Empty response from Groq.")
        except _QuotaError:
            raise
        except _req.Timeout:
            last_err = RuntimeError(f"Groq model {mid} timed out.")
            logger.warning("Groq model %s timed out", mid)
        except Exception as e:
            last_err = e
            logger.warning("Groq model %s failed: %s", mid, e)

    raise RuntimeError(f"Groq failed: {last_err}") from last_err


class _QuotaError(RuntimeError):
    """Raised when a provider returns HTTP 429."""


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_gemini_text(
    api_key: str,
    prompt: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate text using the best available provider.

    Priority:
      1. Gemini  — if api_key is non-empty
         On HTTP 429 (quota), automatically falls back to Groq.
      2. Groq    — if GROQ_API_KEY is set (or Gemini quota exceeded)
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if api_key:
        try:
            return _call_gemini(api_key, prompt, model_name)
        except _QuotaError as qe:
            logger.warning("%s — falling back to Groq", qe)
            if groq_key:
                return _call_groq(groq_key, prompt)
            raise RuntimeError(
                "Gemini quota exceeded and GROQ_API_KEY is not set. "
                "Add GROQ_API_KEY to your .env (free at console.groq.com)."
            ) from qe

    if groq_key:
        logger.info("GEMINI_API_KEY not set — using Groq directly")
        return _call_groq(groq_key, prompt)

    raise RuntimeError("No AI API key available. Set GEMINI_API_KEY or GROQ_API_KEY.")
