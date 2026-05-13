"""
tests/test_gemini_helper.py
Unit tests for src.gemini_helper (prompt builders & truncation).
No API calls are made.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gemini_helper import (
    truncate_article,
    build_explain_prompt,
    build_summarise_prompt,
    build_chat_prompt,
)


SAMPLE = "Scientists say the new treatment could cure cancer. " * 20


def test_truncate_no_op_short():
    assert truncate_article("hello world") == "hello world"


def test_truncate_applies_limit():
    long_text = "x" * 50_000
    result = truncate_article(long_text, max_chars=100)
    assert len(result) <= 103  # 100 chars + "…"
    assert result.endswith("…")


def test_truncate_empty():
    assert truncate_article("") == ""
    assert truncate_article(None) == ""   # type: ignore[arg-type]


def test_explain_prompt_contains_label():
    prompt = build_explain_prompt(SAMPLE, "FAKE", 92.5)
    assert "FAKE" in prompt
    assert "92.5" in prompt


def test_explain_prompt_contains_article():
    article = "Unique marker text in article."
    prompt = build_explain_prompt(article, "REAL", 80.0)
    assert "Unique marker text" in prompt


def test_summarise_prompt_contains_article():
    article = "Another unique snippet."
    prompt = build_summarise_prompt(article)
    assert "Another unique snippet" in prompt


def test_chat_prompt_contains_question():
    prompt = build_chat_prompt(SAMPLE, "Who made this claim?")
    assert "Who made this claim?" in prompt


def test_chat_prompt_contains_article():
    article = "Distinct content here."
    prompt = build_chat_prompt(article, "What happened?")
    assert "Distinct content here" in prompt
