"""
tests/test_preprocess.py
Unit tests for src.preprocess.clean_text()
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_text


def test_returns_string():
    assert isinstance(clean_text("Hello world"), str)


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_none_like_input():
    # non-string should return ""
    assert clean_text(None) == ""   # type: ignore[arg-type]


def test_lowercasing():
    result = clean_text("BREAKING NEWS President")
    assert result == result.lower()


def test_urls_removed():
    result = clean_text("Visit https://reuters.com for more info")
    assert "https" not in result
    assert "reuters" not in result


def test_stopwords_removed():
    result = clean_text("this is a very long sentence with many stopwords inside")
    # "this", "is", "a", "with", "many" should be stripped
    assert "this" not in result.split()
    assert " is " not in f" {result} "


def test_non_alpha_removed():
    result = clean_text("Hello, World! It costs $100.00 today.")
    assert "$" not in result
    assert "100" not in result


def test_short_tokens_dropped():
    # Single-char tokens should be dropped
    result = clean_text("a b c hello world")
    tokens = result.split()
    assert all(len(t) > 2 for t in tokens), f"Short token found: {tokens}"


def test_non_empty_output_for_normal_text():
    result = clean_text("Scientists discovered water on the surface of Mars.")
    assert len(result) > 0
