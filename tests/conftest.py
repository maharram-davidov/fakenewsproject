"""
tests/conftest.py
Download NLTK data before tests run (handles SSL issues on macOS).
"""
import ssl
import nltk


def pytest_configure(config):
    """Download required NLTK corpora once before any tests collect."""
    try:
        _ctx = ssl.create_default_context()
        _ctx.check_hostname = False
        _ctx.verify_mode = ssl.CERT_NONE
        import urllib.request
        _opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=_ctx)
        )
        urllib.request.install_opener(_opener)
    except Exception:
        pass

    for resource in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
