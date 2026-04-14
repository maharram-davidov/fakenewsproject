"""
app/cli.py
==========
Week 2 — Terminal CLI Application
AI-Based Fake News Detection System

Modes
-----
  Interactive  :  python -m app.cli
  Single text  :  python -m app.cli --text "article text here"
  URL          :  python -m app.cli --url https://example.com/news-article
  Batch CSV    :  python -m app.cli --file articles.csv --col text
  Custom model :  python -m app.cli --model naive_bayes
"""

import argparse
import csv
import os
import sys
import time
import logging

# Suppress verbose pipeline logs when running in CLI mode
logging.disable(logging.CRITICAL)

# ── Colour support (graceful fallback if colorama is not installed) ────────────
try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False

    class _NoColor:
        def __getattr__(self, _):
            return ""

    Fore = Style = _NoColor()  # type: ignore[assignment]

from src.train_model import load_model, predict
from src.config import MODELS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║        🔍  Fake News Detector  CLI  v2.0             ║
║        AI-Based Capstone Detection System            ║
╚══════════════════════════════════════════════════════╝"""

HELP_TEXT = """
Commands  (interactive mode)
────────────────────────────────────────────────────────
  :url <link>    Fetch and analyse a news article URL
  :history       Show last 10 predictions
  :clear         Clear the screen
  :help          Show this help text
  :quit  /  :q   Exit the program
────────────────────────────────────────────────────────
To analyse text:
  Paste or type the article, then press ENTER on a blank
  line to submit.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _c(text: str, color: str) -> str:
    """Apply a colorama colour to text, or return plain text as fallback."""
    return f"{color}{text}{Style.RESET_ALL}" if _HAS_COLOR else text


def _bar(conf: float, width: int = 28) -> str:
    filled = round(conf * width)
    return "█" * filled + "░" * (width - filled)


def _print_banner() -> None:
    print(_c(BANNER, Fore.CYAN))


def _print_result(result: dict, elapsed_s: float, text_preview: str = "") -> None:
    label    = result["label_name"].upper()
    conf     = result.get("confidence") or 0.0
    conf_pct = f"{conf * 100:.1f}%"
    is_fake  = label == "FAKE"
    clr      = Fore.RED   if is_fake else Fore.GREEN
    icon     = "🚨" if is_fake else "✅"

    bar = _bar(conf)
    preview = (text_preview[:75] + "…") if len(text_preview) > 75 else text_preview
    preview = preview.replace("\n", " ")

    print()
    print(_c("─" * 54, Fore.CYAN))
    print(_c(f"  {icon}  PREDICTION  : {label}", clr))
    print(_c(f"  Confidence  : {conf_pct}", clr))
    print(_c(f"  [{bar}] {conf_pct}", clr))
    print(_c(f"  Time        : {elapsed_s * 1000:.1f} ms", Fore.WHITE))
    if text_preview:
        print(_c(f"  Preview     : {preview}", Fore.WHITE))
    print(_c("─" * 54, Fore.CYAN))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_saved_model_name() -> str | None:
    """Scan models/ and return the name of the first saved classifier."""
    if not os.path.isdir(MODELS_DIR):
        return None
    for fname in sorted(os.listdir(MODELS_DIR)):
        if fname.endswith(".pkl") and "vectorizer" not in fname:
            return fname[:-4]   # strip .pkl suffix
    return None


def _auto_load_model() -> tuple:
    """Auto-detect and load the saved model. Exits with a message if not found."""
    name = _find_saved_model_name()
    if name is None:
        print(_c("\n  [✗] No saved model found in models/", Fore.RED))
        print(_c("      Run:  python main.py   to train and save a model first.\n", Fore.YELLOW))
        sys.exit(1)
    model, vectorizer = load_model(name)
    print(_c(f"\n  [✓] Model loaded : {name}.pkl", Fore.GREEN))
    return model, vectorizer, name


# ─────────────────────────────────────────────────────────────────────────────
# URL scraping
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_url(url: str) -> str:
    """Download and extract visible text from a news URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print(_c("  [!] URL mode requires additional packages:", Fore.YELLOW))
        print(_c("      pip install requests beautifulsoup4", Fore.WHITE))
        return ""

    try:
        print(_c(f"  Fetching  {url} …", Fore.WHITE))
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsBot/2.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        word_count = len(text.split())
        print(_c(f"  Extracted {word_count:,} words from the page.", Fore.WHITE))
        return text
    except Exception as exc:
        print(_c(f"  [✗] Could not fetch URL: {exc}", Fore.RED))
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Prediction wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _run_predict(text: str, model, vectorizer) -> tuple[dict, float] | None:
    """Return (result_dict, elapsed_seconds) or None if the text is empty."""
    if not text or not text.strip():
        return None
    t0 = time.perf_counter()
    result = predict(text, model, vectorizer)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _record(history: list, result: dict, text: str) -> None:
    history.append({
        "label":   result["label_name"].upper(),
        "conf":    (result.get("confidence") or 0.0) * 100,
        "preview": text.replace("\n", " ")[:50],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Mode: interactive
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive(model, vectorizer, model_name: str) -> None:
    print(_c(f"  Model  : {model_name}", Fore.WHITE))
    print(_c("  Mode   : interactive  (type :help for commands)\n", Fore.WHITE))

    history: list[dict] = []

    while True:
        try:
            print(_c("Paste article text (blank line to submit):", Fore.YELLOW))
            lines: list[str] = []

            while True:
                try:
                    line = input()
                except EOFError:
                    raise KeyboardInterrupt

                cmd = line.strip()

                # ── Commands ────────────────────────────────────────────────
                if cmd in (":quit", ":q"):
                    raise KeyboardInterrupt

                if cmd == ":help":
                    print(_c(HELP_TEXT, Fore.WHITE))
                    lines = []
                    break

                if cmd == ":clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    _print_banner()
                    lines = []
                    break

                if cmd == ":history":
                    if not history:
                        print(_c("  (no history yet)\n", Fore.WHITE))
                    else:
                        print(_c("\n  Last predictions:", Fore.CYAN))
                        for i, h in enumerate(history[-10:], 1):
                            clr = Fore.RED if h["label"] == "FAKE" else Fore.GREEN
                            print(_c(
                                f"  {i:2}. {h['label']:4}  {h['conf']:5.1f}%  {h['preview']}",
                                clr,
                            ))
                        print()
                    lines = []
                    break

                if cmd.startswith(":url "):
                    url  = cmd[5:].strip()
                    text = _fetch_url(url)
                    if text:
                        out = _run_predict(text, model, vectorizer)
                        if out:
                            result, elapsed = out
                            _print_result(result, elapsed, text)
                            _record(history, result, text)
                    lines = []
                    break

                # ── Empty line = end of paste ────────────────────────────────
                if cmd == "":
                    break   # analyse whatever is in `lines` (may be empty)

                lines.append(line)

            # ── Analyse collected text ───────────────────────────────────────
            if lines:
                text = " ".join(lines)
                out  = _run_predict(text, model, vectorizer)
                if out:
                    result, elapsed = out
                    _print_result(result, elapsed, text)
                    _record(history, result, text)

        except KeyboardInterrupt:
            print(_c("\n\n  Goodbye! 👋\n", Fore.CYAN))
            break


# ─────────────────────────────────────────────────────────────────────────────
# Mode: single text
# ─────────────────────────────────────────────────────────────────────────────

def run_single(text: str, model, vectorizer) -> int:
    out = _run_predict(text, model, vectorizer)
    if out is None:
        print(_c("  [!] Empty text provided.", Fore.YELLOW))
        return 1
    result, elapsed = out
    _print_result(result, elapsed, text)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Mode: URL
# ─────────────────────────────────────────────────────────────────────────────

def run_url(url: str, model, vectorizer) -> int:
    text = _fetch_url(url)
    if not text:
        return 1
    out = _run_predict(text, model, vectorizer)
    if out is None:
        print(_c("  [!] No usable text extracted from URL.", Fore.YELLOW))
        return 1
    result, elapsed = out
    _print_result(result, elapsed, text)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Mode: batch CSV
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(filepath: str, col: str, model, vectorizer) -> int:
    if not os.path.isfile(filepath):
        print(_c(f"  [✗] File not found: {filepath}", Fore.RED))
        return 1

    with open(filepath, newline="", encoding="utf-8") as f:
        reader    = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if col not in fieldnames:
            print(_c(f"  [✗] Column '{col}' not found in CSV.", Fore.RED))
            print(_c(f"      Available columns: {fieldnames}", Fore.WHITE))
            return 1
        rows = list(reader)

    total = len(rows)
    print(_c(f"\n  Batch mode — {total:,} rows  ·  file: {filepath}\n", Fore.CYAN))

    results_out             = []
    fake_count = real_count = skip_count = 0

    for i, row in enumerate(rows, 1):
        text = (row.get(col) or "").strip()
        if not text:
            skip_count += 1
            continue

        out = _run_predict(text, model, vectorizer)
        if out is None:
            skip_count += 1
            continue

        result, elapsed = out
        label   = result["label_name"].upper()
        conf    = (result.get("confidence") or 0.0) * 100
        conf_s  = f"{conf:.1f}%"
        preview = text.replace("\n", " ")[:60]
        is_fake = label == "FAKE"
        clr     = Fore.RED  if is_fake else Fore.GREEN
        icon    = "🚨"      if is_fake else "✅"

        print(_c(f"  [{i:5}/{total}]  {icon} {label:4}  {conf_s:6}  {preview}…", clr))

        results_out.append({**row, "prediction": label, "confidence_%": conf_s})
        if is_fake:
            fake_count += 1
        else:
            real_count += 1

    processed = fake_count + real_count
    print()
    print(_c("─" * 54, Fore.CYAN))
    print(_c(
        f"  Processed : {processed:,}   "
        f"Fake : {fake_count:,}   "
        f"Real : {real_count:,}   "
        f"Skipped : {skip_count:,}",
        Fore.WHITE,
    ))
    print(_c("─" * 54, Fore.CYAN))

    if results_out:
        out_path   = filepath.replace(".csv", "_predictions.csv")
        out_fields = list(results_out[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(results_out)
        print(_c(f"\n  [✓] Results saved → {out_path}\n", Fore.GREEN))

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="AI-Based Fake News Detection — Terminal CLI (Week 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m app.cli
  python -m app.cli --text "Scientists confirm water on Mars..."
  python -m app.cli --url  https://reuters.com/article/some-news
  python -m app.cli --file data/articles.csv --col text
  python -m app.cli --model naive_bayes --text "Breaking: ..."
""",
    )
    parser.add_argument("--text",  type=str,
                        help="Article text to classify directly")
    parser.add_argument("--url",   type=str,
                        help="URL of a news article to fetch and classify")
    parser.add_argument("--file",  type=str,
                        help="Path to a CSV file for batch prediction")
    parser.add_argument("--col",   type=str, default="text",
                        help="Column in the CSV that contains article text  [default: text]")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to load from models/  [default: auto-detect]")
    args = parser.parse_args()

    _print_banner()

    # ── Load model ──────────────────────────────────────────────────────────
    if args.model:
        try:
            model, vectorizer = load_model(args.model)
            model_name        = args.model
            print(_c(f"\n  [✓] Model loaded : {model_name}.pkl", Fore.GREEN))
        except Exception as exc:
            print(_c(f"\n  [✗] Could not load model '{args.model}': {exc}", Fore.RED))
            sys.exit(1)
    else:
        model, vectorizer, model_name = _auto_load_model()

    print(_c(f"  Using model  : {model_name}\n", Fore.WHITE))

    # ── Dispatch to the requested mode ─────────────────────────────────────
    if args.text:
        sys.exit(run_single(args.text, model, vectorizer))
    elif args.url:
        sys.exit(run_url(args.url, model, vectorizer))
    elif args.file:
        sys.exit(run_batch(args.file, args.col, model, vectorizer))
    else:
        run_interactive(model, vectorizer, model_name)


if __name__ == "__main__":
    main()
