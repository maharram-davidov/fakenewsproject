"""
app/telegram_bot.py
===================
Week 7–10 — Telegram bot with rich HTML UI, inline keyboards, and Gemini AI features.

Commands: /start, /help, /check, /url, /history, /stats, /explain, /summarise, /ask

Env:
    TELEGRAM_BOT_TOKEN     — required (from @BotFather)
    GEMINI_API_KEY         — optional; enables /explain (Week 9–10)
    TELEGRAM_DB_PATH       — optional SQLite path (default: data/telegram_bot.sqlite3)
    TELEGRAM_WEBHOOK_URL   — optional public HTTPS base URL for webhook mode
    TELEGRAM_WEBHOOK_PATH  — optional URL path segment (default: tg-webhook)
    PORT                   — listen port in webhook mode

Run locally (polling):
    export TELEGRAM_BOT_TOKEN="..."
    python -m app.telegram_bot
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from src.config import BASE_DIR, MODELS_DIR
from src.train_model import load_model, predict
from src.gemini_helper import (
    build_explain_prompt,
    build_summarise_prompt,
    build_chat_prompt,
    generate_gemini_text,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)

GEMINI_TIMEOUT = 45  # seconds before giving up on Gemini API
logger = logging.getLogger(__name__)

DEFAULT_DB = os.path.join(BASE_DIR, "data", "telegram_bot.sqlite3")

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _conf_bar(pct: float, width: int = 12) -> str:
    """Unicode progress bar, e.g. ████████░░░░  82%"""
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _esc(text: str) -> str:
    """Escape characters that break HTML mode inside <code> blocks."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _preview(text: str, n: int = 100) -> str:
    one_line = " ".join(text.split())
    return (one_line[: n - 1] + "…") if len(one_line) > n else one_line


def _has_ai() -> bool:
    """True if any AI provider key is configured."""
    return bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )

# backwards-compat alias used in cmd_explain / cmd_summarise
_has_gemini = _has_ai


# ─────────────────────────────────────────────────────────────────────────────
# Inline keyboard builders
# ─────────────────────────────────────────────────────────────────────────────

def _result_keyboard(has_gemini: bool) -> InlineKeyboardMarkup:
    row1 = [
        InlineKeyboardButton("📊 Stats", callback_data="cb:stats"),
        InlineKeyboardButton("📋 History", callback_data="cb:history"),
    ]
    if not has_gemini:
        return InlineKeyboardMarkup([row1])
    row2 = [
        InlineKeyboardButton("✨ Explain", callback_data="cb:explain"),
        InlineKeyboardButton("📝 Summarise", callback_data="cb:summarise"),
        InlineKeyboardButton("💬 Ask", callback_data="cb:ask"),
    ]
    return InlineKeyboardMarkup([row2, row1])


def _back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("◀ Back", callback_data="cb:back"),
    ]])


# ─────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ─────────────────────────────────────────────────────────────────────────────

def db_connect() -> sqlite3.Connection:
    path = os.getenv("TELEGRAM_DB_PATH", DEFAULT_DB)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id    INTEGER NOT NULL,
            user_id    INTEGER,
            created_at TEXT    NOT NULL,
            source     TEXT    NOT NULL,
            preview    TEXT    NOT NULL,
            label      TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            raw_text   TEXT
        )
    """)
    # migration: add raw_text column to existing databases
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN raw_text TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()


def db_insert(
    conn: sqlite3.Connection,
    *,
    chat_id: int,
    user_id: int | None,
    source: str,
    preview: str,
    label: str,
    confidence: float,
    raw_text: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO predictions
            (chat_id, user_id, created_at, source, preview, label, confidence, raw_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            user_id,
            datetime.now(timezone.utc).isoformat(),
            source,
            preview[:500],
            label,
            confidence,
            raw_text[:10000],
        ),
    )
    conn.commit()


def db_last_text(conn: sqlite3.Connection, chat_id: int) -> str:
    row = conn.execute(
        "SELECT raw_text FROM predictions WHERE chat_id=? ORDER BY id DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    return (row["raw_text"] or "") if row else ""


def db_last_result(conn: sqlite3.Connection, chat_id: int) -> tuple[str, float] | None:
    row = conn.execute(
        "SELECT label, confidence FROM predictions WHERE chat_id=? ORDER BY id DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    return (str(row["label"]).upper(), float(row["confidence"])) if row else None


def db_history(conn: sqlite3.Connection, chat_id: int, limit: int = 10) -> list[sqlite3.Row]:
    return list(conn.execute(
        """
        SELECT created_at, source, preview, label, confidence
        FROM predictions
        WHERE chat_id = ?
        ORDER BY id DESC LIMIT ?
        """,
        (chat_id, limit),
    ).fetchall())


def db_stats(conn: sqlite3.Connection, chat_id: int) -> tuple[int, int, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN label IN ('Fake','FAKE') THEN 1 ELSE 0 END) AS fake_n,
            SUM(CASE WHEN label IN ('Real','REAL') THEN 1 ELSE 0 END) AS real_n
        FROM predictions WHERE chat_id = ?
        """,
        (chat_id,),
    ).fetchone()
    return int(row["total"] or 0), int(row["fake_n"] or 0), int(row["real_n"] or 0)


# ─────────────────────────────────────────────────────────────────────────────
# ML / URL helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _gemini(api_key: str, prompt: str) -> str:
    """Run generate_gemini_text in a thread with a hard timeout."""
    return await asyncio.wait_for(
        asyncio.to_thread(generate_gemini_text, api_key, prompt),
        timeout=GEMINI_TIMEOUT,
    )


def _fix_nltk_ssl() -> None:
    """Bypass macOS SSL cert issue and pre-download NLTK data."""
    import ssl
    import urllib.request
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        urllib.request.install_opener(
            urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        )
    except Exception:
        pass
    import nltk
    for res in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


def load_ml_or_exit():
    if not os.path.isdir(MODELS_DIR):
        logger.error("models/ missing — run python main.py first")
        sys.exit(1)
    for fname in sorted(os.listdir(MODELS_DIR)):
        if fname.endswith(".pkl") and "vectorizer" not in fname:
            name = fname[:-4]
            model, vectorizer = load_model(name)
            logger.info("Loaded model %s", fname)
            return model, vectorizer, name
    logger.error("No model .pkl found in models/")
    sys.exit(1)


def fetch_url_text(url: str) -> tuple[str, str | None]:
    """Return (text, error_message)."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return (text, None) if text.strip() else ("", "No readable text on this page.")
    except Exception as e:
        logger.exception("fetch_url_text failed")
        return "", str(e)


# ─────────────────────────────────────────────────────────────────────────────
# BotState
# ─────────────────────────────────────────────────────────────────────────────

class BotState:
    __slots__ = ("conn", "model", "vectorizer", "model_name")

    def __init__(self, conn, model, vectorizer, model_name: str):
        self.conn = conn
        self.model = model
        self.vectorizer = vectorizer
        self.model_name = model_name


# ─────────────────────────────────────────────────────────────────────────────
# Message builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_result_html(label: str, conf: float, ms: float, source: str, preview: str) -> str:
    is_fake = label == "FAKE"
    icon    = "🚨" if is_fake else "✅"
    verdict = "FAKE NEWS" if is_fake else "REAL NEWS"
    bar     = _conf_bar(conf)

    src_label = "URL" if source == "url" else "Text"

    return (
        f"{icon} <b>{verdict}</b>\n"
        f"{'─' * 28}\n"
        f"<b>Confidence</b>  <code>{bar}</code>  <b>{conf:.1f}%</b>\n"
        f"<b>Time</b>        {ms:.0f} ms\n"
        f"<b>Source</b>      {src_label}\n"
        f"{'─' * 28}\n"
        f"<i>{_esc(preview)}</i>"
    )


def _build_history_html(rows: list) -> str:
    lines = ["<b>📋 Last predictions</b>\n"]
    for i, r in enumerate(rows, 1):
        lbl  = str(r["label"]).upper()
        icon = "🚨" if lbl == "FAKE" else "✅"
        ts   = r["created_at"][:16].replace("T", " ")
        conf = float(r["confidence"])
        lines.append(
            f"{i}. {icon} <b>{lbl}</b>  {conf:.0f}%  <code>{r['source']}</code>  <i>{ts}</i>\n"
            f"   {_esc(str(r['preview'])[:80])}"
        )
    return "\n\n".join(lines)


def _build_stats_html(total: int, fake_n: int, real_n: int) -> str:
    fake_pct = fake_n / total * 100 if total else 0
    real_pct = real_n / total * 100 if total else 0
    fake_bar = _conf_bar(fake_pct)
    real_bar = _conf_bar(real_pct)
    return (
        f"<b>📊 Session statistics</b>\n"
        f"{'─' * 28}\n"
        f"Total  <b>{total}</b> predictions\n\n"
        f"🚨 Fake  <code>{fake_bar}</code>  <b>{fake_n}</b> ({fake_pct:.1f}%)\n"
        f"✅ Real  <code>{real_bar}</code>  <b>{real_n}</b> ({real_pct:.1f}%)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st: BotState = context.bot_data["state"]
    gemini_lines = (
        "\n\n<b>✨ Gemini AI</b>\n"
        "✨ /explain — why the model decided Fake/Real\n"
        "📝 /summarise — bullet-point summary of article\n"
        "💬 /ask <i>question</i> — ask anything about the article"
    ) if _has_gemini() else ""
    text = (
        "🔍 <b>Fake News Detector</b>\n"
        f"{'─' * 28}\n"
        f"Model  <code>{st.model_name}.pkl</code>\n"
        f"Accuracy  <b>99.22%</b>  ·  F1  <b>99.25%</b>\n"
        f"{'─' * 28}\n\n"
        "Send me a news article and I'll classify it.\n\n"
        "<b>Commands</b>\n"
        "📝 /check — classify pasted text\n"
        "🔗 /url — classify a news URL\n"
        "📋 /history — last predictions\n"
        "📊 /stats — session counts\n"
        "❓ /help — usage tips"
        f"{gemini_lines}"
    )
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "❓ <b>How to use</b>\n"
        f"{'─' * 28}\n\n"
        "<b>/check</b> <i>article text…</i>\n"
        "  Classify text you paste after the command.\n\n"
        "<b>/check</b> (reply)\n"
        "  Reply to any message that contains the article with /check.\n\n"
        "<b>/url</b> <i>https://…</i>\n"
        "  Bot fetches the page and classifies the article.\n\n"
        "<b>/history</b>\n"
        "  Last 10 predictions for this chat.\n\n"
        "<b>/stats</b>\n"
        "  Fake vs Real counts with a progress bar.\n\n"
        f"{'─' * 28}\n"
        "<b>✨ Gemini AI commands</b>  <i>(needs GEMINI_API_KEY)</i>\n\n"
        "<b>/explain</b> <i>(reply or text)</i>\n"
        "  Balanced media-literacy explanation of why the model\n"
        "  said Fake/Real — cites concrete cues from the text.\n\n"
        "<b>/summarise</b> <i>(reply or text)</i>\n"
        "  Bullet-point summary: headline takeaway + key facts.\n\n"
        "<b>/ask</b> <i>your question</i>\n"
        "  Ask anything about the last analysed article.\n"
        "  Gemini answers using only that article's content.\n\n"
        "<i>Tip: after /check, tap the inline buttons to Explain,\n"
        "Summarise, or Ask without re-pasting anything.</i>"
    )
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    st: BotState = context.bot_data["state"]
    text = None
    if msg.reply_to_message and msg.reply_to_message.text:
        text = msg.reply_to_message.text
    elif context.args:
        text = " ".join(context.args)

    if not text or not text.strip():
        await msg.reply_text(
            "📝 <b>Usage</b>\n"
            "• <code>/check Your article text here…</code>\n"
            "• Or reply to a message with <code>/check</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await _run_predict_and_reply(msg, st, text, source="text")


async def cmd_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    st: BotState = context.bot_data["state"]

    if not context.args:
        await msg.reply_text(
            "🔗 <b>Usage</b>\n"
            "<code>/url https://example.com/news/article</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    url = " ".join(context.args).strip()
    await msg.reply_chat_action(action="typing")

    body, err = fetch_url_text(url)
    if err:
        await msg.reply_text(f"⚠️ Could not fetch URL:\n<code>{_esc(err)}</code>", parse_mode=ParseMode.HTML)
        return
    if not body.strip():
        await msg.reply_text("⚠️ No readable text found on that page.")
        return

    wc = len(body.split())
    await msg.reply_text(f"🔗 Fetched <b>{wc:,}</b> words. Analysing…", parse_mode=ParseMode.HTML)
    await _run_predict_and_reply(msg, st, body, source="url")


async def _run_predict_and_reply(msg, st: BotState, text: str, source: str) -> None:
    await msg.reply_chat_action(action="typing")
    t0 = time.perf_counter()
    result = predict(text, st.model, st.vectorizer)
    ms = (time.perf_counter() - t0) * 1000

    label   = (result.get("label_name") or "?").upper()
    conf    = (result.get("confidence") or 0.0) * 100
    preview = _preview(text)
    uid     = msg.from_user.id if msg.from_user else None

    db_insert(
        st.conn,
        chat_id=msg.chat_id,
        user_id=uid,
        source=source,
        preview=preview,
        label=label,
        confidence=conf,
        raw_text=text,
    )

    reply_html = _build_result_html(label, conf, ms, source, preview)
    await msg.reply_text(
        reply_html,
        parse_mode=ParseMode.HTML,
        reply_markup=_result_keyboard(_has_gemini()),
    )


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st: BotState = context.bot_data["state"]
    rows = db_history(st.conn, update.effective_chat.id, limit=10)
    if not rows:
        await update.effective_message.reply_text(
            "📋 No predictions yet.\nUse /check or /url to analyse an article."
        )
        return
    await update.effective_message.reply_text(
        _build_history_html(rows),
        parse_mode=ParseMode.HTML,
        reply_markup=_back_keyboard(),
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st: BotState = context.bot_data["state"]
    total, fake_n, real_n = db_stats(st.conn, update.effective_chat.id)
    if total == 0:
        await update.effective_message.reply_text(
            "📊 No statistics yet.\nUse /check or /url first."
        )
        return
    await update.effective_message.reply_text(
        _build_stats_html(total, fake_n, real_n),
        parse_mode=ParseMode.HTML,
        reply_markup=_back_keyboard(),
    )


async def cmd_explain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        await update.effective_message.reply_text(
            "⚠️ Gemini not configured.\nSet <code>GEMINI_API_KEY</code> on the server.",
            parse_mode=ParseMode.HTML,
        )
        return

    msg        = update.effective_message
    bot_state: BotState = context.bot_data["state"]

    text = None
    if msg.reply_to_message and msg.reply_to_message.text:
        text = msg.reply_to_message.text
    elif context.args:
        text = " ".join(context.args)

    if not text or not text.strip():
        await msg.reply_text(
            "✨ <b>Usage</b>\n"
            "Reply to an article message with <code>/explain</code>, "
            "or send <code>/explain</code> followed by the article text.",
            parse_mode=ParseMode.HTML,
        )
        return

    await msg.reply_chat_action(action="typing")
    result = predict(text, bot_state.model, bot_state.vectorizer)
    label  = str(result.get("label_name") or "?").upper()
    conf   = (result.get("confidence") or 0.0) * 100
    prompt = build_explain_prompt(text, label, conf)

    await msg.reply_text("✨ <i>Asking Gemini…</i>", parse_mode=ParseMode.HTML)
    try:
        out = await _gemini(api_key, prompt)
    except asyncio.TimeoutError:
        await msg.reply_text("⚠️ Gemini timed out. Try again in a moment.")
        return
    except Exception as e:
        err_text = str(e)[:300]
        logger.error("cmd_explain Gemini error: %s", err_text)
        await msg.reply_text(f"⚠️ Gemini error:\n{err_text}")
        return

    if len(out) > 4090:
        out = out[:4087] + "…"

    await msg.reply_text(
        f"✨ <b>Gemini explanation</b>\n{'─' * 28}\n{out}",
        parse_mode=ParseMode.HTML,
    )


async def cmd_summarise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        await update.effective_message.reply_text(
            "⚠️ Gemini not configured.\nSet <code>GEMINI_API_KEY</code> on the server.",
            parse_mode=ParseMode.HTML,
        )
        return

    msg = update.effective_message
    st: BotState = context.bot_data["state"]

    text = None
    if msg.reply_to_message and msg.reply_to_message.text:
        text = msg.reply_to_message.text
    elif context.args:
        text = " ".join(context.args)
    else:
        text = db_last_text(st.conn, msg.chat_id)

    if not text or not text.strip():
        await msg.reply_text(
            "📝 <b>Usage</b>\n"
            "• Reply to an article message with <code>/summarise</code>\n"
            "• Or run <code>/check</code> first, then tap <b>📝 Summarise</b>",
            parse_mode=ParseMode.HTML,
        )
        return

    await msg.reply_chat_action(action="typing")
    prompt = build_summarise_prompt(text)
    await msg.reply_text("📝 <i>Summarising…</i>", parse_mode=ParseMode.HTML)
    try:
        out = await _gemini(api_key, prompt)
    except asyncio.TimeoutError:
        await msg.reply_text("⚠️ Gemini timed out. Try again in a moment.")
        return
    except Exception as e:
        err_text = str(e)[:300]
        logger.error("cmd_summarise Gemini error: %s", err_text)
        await msg.reply_text(f"⚠️ Gemini error:\n{err_text}")
        return

    if len(out) > 4090:
        out = out[:4087] + "…"
    await msg.reply_text(
        f"📝 <b>Summary</b>\n{'─' * 28}\n{out}",
        parse_mode=ParseMode.HTML,
    )


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        await update.effective_message.reply_text(
            "⚠️ Gemini not configured.\nSet <code>GEMINI_API_KEY</code> on the server.",
            parse_mode=ParseMode.HTML,
        )
        return

    msg = update.effective_message
    st: BotState = context.bot_data["state"]

    question = " ".join(context.args).strip() if context.args else ""
    if not question:
        await msg.reply_text(
            "💬 <b>Usage</b>\n"
            "<code>/ask Is this article biased?</code>\n\n"
            "<i>Gemini will answer using the last article you analysed.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    article = db_last_text(st.conn, msg.chat_id)
    if not article:
        await msg.reply_text(
            "💬 No article on record.\nRun /check or /url first, then ask your question."
        )
        return

    await msg.reply_chat_action(action="typing")
    prompt = build_chat_prompt(article, question)
    try:
        out = await _gemini(api_key, prompt)
    except asyncio.TimeoutError:
        await msg.reply_text("⚠️ Gemini timed out. Try again in a moment.")
        return
    except Exception as e:
        err_text = str(e)[:300]
        logger.error("cmd_ask Gemini error: %s", err_text)
        await msg.reply_text(f"⚠️ Gemini error:\n{err_text}")
        return

    if len(out) > 4090:
        out = out[:4087] + "…"
    await msg.reply_text(
        f"💬 <b>Answer</b>\n{'─' * 28}\n{out}",
        parse_mode=ParseMode.HTML,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inline button callbacks
# ─────────────────────────────────────────────────────────────────────────────

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    st: BotState = context.bot_data["state"]
    data = query.data or ""

    if data == "cb:stats":
        total, fake_n, real_n = db_stats(st.conn, query.message.chat_id)
        if total == 0:
            await query.message.reply_text("📊 No statistics yet.")
            return
        await query.message.reply_text(
            _build_stats_html(total, fake_n, real_n),
            parse_mode=ParseMode.HTML,
            reply_markup=_back_keyboard(),
        )

    elif data == "cb:history":
        rows = db_history(st.conn, query.message.chat_id, limit=10)
        if not rows:
            await query.message.reply_text("📋 No history yet.")
            return
        await query.message.reply_text(
            _build_history_html(rows),
            parse_mode=ParseMode.HTML,
            reply_markup=_back_keyboard(),
        )

    elif data == "cb:explain":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            await query.message.reply_text("⚠️ GEMINI_API_KEY not set on server.")
            return
        last_text = db_last_text(st.conn, query.message.chat_id)
        if not last_text:
            await query.message.reply_text("No article text found. Run /check or /url first.")
            return
        last = db_last_result(st.conn, query.message.chat_id)
        if not last:
            return
        label, conf_raw = last
        conf = conf_raw * 100 if conf_raw <= 1.0 else conf_raw
        prompt = build_explain_prompt(last_text, label, conf)
        await query.message.reply_text("✨ <i>Asking Gemini…</i>", parse_mode=ParseMode.HTML)
        try:
            out = await _gemini(api_key, prompt)
        except asyncio.TimeoutError:
            await query.message.reply_text("⚠️ Gemini timed out. Try again in a moment.")
            return
        except Exception as e:
            err_text = str(e)[:300]
            logger.error("cb:explain Gemini error: %s", err_text)
            await query.message.reply_text(f"⚠️ Gemini error:\n{err_text}")
            return
        if len(out) > 4090:
            out = out[:4087] + "…"
        await query.message.reply_text(
            f"✨ <b>Gemini explanation</b>\n{'─' * 28}\n{out}",
            parse_mode=ParseMode.HTML,
        )

    elif data == "cb:summarise":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            await query.message.reply_text("⚠️ GEMINI_API_KEY not set on server.")
            return
        last_text = db_last_text(st.conn, query.message.chat_id)
        if not last_text:
            await query.message.reply_text("No article text found. Run /check or /url first.")
            return
        await query.message.reply_text("📝 <i>Summarising…</i>", parse_mode=ParseMode.HTML)
        prompt = build_summarise_prompt(last_text)
        try:
            out = await _gemini(api_key, prompt)
        except asyncio.TimeoutError:
            await query.message.reply_text("⚠️ Gemini timed out. Try again in a moment.")
            return
        except Exception as e:
            err_text = str(e)[:300]
            logger.error("cb:summarise Gemini error: %s", err_text)
            await query.message.reply_text(f"⚠️ Gemini error:\n{err_text}")
            return
        if len(out) > 4090:
            out = out[:4087] + "…"
        await query.message.reply_text(
            f"📝 <b>Summary</b>\n{'─' * 28}\n{out}",
            parse_mode=ParseMode.HTML,
        )

    elif data == "cb:ask":
        await query.message.reply_text(
            "💬 <b>Ask about this article</b>\n"
            f"{'─' * 28}\n"
            "Send your question like this:\n"
            "<code>/ask Is this article biased?</code>",
            parse_mode=ParseMode.HTML,
        )

    elif data == "cb:back":
        await query.message.delete()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _fix_nltk_ssl()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("Set TELEGRAM_BOT_TOKEN (from @BotFather)")
        sys.exit(1)

    model, vectorizer, model_name = load_ml_or_exit()
    conn = db_connect()
    db_init(conn)
    state = BotState(conn, model, vectorizer, model_name)

    app = Application.builder().token(token).build()
    app.bot_data["state"] = state

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("help",      cmd_help))
    app.add_handler(CommandHandler("check",     cmd_check))
    app.add_handler(CommandHandler("url",       cmd_url))
    app.add_handler(CommandHandler("history",   cmd_history))
    app.add_handler(CommandHandler("stats",     cmd_stats))
    app.add_handler(CommandHandler("explain",   cmd_explain))
    app.add_handler(CommandHandler("summarise", cmd_summarise))
    app.add_handler(CommandHandler("ask",       cmd_ask))
    app.add_handler(CallbackQueryHandler(on_button))

    webhook_base = os.getenv("TELEGRAM_WEBHOOK_URL")
    if webhook_base:
        path     = os.getenv("TELEGRAM_WEBHOOK_PATH", "tg-webhook").strip("/")
        port     = int(os.getenv("PORT", "8080"))
        full_url = f"{webhook_base.rstrip('/')}/{path}"
        logger.info("Webhook mode  port=%s  url=%s", port, full_url)
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=path,
            webhook_url=full_url,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        logger.info("Polling mode (development)")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
