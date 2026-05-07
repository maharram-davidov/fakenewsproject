"""
app/telegram_bot.py
===================
Week 7–10 — Telegram bot (python-telegram-bot) + optional Gemini /explain (Week 9–10).

Commands: /start, /help, /check, /url, /history, /stats, /explain

Env:
    TELEGRAM_BOT_TOKEN     — required (from @BotFather)
    GEMINI_API_KEY         — optional; enables /explain (Week 9–10)
    TELEGRAM_DB_PATH       — optional SQLite path (default: data/telegram_bot.sqlite3)
    TELEGRAM_WEBHOOK_URL   — optional public base URL for webhook mode (e.g. Railway)
    TELEGRAM_WEBHOOK_PATH  — optional URL path segment (default: tg-webhook)
    PORT                   — listen port in webhook mode (hosts often set this automatically)

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
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.config import BASE_DIR, MODELS_DIR
from src.train_model import load_model, predict
from src.gemini_helper import build_explain_prompt, generate_gemini_text

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEFAULT_DB = os.path.join(BASE_DIR, "data", "telegram_bot.sqlite3")


def _db_path() -> str:
    return os.getenv("TELEGRAM_DB_PATH", DEFAULT_DB)


def db_connect() -> sqlite3.Connection:
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            preview TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )
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
) -> None:
    conn.execute(
        """
        INSERT INTO predictions (chat_id, user_id, created_at, source, preview, label, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            user_id,
            datetime.now(timezone.utc).isoformat(),
            source,
            preview[:500],
            label,
            confidence,
        ),
    )
    conn.commit()


def db_history(conn: sqlite3.Connection, chat_id: int, limit: int = 10) -> list[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT created_at, source, preview, label, confidence
        FROM predictions
        WHERE chat_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (chat_id, limit),
    )
    return list(cur.fetchall())


def db_stats(conn: sqlite3.Connection, chat_id: int) -> tuple[int, int, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN label IN ('Fake','FAKE') THEN 1 ELSE 0 END) AS fake_n,
            SUM(CASE WHEN label IN ('Real','REAL') THEN 1 ELSE 0 END) AS real_n
        FROM predictions
        WHERE chat_id = ?
        """,
        (chat_id,),
    ).fetchone()
    total = int(row["total"] or 0)
    fake_n = int(row["fake_n"] or 0)
    real_n = int(row["real_n"] or 0)
    return total, fake_n, real_n


def fetch_url_text(url: str) -> tuple[str, str | None]:
    """Return (text, error_message)."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if not text.strip():
            return "", "No readable text on this page."
        return text, None
    except Exception as e:
        logger.exception("fetch_url_text failed")
        return "", str(e)


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


def _preview(text: str, n: int = 90) -> str:
    one_line = " ".join(text.split())
    return (one_line[: n - 1] + "…") if len(one_line) > n else one_line


class BotState:
    __slots__ = ("conn", "model", "vectorizer", "model_name")

    def __init__(
        self,
        conn: sqlite3.Connection,
        model,
        vectorizer,
        model_name: str,
    ):
        self.conn = conn
        self.model = model
        self.vectorizer = vectorizer
        self.model_name = model_name


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st: BotState = context.bot_data["state"]
    gem = ""
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        gem = "\n/explain — Gemini explanation (reply to article or paste text)"
    await update.effective_message.reply_text(
        f"🔍 Fake News Detector — Telegram\n"
        f"Active model: {st.model_name}.pkl\n\n"
        "Commands: /help · /check · /url · /history · /stats"
        f"{gem}\n\n"
        "Same ML pipeline as the Streamlit app."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [
        "Commands:",
        "/start — Welcome",
        "/help — This message",
        "/check <text> — classify text (or reply with /check)",
        "/url <https://…> — fetch URL and classify",
        "/history — last predictions",
        "/stats — counts this chat",
        "/explain — Gemini explains ML verdict (reply to article or /explain <text>)",
        "  (needs GEMINI_API_KEY on server)",
    ]
    await update.effective_message.reply_text("\n".join(lines))


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
            "Usage:\n"
            "• /check Your article text here…\n"
            "• Or reply to a message that contains the article with /check",
        )
        return

    await _run_predict_and_reply(msg, st, text, source="text")


async def cmd_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    st: BotState = context.bot_data["state"]
    if not context.args:
        await msg.reply_text("Usage: /url https://example.com/news/article")
        return
    url = " ".join(context.args).strip()
    await msg.reply_chat_action(action="typing")
    body, err = fetch_url_text(url)
    if err:
        await msg.reply_text(f"Could not fetch URL: {err}")
        return
    if not body.strip():
        await msg.reply_text("No text extracted from that URL.")
        return
    wc = len(body.split())
    await msg.reply_text(f"Fetched ~{wc:,} words. Analysing…")
    await _run_predict_and_reply(msg, st, body, source="url")


async def _run_predict_and_reply(msg, st: BotState, text: str, source: str) -> None:
    await msg.reply_chat_action(action="typing")
    t0 = time.perf_counter()
    result = predict(text, st.model, st.vectorizer)
    ms = (time.perf_counter() - t0) * 1000
    label = (result.get("label_name") or "?").upper()
    conf = (result.get("confidence") or 0.0) * 100
    preview = _preview(text)

    uid = msg.from_user.id if msg.from_user else None
    db_insert(
        st.conn,
        chat_id=msg.chat_id,
        user_id=uid,
        source=source,
        preview=preview,
        label=label,
        confidence=conf,
    )

    icon = "🚨" if label == "FAKE" else "✅"
    await msg.reply_text(
        f"{icon} {label}\n"
        f"Confidence: {conf:.1f}%\n"
        f"Time: {ms:.0f} ms\n"
        f"Preview: {preview}",
    )


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st: BotState = context.bot_data["state"]
    rows = db_history(st.conn, update.effective_chat.id, limit=10)
    if not rows:
        await update.effective_message.reply_text("No predictions stored yet for this chat.")
        return
    lines = []
    for r in rows:
        lbl = str(r["label"]).upper()
        icon = "🚨" if lbl == "FAKE" else "✅"
        ts = r["created_at"][:19].replace("T", " ")
        lines.append(
            f"{icon} {lbl} ({float(r['confidence']):.1f}%) · {r['source']} · {ts}\n"
            f"   {r['preview']}"
        )
    await update.effective_message.reply_text("\n\n".join(lines))


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st: BotState = context.bot_data["state"]
    total, fake_n, real_n = db_stats(st.conn, update.effective_chat.id)
    if total == 0:
        await update.effective_message.reply_text("No statistics yet — run /check or /url first.")
        return
    await update.effective_message.reply_text(
        f"This chat\n"
        f"Total predictions: {total}\n"
        f"Fake: {fake_n} ({fake_n / total * 100:.1f}%)\n"
        f"Real: {real_n} ({real_n / total * 100:.1f}%)",
    )


async def cmd_explain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        await update.effective_message.reply_text(
            "Gemini not configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY) on the server."
        )
        return
    msg = update.effective_message
    bot_state: BotState = context.bot_data["state"]
    text = None
    if msg.reply_to_message and msg.reply_to_message.text:
        text = msg.reply_to_message.text
    elif context.args:
        text = " ".join(context.args)
    if not text or not text.strip():
        await msg.reply_text(
            "Reply to the article message with /explain, or send /explain followed by the article text."
        )
        return
    await msg.reply_chat_action(action="typing")
    result = predict(text, bot_state.model, bot_state.vectorizer)
    label = str(result.get("label_name") or "?").upper()
    conf = (result.get("confidence") or 0.0) * 100
    prompt = build_explain_prompt(text, label, conf)
    try:
        out = await asyncio.to_thread(generate_gemini_text, api_key, prompt)
    except Exception as e:
        await msg.reply_text(f"Gemini error: {e}")
        return
    if len(out) > 4090:
        out = out[:4087] + "..."
    await msg.reply_text(out)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("Set TELEGRAM_BOT_TOKEN (from @BotFather)")
        sys.exit(1)

    model, vectorizer, model_name = load_ml_or_exit()
    conn = db_connect()
    db_init(conn)
    state = BotState(conn, model, vectorizer, model_name)

    application = Application.builder().token(token).build()
    application.bot_data["state"] = state

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("check", cmd_check))
    application.add_handler(CommandHandler("url", cmd_url))
    application.add_handler(CommandHandler("history", cmd_history))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("explain", cmd_explain))

    webhook_base = os.getenv("TELEGRAM_WEBHOOK_URL")
    if webhook_base:
        path = os.getenv("TELEGRAM_WEBHOOK_PATH", "tg-webhook").strip("/")
        port = int(os.getenv("PORT", "8080"))
        full_url = f"{webhook_base.rstrip('/')}/{path}"
        logger.info("Webhook mode on port %s url=%s", port, full_url)
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=path,
            webhook_url=full_url,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        logger.info("Polling mode (development)")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
