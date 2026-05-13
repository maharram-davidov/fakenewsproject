"""
app/web_app.py
==============
Week 9–10 — Streamlit UI + Google Gemini (explainability, summaries, Q&A).

Run:
    streamlit run app/web_app.py
"""

import sys
import os
import time
import logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.train_model import load_model, predict
from src.config import MODELS_DIR
from src.gemini_helper import (
    build_chat_prompt,
    build_explain_prompt,
    build_summarise_prompt,
    generate_gemini_text,
)


def resolve_gemini_api_key():
    """Gemini API key from env or Streamlit secrets (never logged)."""
    k = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if k:
        return k.strip()
    try:
        sec = st.secrets
        if "GEMINI_API_KEY" in sec:
            return str(sec["GEMINI_API_KEY"]).strip() or None
        if "GOOGLE_API_KEY" in sec:
            return str(sec["GOOGLE_API_KEY"]).strip() or None
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Fake News Detector",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400..800;1,400..800&display=swap');

/* ── Base ──────────────────────────────────────────── */
html, body, input, textarea, button {
    font-family:"Plus Jakarta Sans",system-ui,-apple-system,sans-serif !important;
}
.stApp {
    background-color:#0B1524;
    background-image:
        radial-gradient(ellipse 900px 420px at 18% -8%, #00B4D828 0%, transparent 55%),
        radial-gradient(ellipse 700px 380px at 92% 8%, #06D6A018 0%, transparent 50%),
        radial-gradient(ellipse 600px 300px at 50% 110%, #EF476F12 0%, transparent 45%);
    color:#E8F0F8;
}
.block-container { padding-top:1.25rem; max-width:1180px; }
#MainMenu, footer, header { visibility:hidden; }

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(165deg,#070F18 0%,#0D1F35 48%,#091422 100%);
    border-right: 1px solid rgba(30,52,80,0.85);
    box-shadow: inset -1px 0 0 rgba(0,180,216,0.06);
}
[data-testid="stSidebar"] * { color:#CCD6E0 !important; }
[data-testid="stSidebar"] .stMarkdown h2 { letter-spacing:-0.02em; }

/* ── Hero ───────────────────────────────────────────── */
.hero-shell {
    position:relative;
    margin:-6px 0 10px 0;
    padding:1px;
    border-radius:18px;
    background:linear-gradient(135deg,#00B4D855 0%,#1E345066 38%,#06D6A044 100%);
    animation: fadeIn .45s ease;
}
.hero-inner {
    border-radius:17px;
    padding:22px 26px 20px;
    background:linear-gradient(180deg,rgba(15,36,58,0.94) 0%,rgba(9,20,34,0.92) 100%);
    border:1px solid rgba(30,52,80,0.95);
    box-shadow:0 18px 42px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
}
.hero-kicker {
    display:inline-block;
    font-size:0.72rem;
    font-weight:700;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:#48CAE4;
    background:rgba(0,180,216,0.12);
    border:1px solid rgba(0,180,216,0.35);
    border-radius:999px;
    padding:5px 12px;
    margin-bottom:12px;
}
.hero-title {
    margin:0;
    font-size:clamp(1.55rem,3vw,2.05rem);
    font-weight:800;
    letter-spacing:-0.03em;
    color:#E8F0F8;
    line-height:1.15;
}
.hero-icon { margin-right:8px; filter:drop-shadow(0 0 12px #00B4D866); }
.hero-sub {
    margin:10px 0 0 0;
    color:#8899AA;
    font-size:0.95rem;
    line-height:1.45;
    max-width:52ch;
}
.hero-chips { margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }
.hero-chip {
    font-size:0.76rem;
    font-weight:600;
    color:#AAB9C9;
    background:rgba(16,40,63,0.85);
    border:1px solid #1E3450;
    border-radius:999px;
    padding:6px 12px;
}
.hero-chip-accent {
    color:#0D1B2A;
    background:linear-gradient(135deg,#48CAE4,#00B4D8);
    border:none;
}

/* ── Section titles (Streamlit markdown headings) ─────── */
.stMarkdown h4 {
    color:#E8F0F8 !important;
    font-size:1.06rem !important;
    font-weight:700 !important;
    letter-spacing:-0.02em !important;
    margin:8px 0 14px 0 !important;
}
.stMarkdown h5 {
    color:#CCD6E0 !important;
    font-size:0.98rem !important;
    font-weight:700 !important;
    margin:12px 0 12px 0 !important;
}

/* ── Section titles (explicit) ───────────────────────── */
.section-title {
    font-size:1.05rem !important;
    font-weight:700 !important;
    letter-spacing:-0.02em;
    color:#E8F0F8 !important;
    margin:6px 0 12px 0 !important;
}

/* ── Panel cards (tips / format boxes) ───────────────── */
.panel-card {
    background:rgba(9,20,34,0.92);
    border:1px solid #1E3450;
    border-radius:14px;
    padding:18px 18px;
    box-shadow:0 8px 24px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.03);
}
.panel-title {
    color:#48CAE4;
    font-weight:700;
    margin-bottom:10px;
    font-size:0.92rem;
}
.panel-body {
    color:#8899AA;
    font-size:0.82rem;
    line-height:1.75;
}

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(9,20,34,0.85);
    border-radius:12px;
    padding:5px 7px;
    gap:5px;
    border:1px solid #1E3450;
}
.stTabs [data-baseweb="tab"] {
    border-radius:9px;
    padding:9px 16px;
    font-weight:600;
    font-size:0.88rem;
    color:#8899AA !important;
    background:transparent;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#48CAE4,#00B4D8) !important;
    color:#061018 !important;
    box-shadow:0 6px 18px rgba(0,180,216,0.28);
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
    background:linear-gradient(135deg,#48CAE4,#00B4D8);
    color:#061018;
    font-weight:700;
    border:none;
    border-radius:11px;
    padding:11px 26px;
    font-size:1rem;
    width:100%;
    transition:filter 0.2s, transform 0.12s, box-shadow 0.2s;
    box-shadow:0 10px 26px rgba(0,180,216,0.22);
}
.stButton > button:hover {
    filter:brightness(1.06);
    transform:translateY(-1px);
    box-shadow:0 14px 32px rgba(0,180,216,0.3);
}

/* ── Text inputs ────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background:#10283F !important;
    color:#E8F0F8 !important;
    border:1px solid #1E3450 !important;
    border-radius:11px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:#00B4D8 !important;
    box-shadow:0 0 0 3px #00B4D835 !important;
}

/* ── Selectbox / slider ─────────────────────────────── */
[data-baseweb="select"] > div {
    background:#10283F !important;
    border-color:#1E3450 !important;
    border-radius:10px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background:#00B4D8 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBarMin"], 
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBarMax"] {
    background:rgba(0,180,216,0.35) !important;
}

/* ── Progress ─────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background:linear-gradient(90deg,#48CAE4,#00B4D8) !important;
}

/* ── Streamlit metrics ───────────────────────────────── */
[data-testid="stMetricValue"] {
    color:#E8F0F8 !important;
    font-weight:800 !important;
}

/* ── Labels ─────────────────────────────────────────── */
label, .stTextArea label, .stTextInput label,
[data-testid="stWidgetLabel"] { color:#8899AA !important; font-weight:600; }

/* ── Result card ────────────────────────────────────── */
.result-card {
    border-radius:16px;
    padding:30px 36px;
    margin-top:20px;
    text-align:center;
    animation: fadeIn .4s ease;
}
.card-fake {
    background:linear-gradient(135deg,#3D0C11 0%,#1A0508 100%);
    border:2px solid #EF476F;
    box-shadow:0 0 28px #EF476F38, inset 0 1px 0 rgba(255,255,255,0.04);
}
.card-real {
    background:linear-gradient(135deg,#043D25 0%,#021A10 100%);
    border:2px solid #06D6A0;
    box-shadow:0 0 28px #06D6A038, inset 0 1px 0 rgba(255,255,255,0.04);
}
.verdict       { font-size:2.6rem; font-weight:800; letter-spacing:3px; }
.verdict-fake  { color:#EF476F; }
.verdict-real  { color:#06D6A0; }
.conf-label    { color:#8899AA; font-size:0.9rem; margin-top:6px; }

/* ── Metric row ─────────────────────────────────────── */
.metric-row {
    display:flex; gap:14px; margin-top:18px; justify-content:center; flex-wrap:wrap;
}
.metric-box {
    background:rgba(16,40,63,0.9);
    border-radius:12px;
    padding:14px 24px;
    text-align:center;
    min-width:120px;
    border:1px solid #1E3450;
}
.metric-val  { font-size:1.55rem; font-weight:700; }
.metric-name { font-size:0.75rem; color:#8899AA; margin-top:3px; }
.interp      { color:#8899AA; font-size:0.88rem; margin-top:14px; font-style:italic; }

/* ── Stat pill ──────────────────────────────────────── */
.stat-pill {
    display:inline-block;
    border-radius:20px;
    padding:5px 14px;
    font-size:0.78rem;
    font-weight:700;
    margin:3px;
}
.pill-total { background:#10283F; color:#48CAE4; border:1px solid #00B4D8; }
.pill-fake  { background:#3D0C1120; color:#EF476F; border:1px solid #EF476F; }
.pill-real  { background:#04452A20; color:#06D6A0; border:1px solid #06D6A0; }

/* ── History row ────────────────────────────────────── */
.hist-row {
    display:flex;
    align-items:center;
    gap:12px;
    padding:11px 15px;
    border-radius:11px;
    margin-bottom:7px;
    background:rgba(16,40,63,0.72);
    border:1px solid #1E3450;
    transition:background 0.15s, border-color 0.15s;
}
.hist-row:hover { background:#152E48; border-color:#243E63; }
.hist-badge {
    font-size:0.72rem;
    font-weight:700;
    padding:4px 11px;
    border-radius:12px;
    min-width:50px;
    text-align:center;
    flex-shrink:0;
}
.badge-fake { background:#EF476F22; color:#EF476F; border:1px solid #EF476F; }
.badge-real { background:#06D6A022; color:#06D6A0; border:1px solid #06D6A0; }
.hist-text  { color:#CCD6E0; font-size:0.88rem; flex:1; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.hist-conf  { color:#8899AA; font-size:0.82rem; flex-shrink:0; }

/* ── Dashboard KPI card ─────────────────────────────── */
.kpi-card {
    background:rgba(16,40,63,0.72);
    border-radius:14px;
    padding:20px 14px;
    text-align:center;
    border-top:3px solid;
    border:1px solid #1E3450;
    border-top-width:3px;
    box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);
}
.kpi-val   { font-size:2rem; font-weight:800; }
.kpi-label { color:#8899AA; font-size:0.78rem; margin-top:5px; }

/* ── Pipeline step cards ─────────────────────────────── */
.pipe-card {
    background:rgba(9,20,34,0.92);
    border:1px solid #1E3450;
    border-radius:14px;
    padding:15px 10px;
    text-align:center;
    box-shadow:0 8px 22px rgba(0,0,0,0.18);
    transition:transform 0.15s, border-color 0.15s;
}
.pipe-card:hover {
    transform:translateY(-2px);
    border-color:#2A4A73;
}

/* ── Confidence threshold warning ───────────────────── */
.low-conf {
    background:#3D3800;
    border:1px solid #FFD166;
    border-radius:11px;
    padding:12px 17px;
    margin-top:14px;
    color:#FFD166;
    font-size:0.9rem;
}

/* ── Fade-in animation ───────────────────────────────── */
@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

/* ── Divider ────────────────────────────────────────── */
hr { border-color:#1E3450; opacity:0.85; }

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploader"] {
    border:2px dashed rgba(30,52,80,0.95) !important;
    border-radius:14px !important;
    background:rgba(9,20,34,0.65) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color:rgba(0,180,216,0.55) !important;
}

/* ── Captions ───────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color:#6B7C90 !important; }

/* ── Scrollbars (WebKit) ─────────────────────────────── */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-track { background:#091422; }
::-webkit-scrollbar-thumb {
    background:#1E3450;
    border-radius:10px;
    border:2px solid #091422;
}
::-webkit-scrollbar-thumb:hover { background:#2A4A73; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("history",     []),
    ("total",       0),
    ("fake_count",  0),
    ("real_count",  0),
    ("last_text",   ""),
    ("last_ml_result", None),
    ("gemini_article_box", ""),
    ("gemini_msgs", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached — runs only once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    if not os.path.isdir(MODELS_DIR):
        return None, None, None
    for fname in sorted(os.listdir(MODELS_DIR)):
        if fname.endswith(".pkl") and "vectorizer" not in fname:
            name = fname[:-4]
            model, vectorizer = load_model(name)
            return model, vectorizer, name
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# URL scraping
# ─────────────────────────────────────────────────────────────────────────────
def fetch_url_text(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        st.error(f"Could not fetch URL: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _record(result: dict, text: str):
    label    = result["label_name"].upper()
    conf_pct = (result.get("confidence") or 0.0) * 100
    preview  = text.replace("\n", " ")
    preview  = (preview[:90] + "…") if len(preview) > 90 else preview

    st.session_state.history.insert(0, {
        "label":   label,
        "conf":    conf_pct,
        "preview": preview,
    })
    st.session_state.total      += 1
    if label == "FAKE":
        st.session_state.fake_count += 1
    else:
        st.session_state.real_count += 1


def _render_result(result: dict, elapsed_ms: float, conf_threshold: int):
    label    = result["label_name"].upper()
    conf     = result.get("confidence") or 0.0
    conf_pct = conf * 100
    is_fake  = label == "FAKE"

    card_cls  = "card-fake"    if is_fake else "card-real"
    verd_cls  = "verdict-fake" if is_fake else "verdict-real"
    icon      = "🚨"           if is_fake else "✅"
    val_color = "#EF476F"      if is_fake else "#06D6A0"
    interp    = (
        "This article shows strong signs of misinformation based on language patterns."
        if is_fake else
        "This article matches the linguistic patterns of credible news sources."
    )

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="verdict {verd_cls}">{icon} &nbsp; {label}</div>
        <div class="conf-label">Model Confidence Score</div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val" style="color:{val_color}">{conf_pct:.1f}%</div>
                <div class="metric-name">Confidence</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color:#FFD166">{elapsed_ms:.0f} ms</div>
                <div class="metric-name">Analysis Time</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color:#48CAE4">{len((st.session_state.last_text or "").split()):,}</div>
                <div class="metric-name">Words</div>
            </div>
        </div>
        <div class="interp">{interp}</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(conf_pct), text=f"Confidence: {conf_pct:.1f}%")

    if conf_pct < conf_threshold:
        st.markdown(f"""
        <div class="low-conf">
            ⚠️ Confidence is below your threshold ({conf_threshold}%).
            Consider using a full article for a more reliable prediction.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
model, vectorizer, model_name = get_model()

with st.sidebar:
    st.markdown("""
    <h2 style="color:#00B4D8; margin-bottom:4px; font-weight:800; letter-spacing:-0.02em;">
        🔍 Fake News<br>Detector</h2>
    <p style="color:#5C6D82; font-size:0.78rem; margin-top:0; line-height:1.4;">
        AI-Based Detection · Weeks 6–10</p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Model info
    if model is None:
        st.error("No model found.\nRun `python main.py` first.")
        st.stop()

    st.markdown(f"""
    <div style="background:rgba(16,40,63,0.85); border-radius:12px; padding:14px 15px;
                margin-bottom:16px; border:1px solid #1E3450;
                box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);">
        <div style="color:#8899AA; font-size:0.72rem; margin-bottom:6px; letter-spacing:0.08em;
                    font-weight:700;">ACTIVE MODEL</div>
        <div style="color:#48CAE4; font-weight:800; font-size:1rem;">{model_name}.pkl</div>
        <div style="color:#8899AA; font-size:0.75rem; margin-top:8px;">Accuracy&nbsp;&nbsp;
            <span style="color:#06D6A0;font-weight:700;">99.22%</span></div>
        <div style="color:#8899AA; font-size:0.75rem;">F1 Score&nbsp;&nbsp;&nbsp;&nbsp;
            <span style="color:#06D6A0;font-weight:700;">99.25%</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Session stats
    st.markdown("**Session Statistics**")
    t = st.session_state.total
    f = st.session_state.fake_count
    r = st.session_state.real_count
    st.markdown(f"""
    <div style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:16px;">
        <span class="stat-pill pill-total">Total&nbsp;{t}</span>
        <span class="stat-pill pill-fake">Fake&nbsp;{f}</span>
        <span class="stat-pill pill-real">Real&nbsp;{r}</span>
    </div>
    """, unsafe_allow_html=True)

    # Confidence threshold
    st.markdown("**Settings**")
    conf_threshold = st.slider(
        "Min. confidence threshold (%)",
        min_value=50, max_value=99, value=70, step=1,
        help="Predictions below this confidence will show a warning.",
    )

    st.markdown("**Gemini (Week 9–10)**")
    if resolve_gemini_api_key():
        st.caption("API key OK · summaries / explain / chat enabled")
    else:
        st.caption("Set GEMINI_API_KEY or `.streamlit/secrets.toml`")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tips
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Text tab** — paste a full article  
        2. **URL tab** — enter a news article link  
        3. **Batch tab** — upload a CSV with articles  
        4. **Dashboard** — see model & dataset stats  
        5. **History** — review past predictions  
        6. **Gemini AI** — explanations, summaries, Q&A (needs API key)  
        """)

    st.markdown(
        '<p style="color:#334455; font-size:0.72rem; text-align:center; margin-top:22px;">'
        'Capstone · Weeks 9–10 (Gemini)</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-shell">
    <div class="hero-inner">
        <div class="hero-kicker">Week 9–10 · Gemini AI layer</div>
        <h1 class="hero-title"><span class="hero-icon">🔍</span>Fake News Detector</h1>
        <p class="hero-sub">
            Paste text, fetch from a URL, or run batch CSV — English news articles work best.
            Capstone design project.
        </p>
        <div class="hero-chips">
            <span class="hero-chip">📝 Text</span>
            <span class="hero-chip">🔗 URL</span>
            <span class="hero-chip">📂 Batch</span>
            <span class="hero-chip">⚡ ML scores</span>
            <span class="hero-chip hero-chip-accent">Gemini insights</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_text, tab_url, tab_batch, tab_dash, tab_hist, tab_gem = st.tabs([
    "📝 Text Input",
    "🔗 URL Input",
    "📂 Batch CSV",
    "📊 Dashboard",
    "📋 History",
    "✨ Gemini AI",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Text Input
# ═════════════════════════════════════════════════════════════════════════════
with tab_text:
    col_main, col_tip = st.columns([3, 1])

    with col_main:
        st.markdown("#### Paste your news article")
        article_text = st.text_area(
            label="article",
            placeholder="Paste the full news article here…",
            height=240,
            label_visibility="collapsed",
            key="ta_text",
        )

        word_count = len(article_text.split()) if article_text.strip() else 0
        char_count = len(article_text)
        st.caption(f"Words: **{word_count:,}**  ·  Characters: **{char_count:,}**")

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            run_text = st.button("🔍 Analyse Article", key="btn_text")

        if run_text:
            if not article_text.strip():
                st.warning("Please paste some article text first.")
            else:
                with st.spinner("Analysing…"):
                    t0 = time.perf_counter()
                    result = predict(article_text, model, vectorizer)
                    ms = (time.perf_counter() - t0) * 1000
                st.session_state.last_text = article_text
                st.session_state.last_ml_result = {
                    "label": str(result.get("label_name") or "").upper(),
                    "conf_pct": (result.get("confidence") or 0.0) * 100,
                }
                _record(result, article_text)
                _render_result(result, ms, conf_threshold)

    with col_tip:
        st.markdown("""
        <div class="panel-card" style="margin-top:38px;">
            <div class="panel-title">💡 Tips</div>
            <div class="panel-body">
                • Use a <b>full article</b> (not just a headline) for best accuracy<br><br>
                • Longer text → more reliable prediction<br><br>
                • If confidence is low, try adding more context<br><br>
                • The model is trained on <b>English</b> news only
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — URL Input
# ═════════════════════════════════════════════════════════════════════════════
with tab_url:
    col_main, col_tip = st.columns([3, 1])

    with col_main:
        st.markdown("#### Enter a news article URL")
        url_input = st.text_input(
            label="url",
            placeholder="https://reuters.com/article/...",
            label_visibility="collapsed",
            key="ti_url",
        )

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            run_url = st.button("🔗 Fetch & Analyse", key="btn_url")

        if run_url:
            if not url_input.strip():
                st.warning("Please enter a URL first.")
            else:
                with st.spinner("Fetching article…"):
                    text = fetch_url_text(url_input.strip())
                if text:
                    wc = len(text.split())
                    st.caption(f"Extracted **{wc:,}** words from the page.")
                    with st.spinner("Analysing…"):
                        t0 = time.perf_counter()
                        result = predict(text, model, vectorizer)
                        ms = (time.perf_counter() - t0) * 1000
                    st.session_state.last_text = text
                    st.session_state.last_ml_result = {
                        "label": str(result.get("label_name") or "").upper(),
                        "conf_pct": (result.get("confidence") or 0.0) * 100,
                    }
                    _record(result, text)
                    _render_result(result, ms, conf_threshold)

    with col_tip:
        st.markdown("""
        <div class="panel-card" style="margin-top:38px;">
            <div class="panel-title">🔗 Supported sites</div>
            <div class="panel-body">
                • reuters.com<br>
                • bbc.com/news<br>
                • apnews.com<br>
                • theguardian.com<br>
                • cnn.com<br>
                • Any public news URL
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch CSV
# ═════════════════════════════════════════════════════════════════════════════
with tab_batch:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Upload a CSV file")
        st.caption("The CSV must have a column containing article text. Results will be available for download.")

        uploaded = st.file_uploader(
            "Drop CSV here or click to browse",
            type=["csv"],
            label_visibility="collapsed",
        )

    with col_right:
        st.markdown("""
        <div class="panel-card" style="margin-top:40px;">
            <div class="panel-title">📋 CSV format</div>
            <div class="panel-body">
                Required: a column with article text<br><br>
                Example columns:<br>
                <code style="color:#FFD166;">text</code>,
                <code style="color:#FFD166;">article</code>,
                <code style="color:#FFD166;">content</code><br><br>
                Output adds two columns:<br>
                <code style="color:#06D6A0;">prediction</code><br>
                <code style="color:#06D6A0;">confidence_%</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
            st.markdown(f"**Loaded:** `{uploaded.name}`  —  {len(df_input):,} rows, {len(df_input.columns)} columns")

            cols_available = list(df_input.columns)
            default_col    = "text" if "text" in cols_available else cols_available[0]
            text_col       = st.selectbox("Select the text column:", cols_available,
                                          index=cols_available.index(default_col))

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                run_batch = st.button("⚡ Run Batch Analysis", key="btn_batch")

            if run_batch:
                predictions = []
                confidences = []
                progress_bar = st.progress(0, text="Starting…")
                total_rows   = len(df_input)
                fake_b = real_b = skip_b = 0

                for row_idx, (_, row) in enumerate(df_input.iterrows(), start=1):
                    text = str(row.get(text_col, "") or "").strip()
                    pct  = int(row_idx / total_rows * 100)
                    progress_bar.progress(pct, text=f"Analysing row {row_idx} / {total_rows}…")

                    if not text:
                        predictions.append("SKIPPED")
                        confidences.append("—")
                        skip_b += 1
                        continue

                    res  = predict(text, model, vectorizer)
                    lbl  = res["label_name"].upper()
                    conf = (res.get("confidence") or 0.0) * 100
                    predictions.append(lbl)
                    confidences.append(f"{conf:.1f}%")
                    if lbl == "FAKE":
                        fake_b += 1
                    else:
                        real_b += 1

                progress_bar.progress(100, text="Done!")

                df_output = df_input.copy()
                df_output["prediction"]   = predictions
                df_output["confidence_%"] = confidences

                # Summary metrics
                st.markdown("")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Rows",   f"{total_rows:,}")
                m2.metric("Fake",         f"{fake_b:,}",  delta=f"{fake_b/max(total_rows,1)*100:.1f}%")
                m3.metric("Real",         f"{real_b:,}",  delta=f"{real_b/max(total_rows,1)*100:.1f}%")
                m4.metric("Skipped",      f"{skip_b:,}")

                st.markdown("#### Preview (first 20 rows)")
                st.dataframe(
                    df_output[["prediction", "confidence_%"] + [text_col]]
                    .head(20)
                    .style.map(
                        lambda v: "color:#EF476F;font-weight:700" if v == "FAKE"
                                  else ("color:#06D6A0;font-weight:700" if v == "REAL" else ""),
                        subset=["prediction"],
                    ),
                    use_container_width=True,
                )

                # Download
                csv_bytes = df_output.to_csv(index=False).encode("utf-8")
                fname_out = uploaded.name.replace(".csv", "_predictions.csv")
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.download_button(
                        label     = "⬇️ Download Predictions CSV",
                        data      = csv_bytes,
                        file_name = fname_out,
                        mime      = "text/csv",
                        key       = "dl_batch",
                    )

                # Update session stats
                st.session_state.total      += fake_b + real_b
                st.session_state.fake_count += fake_b
                st.session_state.real_count += real_b

        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dashboard
# ═════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("#### Model & Dataset Overview")
    st.markdown("")

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("#06D6A0", "99.22%",  "Accuracy"),
        ("#48CAE4", "99.49%",  "Precision"),
        ("#FFD166", "99.02%",  "Recall"),
        ("#06D6A0", "99.25%",  "F1 Score"),
        ("#EF476F", "99.96%",  "ROC-AUC"),
    ]
    for col, (color, val, name) in zip([k1,k2,k3,k4,k5], kpis):
        col.markdown(f"""
        <div class="kpi-card" style="border-top-color:{color};">
            <div class="kpi-val" style="color:{color};">{val}</div>
            <div class="kpi-label">{name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Two charts side by side ───────────────────────────────────────────────
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("##### Dataset Composition")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor("#0D1B2A")
        ax.set_facecolor("#0D1B2A")

        categories = ["Fake News", "Real News"]
        counts     = [23481, 21417]
        colors     = ["#EF476F", "#06D6A0"]
        bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor="none")

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f"{cnt:,}", ha="center", va="bottom",
                    color="white", fontsize=10, fontweight="bold")

        ax.set_ylim(0, 27000)
        ax.tick_params(colors="white", labelsize=10)
        ax.spines[:].set_visible(False)
        ax.yaxis.set_visible(False)
        fig.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Total: 44,898 articles  ·  Fake: 52.3%  ·  Real: 47.7%")

    with chart_right:
        st.markdown("##### Model Comparison")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor("#0D1B2A")
        ax.set_facecolor("#0D1B2A")

        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        lr_vals = [99.22, 99.49, 99.02, 99.25]
        nb_vals = [95.14, 95.47, 95.23, 95.35]
        x = np.arange(len(metrics))
        w = 0.32

        bars_lr = ax.bar(x - w/2, lr_vals, width=w, label="Logistic Regression",
                         color="#00B4D8", edgecolor="none")
        bars_nb = ax.bar(x + w/2, nb_vals, width=w, label="Naive Bayes",
                         color="#FFD166", edgecolor="none")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, color="white", fontsize=9)
        ax.set_ylim(90, 102)
        ax.tick_params(axis="y", colors="white", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.legend(loc="lower right", facecolor="#0D1B2A",
                  labelcolor="white", fontsize=8, framealpha=0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        fig.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close(fig)

    # ── Pipeline diagram (text-based) ─────────────────────────────────────────
    st.markdown("")
    st.markdown("##### ML Pipeline")
    steps = [
        ("📥", "Load Data",     "44,898 articles\nFake.csv + True.csv"),
        ("🧹", "Preprocess",    "Lowercase · URLs\nStopwords · Stemming"),
        ("📐", "TF-IDF",        "5,000 features\nUnigrams + Bigrams"),
        ("🤖", "Train",         "Logistic Regression\nNaive Bayes"),
        ("📊", "Evaluate",      "Accuracy · F1\nROC-AUC · Confusion"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class="pipe-card">
            <div style="font-size:1.85rem;">{icon}</div>
            <div style="color:#48CAE4; font-weight:700; font-size:0.85rem; margin:8px 0 5px;">{title}</div>
            <div style="color:#8899AA; font-size:0.74rem; line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Dataset source info ────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("##### Dataset Details")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        | Field | Value |
        |---|---|
        | Source | Kaggle — Fake and Real News Dataset |
        | Credit | Ahmed, Hadeer et al. (2017) |
        | Date Range | 2015 – 2018 |
        | Real news source | Reuters.com |
        | Fake news source | PolitiFact + others |
        | Total articles | 44,898 |
        """)
    with d2:
        st.markdown("""
        | Metric | Logistic Regression | Naive Bayes |
        |---|---|---|
        | Accuracy | **99.22%** | 95.14% |
        | Precision | **99.49%** | 95.47% |
        | Recall | **99.02%** | 95.23% |
        | F1 Score | **99.25%** | 95.35% |
        | ROC-AUC | **99.96%** | 98.78% |
        """)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — History
# ═════════════════════════════════════════════════════════════════════════════
with tab_hist:
    history = st.session_state.history

    if not history:
        st.markdown("""
        <div style="text-align:center; padding:52px 18px;">
            <div style="font-size:3.2rem; opacity:0.85;">📋</div>
            <div style="margin-top:14px; font-size:1.05rem; font-weight:600; color:#CCD6E0;">
                No predictions yet
            </div>
            <div style="font-size:0.88rem; margin-top:8px; color:#5C6D82; max-width:320px; margin-left:auto; margin-right:auto;">
                Run an analysis from the Text or URL tab — results appear here for this session.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        h_col, btn_col = st.columns([3, 1])
        with h_col:
            st.markdown(f"**{len(history)} prediction(s) this session**")
        with btn_col:
            if st.button("🗑️ Clear History", key="btn_clear"):
                st.session_state.history    = []
                st.session_state.total      = 0
                st.session_state.fake_count = 0
                st.session_state.real_count = 0
                st.rerun()

        # Summary mini-chart
        if len(history) >= 2:
            fig, ax = plt.subplots(figsize=(4, 1.4))
            fig.patch.set_facecolor("#091422")
            ax.set_facecolor("#091422")
            f_cnt = sum(1 for h in history if h["label"] == "FAKE")
            r_cnt = len(history) - f_cnt
            ax.barh(["REAL", "FAKE"], [r_cnt, f_cnt],
                    color=["#06D6A0", "#EF476F"], edgecolor="none", height=0.5)
            ax.spines[:].set_visible(False)
            ax.tick_params(colors="white", labelsize=9)
            ax.xaxis.set_visible(False)
            fig.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        st.markdown("")
        for h in history:
            badge_cls = "badge-fake" if h["label"] == "FAKE" else "badge-real"
            icon      = "🚨" if h["label"] == "FAKE" else "✅"
            st.markdown(f"""
            <div class="hist-row">
                <span class="hist-badge {badge_cls}">{icon} {h['label']}</span>
                <span class="hist-text">{h['preview']}</span>
                <span class="hist-conf">{h['conf']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Download history as CSV
        st.markdown("")
        hist_df   = pd.DataFrame(history)
        hist_csv  = hist_df.to_csv(index=False).encode("utf-8")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.download_button(
                "⬇️ Export History CSV",
                hist_csv,
                "prediction_history.csv",
                "text/csv",
                key="dl_hist",
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — Gemini AI (Week 9–10)
# ═════════════════════════════════════════════════════════════════════════════
with tab_gem:
    st.markdown("#### Gemini AI — Explain · Summarise · Ask")
    api_k = resolve_gemini_api_key()

    if not api_k:
        st.warning(
            "Add a Google Gemini API key to enable this tab: set **GEMINI_API_KEY** "
            "(or **GOOGLE_API_KEY**) in your environment, or add it to "
            "`.streamlit/secrets.toml`. Get a key from "
            "[Google AI Studio](https://aistudio.google.com/apikey)."
        )
    else:
        st.caption(
            "Uses Google's Gemini API. Override model with env **GEMINI_MODEL** "
            "(otherwise tries gemini-2.0-flash, then gemini-1.5-flash)."
        )

    row_top = st.columns([2, 1])
    with row_top[1]:
        if st.button("📥 Load last analysed article", key="btn_gemini_load"):
            st.session_state.gemini_article_box = st.session_state.last_text or ""
            st.rerun()
        if st.button("🗑️ Clear chat", key="btn_gemini_clr_chat"):
            st.session_state.gemini_msgs = []
            st.rerun()
        if st.button("Clear explain/summary panels", key="btn_gemini_clr_panel"):
            st.session_state.pop("gemini_panel_explain", None)
            st.session_state.pop("gemini_panel_summary", None)
            st.rerun()

    article_ctx = st.text_area(
        "Article text for Gemini",
        height=220,
        key="gemini_article_box",
        placeholder="Paste an article, or load the last one from Text / URL analysis.",
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        do_explain = st.button(
            "✨ Explain ML verdict",
            key="btn_gemini_explain",
            disabled=not api_k,
            help="Uses the last TF-IDF prediction from Text/URL plus article text.",
        )
    with bc2:
        do_summary = st.button(
            "📄 Summarise article",
            key="btn_gemini_summary",
            disabled=not api_k,
        )

    # Prefer what the user explicitly pasted in the Gemini tab; fall back to last ML text
    text_for_explain = article_ctx.strip() or (st.session_state.last_text or "").strip()

    if do_explain and api_k:
        if not text_for_explain:
            st.warning("No article text — paste content or analyse an article first.")
        elif not st.session_state.last_ml_result:
            st.warning("Run **Analyse** on the Text or URL tab first so an ML verdict exists.")
        else:
            with st.spinner("Gemini is reasoning…"):
                try:
                    lm = st.session_state.last_ml_result
                    pr = build_explain_prompt(
                        text_for_explain,
                        lm["label"],
                        lm["conf_pct"],
                    )
                    st.session_state.gemini_panel_explain = generate_gemini_text(api_k, pr)
                except Exception as e:
                    st.error(str(e))

    if do_summary and api_k:
        if not article_ctx.strip():
            st.warning("Paste article text in the box above.")
        else:
            with st.spinner("Gemini is summarising…"):
                try:
                    pr = build_summarise_prompt(article_ctx)
                    st.session_state.gemini_panel_summary = generate_gemini_text(api_k, pr)
                except Exception as e:
                    st.error(str(e))

    ex_out = st.session_state.get("gemini_panel_explain")
    if ex_out:
        st.markdown("##### ML verdict explanation")
        st.markdown(ex_out)

    sm_out = st.session_state.get("gemini_panel_summary")
    if sm_out:
        st.markdown("##### Summary")
        st.markdown(sm_out)

    st.markdown("##### Ask questions about the article")
    if not api_k:
        st.caption("Configure GEMINI_API_KEY to enable chat.")
    elif not article_ctx.strip():
        st.caption("Paste article text above to unlock Q&A.")
    else:
        for msg in st.session_state.gemini_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if user_q := st.chat_input("Ask something about this article…"):
            st.session_state.gemini_msgs.append({"role": "user", "content": user_q})
            try:
                with st.spinner("Thinking…"):
                    ans = generate_gemini_text(
                        api_k,
                        build_chat_prompt(article_ctx, user_q),
                    )
                st.session_state.gemini_msgs.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.session_state.gemini_msgs.append(
                    {"role": "assistant", "content": f"(Error) {e}"}
                )
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#334455; font-size:0.78rem; margin-bottom:8px;">'
    'Capstone Design Project · AI-Based Fake News Detection · Weeks 9–10 (Gemini)&nbsp;·&nbsp;'
    '<a href="https://github.com/maharram-davidov/fakenewsproject" '
    'style="color:#48CAE4; text-decoration:none;">GitHub</a>'
    '</p>',
    unsafe_allow_html=True,
)
