"""
app/web_app.py
==============
Week 3 (Improved) — Streamlit Web Application
AI-Based Fake News Detection System

Run:
    streamlit run app/web_app.py
"""

import sys
import os
import io
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
/* ── Base ──────────────────────────────────────────── */
.stApp                    { background-color:#0D1B2A; color:#E8F0F8; }
#MainMenu, footer, header { visibility:hidden; }

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#091422 0%,#0D1F35 100%);
    border-right: 1px solid #1E3450;
}
[data-testid="stSidebar"] * { color:#CCD6E0 !important; }

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background:#091422;
    border-radius:10px;
    padding:4px 6px;
    gap:4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:7px;
    padding:8px 18px;
    font-weight:600;
    color:#8899AA !important;
    background:transparent;
}
.stTabs [aria-selected="true"] {
    background:#00B4D8 !important;
    color:#0D1B2A !important;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
    background:#00B4D8; color:#0D1B2A;
    font-weight:700; border:none;
    border-radius:8px; padding:10px 28px;
    font-size:1rem; width:100%;
    transition:background 0.2s, transform 0.1s;
}
.stButton > button:hover { background:#48CAE4; transform:translateY(-1px); }

/* ── Text inputs ────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background:#10283F !important;
    color:#E8F0F8 !important;
    border:1px solid #1E3450 !important;
    border-radius:8px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:#00B4D8 !important;
    box-shadow:0 0 0 2px #00B4D840 !important;
}

/* ── Labels ─────────────────────────────────────────── */
label, .stTextArea label, .stTextInput label,
[data-testid="stWidgetLabel"] { color:#8899AA !important; font-weight:600; }

/* ── Result card ────────────────────────────────────── */
.result-card {
    border-radius:14px; padding:30px 36px;
    margin-top:20px; text-align:center;
    animation: fadeIn .4s ease;
}
.card-fake {
    background:linear-gradient(135deg,#3D0C11 0%,#1A0508 100%);
    border:2px solid #EF476F;
    box-shadow:0 0 24px #EF476F33;
}
.card-real {
    background:linear-gradient(135deg,#043D25 0%,#021A10 100%);
    border:2px solid #06D6A0;
    box-shadow:0 0 24px #06D6A033;
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
    background:#10283F; border-radius:10px;
    padding:14px 24px; text-align:center; min-width:120px;
}
.metric-val  { font-size:1.55rem; font-weight:700; }
.metric-name { font-size:0.75rem; color:#8899AA; margin-top:3px; }
.interp      { color:#8899AA; font-size:0.88rem; margin-top:14px; font-style:italic; }

/* ── Stat pill ──────────────────────────────────────── */
.stat-pill {
    display:inline-block; border-radius:20px;
    padding:4px 14px; font-size:0.8rem; font-weight:700; margin:3px;
}
.pill-total { background:#10283F;  color:#48CAE4; border:1px solid #00B4D8; }
.pill-fake  { background:#3D0C1120; color:#EF476F; border:1px solid #EF476F; }
.pill-real  { background:#04452A20; color:#06D6A0; border:1px solid #06D6A0; }

/* ── History row ────────────────────────────────────── */
.hist-row {
    display:flex; align-items:center; gap:12px;
    padding:10px 14px; border-radius:8px; margin-bottom:6px;
    background:#10283F; border:1px solid #1E3450;
    transition:background 0.15s;
}
.hist-row:hover { background:#152E48; }
.hist-badge {
    font-size:0.72rem; font-weight:700; padding:3px 11px;
    border-radius:12px; min-width:50px; text-align:center; flex-shrink:0;
}
.badge-fake { background:#EF476F22; color:#EF476F; border:1px solid #EF476F; }
.badge-real { background:#06D6A022; color:#06D6A0; border:1px solid #06D6A0; }
.hist-text  { color:#CCD6E0; font-size:0.88rem; flex:1; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.hist-conf  { color:#8899AA; font-size:0.82rem; flex-shrink:0; }

/* ── Dashboard KPI card ─────────────────────────────── */
.kpi-card {
    background:#10283F; border-radius:12px;
    padding:20px 16px; text-align:center;
    border-top:3px solid;
}
.kpi-val   { font-size:2rem; font-weight:800; }
.kpi-label { color:#8899AA; font-size:0.8rem; margin-top:4px; }

/* ── Confidence threshold warning ───────────────────── */
.low-conf {
    background:#3D3800; border:1px solid #FFD166;
    border-radius:8px; padding:10px 16px; margin-top:14px;
    color:#FFD166; font-size:0.9rem;
}

/* ── Fade-in animation ───────────────────────────────── */
@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

/* ── Divider ────────────────────────────────────────── */
hr { border-color:#1E3450; }

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploader"] {
    border:2px dashed #1E3450 !important;
    border-radius:10px !important;
    background:#091422 !important;
}
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
                <div class="metric-val" style="color:#48CAE4">{len(st.session_state.last_text.split()):,}</div>
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
    <h2 style="color:#00B4D8; margin-bottom:2px;">🔍 Fake News<br>Detector</h2>
    <p style="color:#445566; font-size:0.8rem; margin-top:0;">AI-Based Detection System</p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Model info
    if model is None:
        st.error("No model found.\nRun `python main.py` first.")
        st.stop()

    st.markdown(f"""
    <div style="background:#10283F; border-radius:8px; padding:12px 14px; margin-bottom:16px;">
        <div style="color:#8899AA; font-size:0.75rem; margin-bottom:4px;">ACTIVE MODEL</div>
        <div style="color:#48CAE4; font-weight:700;">{model_name}.pkl</div>
        <div style="color:#8899AA; font-size:0.75rem; margin-top:6px;">Accuracy&nbsp;&nbsp;<span style="color:#06D6A0;font-weight:700;">99.22%</span></div>
        <div style="color:#8899AA; font-size:0.75rem;">F1 Score&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#06D6A0;font-weight:700;">99.25%</span></div>
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

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tips
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Text tab** — paste a full article  
        2. **URL tab** — enter a news article link  
        3. **Batch tab** — upload a CSV with articles  
        4. **Dashboard** — see model & dataset stats  
        5. **History** — review past predictions  
        """)

    st.markdown(
        '<p style="color:#334455; font-size:0.72rem; text-align:center; margin-top:20px;">'
        'Capstone Project · Week 3</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="color:#00B4D8; font-size:2rem; margin-bottom:2px; margin-top:-10px;">
    🔍 Fake News Detector
</h1>
<p style="color:#8899AA; font-size:0.95rem; margin-bottom:22px;">
    AI-Based Detection System &nbsp;·&nbsp; Capstone Design Project
</p>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_text, tab_url, tab_batch, tab_dash, tab_hist = st.tabs([
    "📝 Text Input",
    "🔗 URL Input",
    "📂 Batch CSV",
    "📊 Dashboard",
    "📋 History",
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
                _record(result, article_text)
                _render_result(result, ms, conf_threshold)

    with col_tip:
        st.markdown("""
        <div style="background:#091422; border:1px solid #1E3450; border-radius:10px; padding:16px; margin-top:34px;">
            <div style="color:#48CAE4; font-weight:700; margin-bottom:10px;">💡 Tips</div>
            <div style="color:#8899AA; font-size:0.82rem; line-height:1.7;">
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
                    _record(result, text)
                    _render_result(result, ms, conf_threshold)

    with col_tip:
        st.markdown("""
        <div style="background:#091422; border:1px solid #1E3450; border-radius:10px; padding:16px; margin-top:34px;">
            <div style="color:#48CAE4; font-weight:700; margin-bottom:10px;">🔗 Supported sites</div>
            <div style="color:#8899AA; font-size:0.82rem; line-height:1.8;">
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
        <div style="background:#091422; border:1px solid #1E3450; border-radius:10px; padding:16px; margin-top:36px;">
            <div style="color:#48CAE4; font-weight:700; margin-bottom:8px;">📋 CSV Format</div>
            <div style="color:#8899AA; font-size:0.82rem; line-height:1.8;">
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
                status_area  = st.empty()
                total_rows   = len(df_input)
                fake_b = real_b = skip_b = 0

                for i, row in df_input.iterrows():
                    text = str(row.get(text_col, "") or "").strip()
                    pct  = int(((list(df_input.index).index(i) + 1) / total_rows) * 100)
                    progress_bar.progress(pct, text=f"Analysing row {list(df_input.index).index(i)+1} / {total_rows}…")

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
                    .style.applymap(
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
        <div style="background:#091422; border:1px solid #1E3450; border-radius:10px;
                    padding:14px 10px; text-align:center;">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="color:#48CAE4; font-weight:700; font-size:0.85rem; margin:6px 0 4px;">{title}</div>
            <div style="color:#8899AA; font-size:0.75rem; line-height:1.5;">{desc}</div>
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
        <div style="text-align:center; padding:60px 0; color:#445566;">
            <div style="font-size:3rem;">📋</div>
            <div style="margin-top:12px; font-size:1rem;">No predictions yet.</div>
            <div style="font-size:0.85rem; margin-top:6px;">Analyse an article in the Text or URL tab.</div>
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


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#334455; font-size:0.78rem;">'
    'Capstone Design Project &nbsp;·&nbsp; AI-Based Fake News Detection System'
    '&nbsp;·&nbsp; Week 3 (Improved) &nbsp;·&nbsp; '
    'github.com/maharram-davidov/fakenewsproject'
    '</p>',
    unsafe_allow_html=True,
)
