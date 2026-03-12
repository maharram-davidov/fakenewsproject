"""
create_presentation.py
======================
Generates the capstone project PowerPoint presentation.
Output: FakeNewsDetection_Capstone.pptx

Run:
    python create_presentation.py
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.dml import MSO_THEME_COLOR
import pptx.oxml.ns as nsmap
from lxml import etree

# ── Colour Palette ─────────────────────────────────────────────────────────────
NAVY        = RGBColor(0x0D, 0x1B, 0x2A)   # slide background
DARK_BLUE   = RGBColor(0x0A, 0x29, 0x5C)   # header bars
ACCENT_BLUE = RGBColor(0x00, 0xB4, 0xD8)   # highlights / bullets
ACCENT_CYAN = RGBColor(0x48, 0xCA, 0xE4)
GREEN       = RGBColor(0x06, 0xD6, 0xA0)   # "Real" / positive
RED         = RGBColor(0xEF, 0x47, 0x6F)   # "Fake" / warning
YELLOW      = RGBColor(0xFF, 0xD1, 0x66)   # highlights
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xCC, 0xD6, 0xE0)
MID_GRAY    = RGBColor(0x88, 0x99, 0xAA)

SW = 10.0   # slide width  inches
SH = 5.63   # slide height inches

prs = Presentation()
prs.slide_width  = Inches(SW)
prs.slide_height = Inches(SH)

blank_layout = prs.slide_layouts[6]   # completely blank


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def add_rect(slide, x, y, w, h, fill_rgb=None, alpha=None):
    """Add a filled rectangle shape."""
    shape = slide.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE_TYPE.AUTO_SHAPE if False else 1,  # MSO_SHAPE.RECTANGLE=1
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.line.fill.background()          # no border
    if fill_rgb:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_rgb
    else:
        shape.fill.background()
    return shape


def add_text(slide, text, x, y, w, h,
             font_size=18, bold=False, italic=False,
             color=WHITE, align=PP_ALIGN.LEFT,
             font_name="Segoe UI"):
    """Add a text box."""
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name  = font_name
    return txb


def add_multiline(slide, lines, x, y, w, h,
                  font_size=14, color=WHITE,
                  bullet_color=ACCENT_BLUE,
                  font_name="Segoe UI",
                  line_spacing=1.2):
    """Add a text box with multiple bullet lines."""
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = True
    for i, (bullet, line) in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        prefix = f"{bullet}  " if bullet else ""
        run.text = prefix + line
        run.font.size  = Pt(font_size)
        run.font.color.rgb = color
        run.font.name  = font_name
    return txb


def add_badge(slide, text, x, y, w=1.5, h=0.38,
              bg=ACCENT_BLUE, fg=NAVY, font_size=11):
    """Pill-shaped info badge (rectangle with text)."""
    add_rect(slide, x, y, w, h, fill_rgb=bg)
    add_text(slide, text, x, y+0.02, w, h,
             font_size=font_size, bold=True, color=fg,
             align=PP_ALIGN.CENTER)


def slide_bg(slide, color=NAVY):
    """Fill slide background."""
    bg_shape = add_rect(slide, 0, 0, SW, SH, fill_rgb=color)
    slide.shapes._spTree.remove(bg_shape._element)
    slide.shapes._spTree.insert(2, bg_shape._element)


def h_rule(slide, y, color=ACCENT_BLUE, thickness=0.03):
    add_rect(slide, 0.4, y, SW - 0.8, thickness, fill_rgb=color)


def section_header_bar(slide, title, subtitle=""):
    """Full-width dark bar with section title."""
    add_rect(slide, 0, 0, SW, 1.05, fill_rgb=DARK_BLUE)
    add_text(slide, title, 0.4, 0.08, SW-0.8, 0.6,
             font_size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle, 0.4, 0.65, SW-0.8, 0.38,
                 font_size=13, color=ACCENT_BLUE, align=PP_ALIGN.LEFT)


def metric_box(slide, label, value, x, y, w=2.0, h=1.1,
               val_color=GREEN, bg=RGBColor(0x10, 0x28, 0x40)):
    add_rect(slide, x, y, w, h, fill_rgb=bg)
    add_rect(slide, x, y, w, 0.04, fill_rgb=val_color)   # top accent strip
    add_text(slide, value, x, y+0.05, w, 0.58,
             font_size=28, bold=True, color=val_color,
             align=PP_ALIGN.CENTER)
    add_text(slide, label, x, y+0.62, w, 0.38,
             font_size=10, color=LIGHT_GRAY,
             align=PP_ALIGN.CENTER)


def roadmap_card(slide, week_label, title, bullets,
                 x, y, w=1.75, h=3.4,
                 bar_color=ACCENT_BLUE, done=False):
    bg = RGBColor(0x10, 0x28, 0x40) if not done else RGBColor(0x05, 0x35, 0x28)
    add_rect(slide, x, y, w, h, fill_rgb=bg)
    add_rect(slide, x, y, w, 0.06, fill_rgb=bar_color)
    add_text(slide, week_label, x, y+0.08, w, 0.28,
             font_size=9, bold=True, color=bar_color,
             align=PP_ALIGN.CENTER)
    add_text(slide, title, x+0.08, y+0.35, w-0.16, 0.55,
             font_size=11, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    h_rule(slide, y+0.88, color=MID_GRAY, thickness=0.015)
    blines = [("›", b) for b in bullets]
    add_multiline(slide, blines, x+0.1, y+0.95, w-0.2, h-1.1,
                  font_size=9, color=LIGHT_GRAY, bullet_color=bar_color)
    if done:
        add_badge(slide, "✓ DONE", x+w-1.1, y+0.06, w=0.9, h=0.26,
                  bg=GREEN, fg=NAVY, font_size=9)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)

# Left accent strip
add_rect(slide, 0, 0, 0.08, SH, fill_rgb=ACCENT_BLUE)

# Diagonal decorative block top-right
add_rect(slide, 7.5, 0, 2.5, 2.0, fill_rgb=DARK_BLUE)
add_rect(slide, 8.2, 0, 1.8, 0.06, fill_rgb=ACCENT_BLUE)

# Tag line chips
add_badge(slide, "CAPSTONE PROJECT", 0.4, 0.28, w=2.0, h=0.32,
          bg=ACCENT_BLUE, fg=NAVY, font_size=10)
add_badge(slide, "WEEK 1", 2.55, 0.28, w=0.9, h=0.32,
          bg=RED, fg=WHITE, font_size=10)

# Main title
add_text(slide, "AI-Based Fake News", 0.4, 0.72, 9.0, 1.0,
         font_size=42, bold=True, color=WHITE)
add_text(slide, "Detection System", 0.4, 1.58, 9.0, 0.9,
         font_size=42, bold=True, color=ACCENT_BLUE)

h_rule(slide, 2.58, color=ACCENT_BLUE, thickness=0.04)

# Subtitle
add_text(slide, "Machine Learning Pipeline  ·  NLP  ·  Multi-Phase Roadmap",
         0.4, 2.72, 9.0, 0.5, font_size=15, color=LIGHT_GRAY)

# Bottom info row
add_rect(slide, 0, 4.8, SW, 0.83, fill_rgb=DARK_BLUE)
info_items = [
    ("Dataset",     "44,898 Articles"),
    ("Best Model",  "Logistic Regression"),
    ("Accuracy",    "99.22%"),
    ("F1 Score",    "99.25%"),
    ("ROC-AUC",     "99.96%"),
]
for i, (lbl, val) in enumerate(info_items):
    bx = 0.22 + i * 1.95
    add_text(slide, val,  bx, 4.82, 1.85, 0.4,
             font_size=13, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    add_text(slide, lbl,  bx, 5.15, 1.85, 0.28,
             font_size=9,  color=MID_GRAY,  align=PP_ALIGN.CENTER)

# Decorative dots
for i, (cx, cy) in enumerate([(8.3,1.1),(8.7,0.6),(9.2,1.4),(9.5,0.9),(8.9,1.8)]):
    dot = slide.shapes.add_shape(1, Inches(cx), Inches(cy),
                                 Inches(0.12+i*0.02), Inches(0.12+i*0.02))
    dot.fill.solid()
    dot.fill.fore_color.rgb = ACCENT_CYAN if i % 2 == 0 else ACCENT_BLUE
    dot.line.fill.background()


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — PROBLEM STATEMENT
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Problem Statement",
                   "Why does fake news detection matter?")

stats = [
    ("86%",   "of internet users\nhave been misled\nby fake news"),
    ("$78B",  "estimated annual\neconomic damage\nfrom misinformation"),
    ("6×",    "faster spread of\nfalse news vs\nreal news (MIT)"),
    ("2.3B",  "social media users\nexposed daily to\nunverified content"),
]
for i, (val, desc) in enumerate(stats):
    bx = 0.35 + i * 2.4
    add_rect(slide, bx, 1.25, 2.1, 2.2, fill_rgb=RGBColor(0x10,0x28,0x40))
    add_rect(slide, bx, 1.25, 2.1, 0.06,
             fill_rgb=[RED, YELLOW, ACCENT_BLUE, GREEN][i])
    add_text(slide, val, bx, 1.35, 2.1, 0.9,
             font_size=30, bold=True,
             color=[RED, YELLOW, ACCENT_BLUE, GREEN][i],
             align=PP_ALIGN.CENTER)
    add_text(slide, desc, bx+0.1, 2.22, 1.9, 1.0,
             font_size=11, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

h_rule(slide, 3.65, color=ACCENT_BLUE)

add_text(slide,
         "Traditional manual fact-checking cannot scale to the volume of online content. "
         "An AI-driven system can analyse thousands of articles per second with high accuracy.",
         0.4, 3.78, SW-0.8, 0.9,
         font_size=13, color=LIGHT_GRAY, italic=True)

add_text(slide, "→  Our Solution: ML + NLP pipeline for automated, real-time fake news detection",
         0.4, 4.58, SW-0.8, 0.5,
         font_size=13, bold=True, color=GREEN)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Dataset Overview",
                   "Kaggle — Fake and Real News Dataset (Ahmed et al.)")

# Left column — dataset facts
facts = [
    ("Total articles",   "44,898"),
    ("Fake articles",    "23,481  (52.3%)"),
    ("Real articles",    "21,417  (47.7%)"),
    ("Missing text",     "0 articles"),
    ("Source",           "Reuters.com + PolitiFact"),
    ("Date range",       "2015 – 2018"),
    ("Columns",          "title, text, subject, date, label"),
]
add_rect(slide, 0.3, 1.1, 4.3, 3.9, fill_rgb=RGBColor(0x10,0x28,0x40))
add_text(slide, "📋  Dataset Facts", 0.5, 1.15, 4.0, 0.4,
         font_size=13, bold=True, color=ACCENT_BLUE)
for i, (k, v) in enumerate(facts):
    ry = 1.55 + i * 0.48
    add_text(slide, k, 0.5, ry, 1.8, 0.38,
             font_size=11, color=MID_GRAY)
    add_text(slide, v, 2.35, ry, 2.15, 0.38,
             font_size=11, bold=True, color=WHITE)

# Right column — subject breakdown
add_rect(slide, 5.0, 1.1, 4.65, 3.9, fill_rgb=RGBColor(0x10,0x28,0x40))
add_text(slide, "📊  Subject Categories", 5.2, 1.15, 4.2, 0.4,
         font_size=13, bold=True, color=ACCENT_BLUE)

subjects = [
    ("politicsNews",     38, GREEN),
    ("worldnews",        22, ACCENT_BLUE),
    ("News",             18, YELLOW),
    ("politics",         11, RED),
    ("left-news",         7, ACCENT_CYAN),
    ("Government News",   4, MID_GRAY),
]
for i, (subj, pct, clr) in enumerate(subjects):
    ry = 1.62 + i * 0.55
    add_text(slide, subj, 5.2, ry, 2.3, 0.35,
             font_size=10, color=LIGHT_GRAY)
    bar_w = (pct / 40) * 2.6
    add_rect(slide, 7.6, ry+0.05, bar_w, 0.24, fill_rgb=clr)
    add_text(slide, f"{pct}%", 7.6+bar_w+0.08, ry, 0.5, 0.34,
             font_size=10, bold=True, color=clr)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — WEEK 1 PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Week 1 — ML Pipeline",
                   "End-to-end baseline pipeline implemented in Python")

steps = [
    ("1", "Load\nData",       "Fake.csv +\nTrue.csv\n44,898 rows",     ACCENT_BLUE),
    ("2", "Pre-\nprocess",    "Lowercase\nURL removal\nStemming\nStop-words", ACCENT_CYAN),
    ("3", "TF-IDF\nFeatures", "Unigrams +\nBigrams\n5,000 features",   YELLOW),
    ("4", "Train\nModels",    "Logistic\nRegression\nNaive Bayes",      GREEN),
    ("5", "Evaluate\n& Save", "Accuracy, F1\nROC-AUC\nConfusion matrix\nSave .pkl", RED),
]
for i, (num, title, desc, clr) in enumerate(steps):
    bx = 0.28 + i * 1.93
    # Box
    add_rect(slide, bx, 1.2, 1.72, 2.9, fill_rgb=RGBColor(0x10,0x28,0x40))
    add_rect(slide, bx, 1.2, 1.72, 0.06, fill_rgb=clr)
    # Number circle
    add_rect(slide, bx+0.6, 1.28, 0.52, 0.52, fill_rgb=clr)
    add_text(slide, num, bx+0.6, 1.27, 0.52, 0.52,
             font_size=16, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_text(slide, title, bx+0.08, 1.88, 1.56, 0.62,
             font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    h_rule(slide, 2.52, color=MID_GRAY, thickness=0.015)
    add_text(slide, desc, bx+0.08, 2.58, 1.56, 1.35,
             font_size=10, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

    # Arrow between steps
    if i < 4:
        add_text(slide, "▶", bx+1.68, 2.35, 0.28, 0.4,
                 font_size=14, bold=True, color=MID_GRAY, align=PP_ALIGN.CENTER)

h_rule(slide, 4.25, color=ACCENT_BLUE)
add_text(slide,
         "src/config.py  ·  src/load_data.py  ·  src/preprocess.py  ·  "
         "src/features.py  ·  src/train_model.py  ·  src/evaluate.py  ·  main.py",
         0.4, 4.38, SW-0.8, 0.5,
         font_size=10, color=MID_GRAY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — WEEK 1 RESULTS
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Week 1 — Results",
                   "Baseline model performance on 8,980 test articles (80/20 split)")

# ── Logistic Regression card ──
add_rect(slide, 0.3, 1.15, 4.3, 4.0, fill_rgb=RGBColor(0x05,0x20,0x35))
add_rect(slide, 0.3, 1.15, 4.3, 0.06, fill_rgb=GREEN)
add_text(slide, "🥇  Logistic Regression  (Best)", 0.5, 1.22, 4.0, 0.45,
         font_size=13, bold=True, color=GREEN)
lr_metrics = [
    ("Accuracy",   "99.22%", GREEN),
    ("Precision",  "99.49%", ACCENT_BLUE),
    ("Recall",     "99.02%", ACCENT_CYAN),
    ("F1 Score",   "99.25%", GREEN),
    ("ROC-AUC",    "99.96%", YELLOW),
]
for i, (lbl, val, clr) in enumerate(lr_metrics):
    my = 1.72 + i * 0.62
    add_rect(slide, 0.4, my, 4.1, 0.52, fill_rgb=RGBColor(0x0A,0x29,0x4A))
    add_text(slide, lbl, 0.55, my+0.08, 2.0, 0.38,
             font_size=12, color=LIGHT_GRAY)
    add_text(slide, val, 2.6, my+0.06, 1.7, 0.42,
             font_size=14, bold=True, color=clr, align=PP_ALIGN.RIGHT)

# ── Naive Bayes card ──
add_rect(slide, 5.0, 1.15, 4.65, 4.0, fill_rgb=RGBColor(0x10,0x1E,0x30))
add_rect(slide, 5.0, 1.15, 4.65, 0.06, fill_rgb=ACCENT_BLUE)
add_text(slide, "🔵  Naive Bayes (Baseline)", 5.2, 1.22, 4.2, 0.45,
         font_size=13, bold=True, color=ACCENT_BLUE)
nb_metrics = [
    ("Accuracy",   "95.14%", ACCENT_BLUE),
    ("Precision",  "95.47%", ACCENT_CYAN),
    ("Recall",     "95.23%", ACCENT_CYAN),
    ("F1 Score",   "95.35%", ACCENT_BLUE),
    ("ROC-AUC",    "98.78%", YELLOW),
]
for i, (lbl, val, clr) in enumerate(nb_metrics):
    my = 1.72 + i * 0.62
    add_rect(slide, 5.1, my, 4.45, 0.52, fill_rgb=RGBColor(0x0A,0x1C,0x30))
    add_text(slide, lbl, 5.25, my+0.08, 2.0, 0.38,
             font_size=12, color=LIGHT_GRAY)
    add_text(slide, val, 7.3, my+0.06, 1.7, 0.42,
             font_size=14, bold=True, color=clr, align=PP_ALIGN.RIGHT)

# VS divider
add_text(slide, "VS", 4.6, 2.5, 0.5, 0.5,
         font_size=20, bold=True, color=MID_GRAY, align=PP_ALIGN.CENTER)

# Bottom winner banner
add_rect(slide, 0.3, 5.22, SW-0.6, 0.32, fill_rgb=GREEN)
add_text(slide,
         "★  Winner: Logistic Regression   |   Saved to models/logistic_regression.pkl",
         0.4, 5.24, SW-0.8, 0.28,
         font_size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — PROJECT ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Project Architecture",
                   "Clean, modular Python package structure")

tree = [
    ("FakeNews/",           0, WHITE,        True),
    ("├── data/",           1, MID_GRAY,     False),
    ("│   ├── Fake.csv",    2, LIGHT_GRAY,   False),
    ("│   └── True.csv",    2, LIGHT_GRAY,   False),
    ("├── models/",         1, YELLOW,       False),
    ("│   └── *.pkl",       2, LIGHT_GRAY,   False),
    ("├── notebooks/",      1, MID_GRAY,     False),
    ("│   └── week1_eda.ipynb", 2, LIGHT_GRAY, False),
    ("├── src/",            1, ACCENT_BLUE,  True),
    ("│   ├── config.py",   2, LIGHT_GRAY,   False),
    ("│   ├── load_data.py",2, LIGHT_GRAY,   False),
    ("│   ├── preprocess.py",2, LIGHT_GRAY,  False),
    ("│   ├── features.py", 2, LIGHT_GRAY,   False),
    ("│   ├── train_model.py",2, LIGHT_GRAY, False),
    ("│   └── evaluate.py", 2, LIGHT_GRAY,   False),
    ("├── app/",            1, GREEN,        False),
    ("│   └── (coming soon)", 2, MID_GRAY,   False),
    ("├── main.py",         1, WHITE,        True),
    ("└── requirements.txt",1, MID_GRAY,     False),
]

add_rect(slide, 0.3, 1.1, 4.0, 4.3, fill_rgb=RGBColor(0x06,0x0F,0x1A))
for i, (line, indent, clr, bld) in enumerate(tree):
    add_text(slide, line, 0.45+indent*0.12, 1.15+i*0.22, 3.8, 0.24,
             font_size=9.5, bold=bld, color=clr,
             font_name="Consolas")

# Right: module responsibilities
modules = [
    (ACCENT_BLUE, "config.py",       "All paths, hyper-parameters, constants"),
    (GREEN,       "load_data.py",    "Read CSVs, label, shuffle, summary stats"),
    (YELLOW,      "preprocess.py",   "Clean text: URLs, stop-words, stemming"),
    (ACCENT_CYAN, "features.py",     "TF-IDF vectoriser (fit / transform)"),
    (RED,         "train_model.py",  "Train models, save / load .pkl files"),
    (MID_GRAY,    "evaluate.py",     "Metrics, confusion matrix, comparison"),
    (WHITE,       "main.py",         "Orchestrates the full 5-step pipeline"),
]
add_rect(slide, 4.55, 1.1, 5.1, 4.3, fill_rgb=RGBColor(0x10,0x28,0x40))
add_text(slide, "Module Responsibilities", 4.75, 1.15, 4.8, 0.38,
         font_size=12, bold=True, color=ACCENT_BLUE)
for i, (clr, mod, desc) in enumerate(modules):
    my = 1.58 + i * 0.54
    add_rect(slide, 4.65, my, 0.06, 0.34, fill_rgb=clr)
    add_text(slide, mod,  4.78, my,     1.55, 0.28,
             font_size=10, bold=True, color=clr)
    add_text(slide, desc, 4.78, my+0.24, 4.7, 0.28,
             font_size=9,  color=LIGHT_GRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — WEEK 2: TERMINAL APP
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Week 2 — Terminal CLI Application",
                   "Real-time fake news prediction from the command line")

# Terminal mockup
add_rect(slide, 0.3, 1.15, 5.7, 4.05, fill_rgb=RGBColor(0x0C,0x0C,0x0C))
add_rect(slide, 0.3, 1.15, 5.7, 0.38, fill_rgb=RGBColor(0x2D,0x2D,0x2D))
# Traffic-light buttons
for bx, clr in [(0.5, RED), (0.78, YELLOW), (1.06, GREEN)]:
    dot = slide.shapes.add_shape(1, Inches(bx), Inches(1.25),
                                 Inches(0.18), Inches(0.18))
    dot.fill.solid(); dot.fill.fore_color.rgb = clr
    dot.line.fill.background()
add_text(slide, "Terminal — FakeNewsDetector v1.0", 1.3, 1.19, 4.5, 0.3,
         font_size=9, color=LIGHT_GRAY, font_name="Consolas")

terminal_lines = [
    (GREEN,       "$ python cli.py"),
    (ACCENT_BLUE, ""),
    (LIGHT_GRAY,  "╔══════════════════════════════════╗"),
    (LIGHT_GRAY,  "║   🔍 Fake News Detector v1.0    ║"),
    (LIGHT_GRAY,  "╚══════════════════════════════════╝"),
    (MID_GRAY,    "Model loaded: logistic_regression.pkl"),
    (ACCENT_BLUE, ""),
    (YELLOW,      "Paste article text (or URL):"),
    (WHITE,       "> Breaking: Scientists confirm water on Mars..."),
    (ACCENT_BLUE, ""),
    (MID_GRAY,    "Analysing...  ████████████  100%"),
    (ACCENT_BLUE, ""),
    (GREEN,       "✅  PREDICTION : REAL NEWS"),
    (GREEN,       "   Confidence : 97.3%"),
    (MID_GRAY,    "   Tokens     : 147  |  Time: 0.02s"),
    (ACCENT_BLUE, ""),
    (YELLOW,      "Analyse another? [y/n]: "),
]
for i, (clr, line) in enumerate(terminal_lines):
    add_text(slide, line, 0.5, 1.62+i*0.19, 5.4, 0.22,
             font_size=8.5, color=clr, font_name="Consolas")

# Features list
add_rect(slide, 6.25, 1.15, 3.45, 4.05, fill_rgb=RGBColor(0x10,0x28,0x40))
add_rect(slide, 6.25, 1.15, 3.45, 0.06, fill_rgb=ACCENT_BLUE)
add_text(slide, "CLI Features", 6.45, 1.22, 3.1, 0.38,
         font_size=13, bold=True, color=ACCENT_BLUE)
cli_features = [
    ("Interactive prompt mode", LIGHT_GRAY),
    ("Single article prediction", LIGHT_GRAY),
    ("Batch file processing  (.csv)", LIGHT_GRAY),
    ("Confidence score display", LIGHT_GRAY),
    ("Coloured terminal output", LIGHT_GRAY),
    ("Processing time stats", LIGHT_GRAY),
    ("Auto-load best saved model", LIGHT_GRAY),
    ("URL text extraction (BeautifulSoup)", YELLOW),
]
for i, (feat, clr) in enumerate(cli_features):
    add_text(slide, f"›  {feat}", 6.45, 1.72+i*0.45, 3.1, 0.38,
             font_size=10.5, color=clr)

add_badge(slide, "app/cli.py", 6.45, 4.8, w=1.5, h=0.28,
          bg=ACCENT_BLUE, fg=NAVY, font_size=10)
add_badge(slide, "Week 2 Target", 8.1, 4.8, w=1.5, h=0.28,
          bg=YELLOW, fg=NAVY, font_size=10)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — WEEK 3: WEB APP
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Week 3 — Streamlit Web Application",
                   "Interactive browser-based UI for article analysis")

# Browser mockup frame
add_rect(slide, 0.3, 1.15, 6.0, 4.1, fill_rgb=RGBColor(0xF0,0xF4,0xF8))
add_rect(slide, 0.3, 1.15, 6.0, 0.45, fill_rgb=RGBColor(0x2D,0x2D,0x2D))
for bx, clr in [(0.48, RED), (0.73, YELLOW), (0.98, GREEN)]:
    dot = slide.shapes.add_shape(1, Inches(bx), Inches(1.23),
                                 Inches(0.16), Inches(0.16))
    dot.fill.solid(); dot.fill.fore_color.rgb = clr
    dot.line.fill.background()
# URL bar
add_rect(slide, 1.25, 1.22, 4.6, 0.3, fill_rgb=RGBColor(0xEE,0xEE,0xEE))
add_text(slide, "localhost:8501  /  Fake News Detector", 1.35, 1.24, 4.4, 0.26,
         font_size=9, color=RGBColor(0x55,0x55,0x55), font_name="Consolas")

# Page header inside browser
add_rect(slide, 0.3, 1.6, 6.0, 0.52, fill_rgb=NAVY)
add_text(slide, "🔍  Fake News Detection System", 0.48, 1.62, 5.7, 0.45,
         font_size=13, bold=True, color=WHITE)

# Sidebar
add_rect(slide, 0.3, 2.12, 1.3, 3.13, fill_rgb=RGBColor(0xE8,0xEE,0xF5))
add_text(slide, "⚙ Settings", 0.38, 2.18, 1.15, 0.28,
         font_size=9, bold=True, color=NAVY)
for i, opt in enumerate(["Model", "Threshold", "Language", "History"]):
    add_text(slide, f"• {opt}", 0.4, 2.5+i*0.38, 1.1, 0.32,
             font_size=8.5, color=RGBColor(0x44,0x44,0x66))

# Main content area
add_rect(slide, 1.65, 2.12, 4.55, 1.05, fill_rgb=WHITE)
add_text(slide, "Paste your article text here...", 1.75, 2.2, 4.3, 0.85,
         font_size=9, color=RGBColor(0xAA,0xAA,0xAA), italic=True)

# Analyse button
add_rect(slide, 1.65, 3.25, 1.5, 0.38, fill_rgb=ACCENT_BLUE)
add_text(slide, "🔍 Analyse", 1.65, 3.26, 1.5, 0.36,
         font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# Result card
add_rect(slide, 1.65, 3.75, 4.55, 1.4, fill_rgb=RGBColor(0xE8,0xF8,0xEE))
add_rect(slide, 1.65, 3.75, 4.55, 0.06, fill_rgb=GREEN)
add_text(slide, "✅  REAL NEWS", 1.8, 3.82, 2.0, 0.45,
         font_size=14, bold=True, color=GREEN)
add_text(slide, "Confidence: 97.3%  |  Tokens: 147", 1.8, 4.25, 3.5, 0.3,
         font_size=9, color=RGBColor(0x33,0x66,0x33))
add_text(slide, "████████████████████░░░░  97.3%", 1.8, 4.6, 4.1, 0.32,
         font_size=9, color=GREEN, font_name="Consolas")

# Right: features
add_rect(slide, 6.55, 1.15, 3.1, 4.1, fill_rgb=RGBColor(0x10,0x28,0x40))
add_rect(slide, 6.55, 1.15, 3.1, 0.06, fill_rgb=GREEN)
add_text(slide, "Web App Features", 6.72, 1.22, 2.8, 0.38,
         font_size=12, bold=True, color=GREEN)
web_features = [
    ("Clean Streamlit UI",           LIGHT_GRAY),
    ("Text & URL input modes",        LIGHT_GRAY),
    ("Real-time confidence bar",      LIGHT_GRAY),
    ("History / log of predictions",  LIGHT_GRAY),
    ("Model selector dropdown",       LIGHT_GRAY),
    ("EDA dashboard tab",             LIGHT_GRAY),
    ("Batch upload (.csv)",           LIGHT_GRAY),
    ("Dark / Light theme toggle",     YELLOW),
    ("Deploy: Streamlit Cloud",       ACCENT_BLUE),
]
for i, (feat, clr) in enumerate(web_features):
    add_text(slide, f"›  {feat}", 6.72, 1.68+i*0.39, 2.8, 0.34,
             font_size=10, color=clr)

add_badge(slide, "app/web_app.py", 6.72, 5.0, w=1.6, h=0.28,
          bg=GREEN, fg=NAVY, font_size=10)
add_badge(slide, "Week 3 Target", 8.4, 5.0, w=1.2, h=0.28,
          bg=YELLOW, fg=NAVY, font_size=10)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — WEEK 4: TELEGRAM BOT
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Week 4 — Telegram Bot Integration",
                   "Instant fake news checks directly in your chat")

# Phone mockup
add_rect(slide, 0.4, 1.1, 2.9, 4.4, fill_rgb=RGBColor(0x1A,0x1A,0x2E))
add_rect(slide, 0.4, 1.1, 2.9, 0.45, fill_rgb=RGBColor(0x22,0x96,0xF3))
add_text(slide, "🤖 @FakeNewsBot", 0.6, 1.14, 2.5, 0.35,
         font_size=11, bold=True, color=WHITE)

chats = [
    (True,  "Hello! Send me any article text or news URL"),
    (False, "https://somesite.com/breaking-news-article"),
    (True,  "🔍 Analysing..."),
    (True,  "━━━━━━━━━━━━━━━━━━━━\n🚨 FAKE NEWS DETECTED\nConfidence: 94.1%\nCategory: politics\n━━━━━━━━━━━━━━━━━━━━"),
    (False, "/history"),
    (True,  "📋 Last 5 predictions:\n1. FAKE  94.1%\n2. REAL  97.3%\n3. FAKE  88.6%"),
]
chat_y = 1.6
for is_bot, msg in chats:
    lines = msg.count('\n') + 1
    box_h = 0.22 * lines + 0.12
    bx    = 0.5  if is_bot  else 1.2
    bw    = 1.9  if is_bot  else 1.7
    bg    = RGBColor(0x22,0x96,0xF3) if is_bot else RGBColor(0x2A,0x2A,0x3E)
    add_rect(slide, bx, chat_y, bw, box_h, fill_rgb=bg)
    add_text(slide, msg, bx+0.08, chat_y+0.04, bw-0.16, box_h,
             font_size=7.5, color=WHITE)
    chat_y += box_h + 0.08
    if chat_y > 5.1:
        break

# Commands panel
add_rect(slide, 3.6, 1.1, 3.1, 4.4, fill_rgb=RGBColor(0x10,0x28,0x40))
add_rect(slide, 3.6, 1.1, 3.1, 0.06, fill_rgb=RGBColor(0x22,0x96,0xF3))
add_text(slide, "Bot Commands", 3.78, 1.17, 2.8, 0.38,
         font_size=12, bold=True, color=RGBColor(0x22,0x96,0xF3))
commands = [
    ("/start",       "Welcome & usage guide"),
    ("/check <text>","Analyse pasted text"),
    ("/url <link>",  "Analyse news URL"),
    ("/history",     "Show last 10 checks"),
    ("/stats",       "Bot usage stats"),
    ("/model",       "Show active model info"),
    ("/help",        "Full command list"),
]
for i, (cmd, desc) in enumerate(commands):
    my = 1.62 + i * 0.52
    add_text(slide, cmd,  3.78, my,      1.35, 0.28,
             font_size=10, bold=True, color=ACCENT_CYAN, font_name="Consolas")
    add_text(slide, desc, 3.78, my+0.25, 2.8,  0.26,
             font_size=9,  color=LIGHT_GRAY)

# Tech stack
add_rect(slide, 7.0, 1.1, 2.65, 4.4, fill_rgb=RGBColor(0x10,0x28,0x40))
add_rect(slide, 7.0, 1.1, 2.65, 0.06, fill_rgb=YELLOW)
add_text(slide, "Tech Stack", 7.18, 1.17, 2.3, 0.38,
         font_size=12, bold=True, color=YELLOW)
tech = [
    ("python-telegram-bot", "Bot framework"),
    ("BeautifulSoup4",       "URL scraping"),
    ("SQLite",               "Chat history DB"),
    ("Webhook / polling",    "Deployment"),
    ("Heroku / Railway",     "Cloud hosting"),
]
for i, (t, d) in enumerate(tech):
    my = 1.62 + i * 0.72
    add_badge(slide, t, 7.12, my, w=2.3, h=0.28,
              bg=RGBColor(0x0A,0x29,0x4A), fg=YELLOW, font_size=9)
    add_text(slide, d, 7.18, my+0.34, 2.2, 0.26,
             font_size=8.5, color=LIGHT_GRAY)

add_badge(slide, "app/telegram_bot.py", 3.78, 5.22, w=2.0, h=0.28,
          bg=RGBColor(0x22,0x96,0xF3), fg=WHITE, font_size=10)
add_badge(slide, "Week 4 Target", 6.0, 5.22, w=1.4, h=0.28,
          bg=YELLOW, fg=NAVY, font_size=10)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — WEEK 5-6: GEMINI AI
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Week 5 — Google Gemini AI Integration",
                   "Explainability & intelligent analysis powered by Gemini Pro")

add_rect(slide, 0.3, 1.15, SW-0.6, 0.95, fill_rgb=RGBColor(0x06,0x1A,0x30))
add_text(slide,
         "Our ML model classifies — Gemini explains WHY. "
         "This adds human-readable reasoning and trustworthiness to every prediction.",
         0.5, 1.25, SW-1.0, 0.72,
         font_size=13, italic=True, color=LIGHT_GRAY)

features = [
    (YELLOW,      "🧠 Explainability",
                  "Gemini generates a natural-language explanation of why an article was flagged as fake",
                  ["Why is it fake?", "Key suspicious phrases", "Source credibility analysis"]),
    (ACCENT_BLUE, "📝 Article Summary",
                  "Automatic 3-sentence summary of any news article for quick review",
                  ["TL;DR summaries", "Key claims extraction", "Named entity analysis"]),
    (GREEN,       "🔎 Cross-Reference",
                  "Gemini searches its knowledge to fact-check claims in the article",
                  ["Known fact comparison", "Date / event verification", "Contradiction detection"]),
    (RED,         "💬 Chatbot Q&A",
                  "User can ask follow-up questions about any article analysis",
                  ["Ask about sources", "Request more context", "Multi-turn dialogue"]),
]
for i, (clr, title, desc, bullets) in enumerate(features):
    bx = 0.3 + i * 2.43
    add_rect(slide, bx, 2.2, 2.28, 3.1, fill_rgb=RGBColor(0x0A,0x1E,0x32))
    add_rect(slide, bx, 2.2, 2.28, 0.06, fill_rgb=clr)
    add_text(slide, title, bx+0.1, 2.28, 2.1, 0.45,
             font_size=11, bold=True, color=clr)
    add_text(slide, desc,  bx+0.1, 2.72, 2.1, 0.72,
             font_size=9.5, color=LIGHT_GRAY)
    h_rule(slide, 3.45, color=MID_GRAY, thickness=0.015)
    for j, b in enumerate(bullets):
        add_text(slide, f"• {b}", bx+0.1, 3.52+j*0.36, 2.1, 0.34,
                 font_size=9, color=ACCENT_CYAN)

add_rect(slide, 0.3, 5.38, SW-0.6, 0.18, fill_rgb=YELLOW)
add_text(slide,
         "API: Google Generative AI SDK (google-generativeai)  ·  Model: gemini-pro  "
         "·  Free tier: 60 req/min",
         0.4, 5.38, SW-0.8, 0.18,
         font_size=9, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — WEEK 6: GOOGLE LENS / VISION API
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Week 6 — Google Lens / Vision API",
                   "Detect fake news from screenshots & images")

# Left: problem illustration
add_rect(slide, 0.3, 1.15, 3.8, 4.1, fill_rgb=RGBColor(0x10,0x28,0x40))
add_rect(slide, 0.3, 1.15, 3.8, 0.06, fill_rgb=RED)
add_text(slide, "The Problem", 0.5, 1.22, 3.5, 0.38,
         font_size=13, bold=True, color=RED)
add_text(slide,
         "Millions of fake news articles spread as screenshots on WhatsApp, "
         "Instagram, and Twitter. Text-based models can't see them.",
         0.5, 1.68, 3.5, 0.9,
         font_size=11, color=LIGHT_GRAY)

# Fake screenshot mockup
add_rect(slide, 0.55, 2.68, 3.2, 2.2, fill_rgb=WHITE)
add_rect(slide, 0.55, 2.68, 3.2, 0.35, fill_rgb=RGBColor(0xDD,0x44,0x44))
add_text(slide, "🗞  BREAKING NEWS", 0.7, 2.72, 2.8, 0.28,
         font_size=10, bold=True, color=WHITE)
add_text(slide,
         '"Scientists confirm chocolate\ncures all diseases — Big Pharma\nhides the truth!"',
         0.7, 3.1, 3.0, 0.7,
         font_size=9, color=RGBColor(0x22,0x22,0x22), italic=True)
add_text(slide, "📸  User sends this screenshot", 0.55, 4.98, 3.3, 0.28,
         font_size=9, color=MID_GRAY, align=PP_ALIGN.CENTER)

# Center arrow
add_text(slide, "→", 4.18, 3.1, 0.5, 0.5,
         font_size=22, bold=True, color=YELLOW, align=PP_ALIGN.CENTER)

# Right: solution pipeline
add_rect(slide, 4.5, 1.15, 5.15, 4.1, fill_rgb=RGBColor(0x05,0x20,0x35))
add_rect(slide, 4.5, 1.15, 5.15, 0.06, fill_rgb=GREEN)
add_text(slide, "Our Vision Pipeline", 4.68, 1.22, 4.8, 0.38,
         font_size=13, bold=True, color=GREEN)

pipeline = [
    (ACCENT_BLUE, "①  Image Input",        "User sends screenshot via bot/web"),
    (YELLOW,      "②  OCR Extraction",     "Google Vision API extracts text from image"),
    (ACCENT_CYAN, "③  Text Cleaning",      "Preprocess extracted text (same pipeline)"),
    (GREEN,       "④  ML Classification",  "TF-IDF + LR model predicts Fake / Real"),
    (RED,         "⑤  Gemini Explanation", "AI explains result in plain language"),
    (WHITE,       "⑥  Response",           "Return verdict + confidence + explanation"),
]
for i, (clr, step, desc) in enumerate(pipeline):
    py = 1.68 + i * 0.58
    add_rect(slide, 4.6, py, 4.95, 0.48, fill_rgb=RGBColor(0x0A,0x29,0x4A))
    add_rect(slide, 4.6, py, 0.06, 0.48, fill_rgb=clr)
    add_text(slide, step, 4.75, py+0.03, 2.0, 0.24,
             font_size=10, bold=True, color=clr)
    add_text(slide, desc, 4.75, py+0.24, 4.6, 0.22,
             font_size=9, color=LIGHT_GRAY)

add_badge(slide, "google-cloud-vision", 4.68, 5.2, w=2.0, h=0.28,
          bg=ACCENT_BLUE, fg=NAVY, font_size=9)
add_badge(slide, "pytesseract (fallback)", 6.85, 5.2, w=2.4, h=0.28,
          bg=MID_GRAY, fg=NAVY, font_size=9)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — FULL ROADMAP
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Project Roadmap",
                   "6-week development plan from baseline ML to AI-powered platform")

cards = [
    ("WEEK 1",  "ML Baseline",
     ["TF-IDF pipeline", "Logistic Regression", "Naive Bayes", "EDA notebook",
      "Model serialisation", "99.22% F1 score"],
     GREEN, True),
    ("WEEK 2",  "Terminal App",
     ["CLI interface", "Interactive prompts", "Batch processing", "URL scraping",
      "Coloured output", "History log"],
     ACCENT_BLUE, False),
    ("WEEK 3",  "Web App",
     ["Streamlit UI", "Text / URL input", "Confidence bars", "EDA dashboard",
      "Batch upload", "Cloud deploy"],
     ACCENT_CYAN, False),
    ("WEEK 4",  "Telegram Bot",
     ["python-telegram-bot", "Inline commands", "Chat history DB", "Broadcast alerts",
      "Admin panel", "Webhook deploy"],
     YELLOW, False),
    ("WEEK 5",  "Gemini AI",
     ["Explainability", "Article summaries", "Fact-checking", "Chatbot Q&A",
      "Multi-language", "Gemini Pro API"],
     RED, False),
    ("WEEK 6",  "Vision API +\nAdvanced ML",
     ["Google Lens OCR", "Screenshot input", "BERT fine-tuning", "Ensemble model",
      "REST API", "Full deployment"],
     ACCENT_BLUE, False),
]
for i, (week, title, bullets, clr, done) in enumerate(cards):
    roadmap_card(slide, week, title, bullets,
                 x=0.25+i*1.63, y=1.12,
                 w=1.58, h=4.4,
                 bar_color=clr, done=done)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — TECH STACK
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)
section_header_bar(slide, "Technology Stack",
                   "Libraries, frameworks and APIs across all project phases")

categories = [
    ("🤖  Machine Learning", ACCENT_BLUE, [
        ("scikit-learn",       "LR, NB, TF-IDF"),
        ("transformers",       "BERT (Week 6)"),
        ("numpy / pandas",     "Data processing"),
        ("joblib",             "Model serialisation"),
    ]),
    ("🔤  NLP & Text", YELLOW, [
        ("NLTK",               "Stopwords, stemmer"),
        ("spaCy",              "NER, tokenization"),
        ("WordCloud",          "EDA visualisation"),
        ("BeautifulSoup4",     "Web scraping"),
    ]),
    ("🌐  Web & API", GREEN, [
        ("Streamlit",          "Web dashboard"),
        ("FastAPI",            "REST endpoints"),
        ("python-telegram-bot","Telegram integration"),
        ("requests",           "HTTP client"),
    ]),
    ("☁️  AI & Cloud", RED, [
        ("Google Gemini Pro",  "Explainability AI"),
        ("Google Vision API",  "OCR from images"),
        ("Firebase / SQLite",  "User data storage"),
        ("Heroku / Railway",   "Deployment"),
    ]),
    ("📊  Visualisation", ACCENT_CYAN, [
        ("matplotlib",         "Plots & charts"),
        ("seaborn",            "Statistical plots"),
        ("Plotly",             "Interactive charts"),
        ("Jupyter",            "EDA notebooks"),
    ]),
]
for i, (cat_title, clr, items) in enumerate(categories):
    bx = 0.22 + i * 1.96
    add_rect(slide, bx, 1.1, 1.82, 4.4, fill_rgb=RGBColor(0x0A,0x1E,0x32))
    add_rect(slide, bx, 1.1, 1.82, 0.06, fill_rgb=clr)
    add_text(slide, cat_title, bx+0.08, 1.17, 1.66, 0.48,
             font_size=10, bold=True, color=clr)
    h_rule(slide, 1.65, color=MID_GRAY, thickness=0.015)
    for j, (lib, desc) in enumerate(items):
        ly = 1.72 + j * 0.88
        add_badge(slide, lib, bx+0.1, ly, w=1.62, h=0.32,
                  bg=RGBColor(0x15,0x32,0x50), fg=clr, font_size=9)
        add_text(slide, desc, bx+0.1, ly+0.36, 1.62, 0.26,
                 font_size=8, color=MID_GRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — THANK YOU
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
slide_bg(slide)

add_rect(slide, 0, 0, SW, SH, fill_rgb=NAVY)
add_rect(slide, 0, 0, 0.08, SH, fill_rgb=GREEN)
add_rect(slide, SW-0.08, 0, 0.08, SH, fill_rgb=ACCENT_BLUE)

# Top decorative grid
for r in range(4):
    for c in range(10):
        if (r + c) % 3 == 0:
            dot = slide.shapes.add_shape(
                1, Inches(0.4+c*0.95), Inches(0.15+r*0.25),
                Inches(0.06), Inches(0.06)
            )
            dot.fill.solid()
            dot.fill.fore_color.rgb = MID_GRAY if (r+c)%2==0 else DARK_BLUE
            dot.line.fill.background()

add_text(slide, "Thank You", 0.5, 0.9, SW-1.0, 1.1,
         font_size=52, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
h_rule(slide, 1.98, color=ACCENT_BLUE, thickness=0.05)
add_text(slide, "AI-Based Fake News Detection System  ·  Capstone Design Project",
         0.5, 2.1, SW-1.0, 0.5,
         font_size=16, color=ACCENT_BLUE, align=PP_ALIGN.CENTER)

# Result summary boxes
results_summary = [
    ("44,898", "Articles\nanalysed",        GREEN),
    ("99.22%", "Best model\naccuracy",       ACCENT_BLUE),
    ("6 Phases","Full project\nroadmap",     YELLOW),
    ("3 APIs",  "Gemini + Vision\n+ Telegram", RED),
]
for i, (val, lbl, clr) in enumerate(results_summary):
    bx = 0.8 + i * 2.22
    add_rect(slide, bx, 2.78, 1.92, 1.28, fill_rgb=RGBColor(0x10,0x28,0x40))
    add_rect(slide, bx, 2.78, 1.92, 0.05, fill_rgb=clr)
    add_text(slide, val, bx, 2.88, 1.92, 0.58,
             font_size=24, bold=True, color=clr, align=PP_ALIGN.CENTER)
    add_text(slide, lbl, bx, 3.45, 1.92, 0.5,
             font_size=10, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

h_rule(slide, 4.3, color=ACCENT_BLUE)
add_text(slide, "Q & A", 0.5, 4.45, SW-1.0, 0.6,
         font_size=28, bold=True, color=ACCENT_BLUE, align=PP_ALIGN.CENTER)
add_text(slide, "github.com/FakeNews  ·  Built with Python, scikit-learn, Gemini AI",
         0.5, 4.98, SW-1.0, 0.38,
         font_size=11, color=MID_GRAY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(os.path.dirname(__file__), "FakeNewsDetection_Capstone.pptx")
prs.save(out_path)
print(f"\n[OK] Presentation saved -> {out_path}")
print(f"     Slides: {len(prs.slides)}")
for i, s in enumerate(prs.slides, 1):
    title_text = ""
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            title_text = sh.text_frame.text.strip().splitlines()[0][:60]
            break
    print(f"     [{i:02d}]  {title_text}")
