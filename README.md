# AI-Based Fake News Detection System
> Capstone Design Project — Machine Learning + NLP Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Week%201-Complete-brightgreen)
![Status](https://img.shields.io/badge/Week%202-Complete-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-99.22%25-brightgreen)

---

## Results (Week 1 Baseline)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **99.22%** | **99.49%** | **99.02%** | **99.25%** | **99.96%** |
| Naive Bayes | 95.14% | 95.47% | 95.23% | 95.35% | 98.78% |

---

## Project Roadmap

| Phase | Description | Status |
|---|---|---|
| **Week 1** | ML baseline pipeline (TF-IDF + LR + NB) | ✅ Done |
| **Week 2** | Terminal CLI application | ✅ Done |
| **Week 3** | Streamlit web application | ✅ Done |
| **Week 4** | Telegram bot integration | 📅 Planned |
| **Week 5** | Google Gemini AI (explainability) | 📅 Planned |
| **Week 6** | Google Vision API (OCR from screenshots) | 📅 Planned |

---

## Project Structure

```
FakeNews/
├── data/                    # CSV files (not in git — see below)
│   ├── Fake.csv
│   └── True.csv
├── models/                  # Saved model artefacts (.pkl)
├── notebooks/
│   └── week1_eda.ipynb      # Exploratory Data Analysis
├── src/
│   ├── config.py            # Central paths & hyperparameters
│   ├── load_data.py         # Data loading & summary
│   ├── preprocess.py        # Text cleaning pipeline
│   ├── features.py          # TF-IDF feature extraction
│   ├── train_model.py       # Training, saving, loading models
│   └── evaluate.py          # Metrics, confusion matrix, comparison
├── app/
│   └── cli.py               # Week 2 — Terminal CLI application
├── main.py                  # Run the full pipeline
├── create_presentation.py   # Generate .pptx presentation
└── requirements.txt
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/FakeNews.git
cd FakeNews
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `Fake.csv` and `True.csv` inside the `data/` folder.

### 5. Run the pipeline
```bash
python main.py
```

### 6. Launch the Web App (Week 3)
```bash
streamlit run app/web_app.py
# opens at http://localhost:8501
```

### 7. Launch the Terminal CLI (Week 2)
```bash
# Interactive mode (default)
python -m app.cli

# Classify a single article text
python -m app.cli --text "Breaking: Scientists confirm water on Mars..."

# Fetch and classify a news URL
python -m app.cli --url https://reuters.com/article/some-news

# Batch-classify all rows in a CSV
python -m app.cli --file data/articles.csv --col text

# Use a specific model by name
python -m app.cli --model naive_bayes
```

### 7. Open the EDA notebook
```bash
jupyter notebook notebooks/week1_eda.ipynb
```

### 8. Regenerate the presentation
```bash
python create_presentation.py
```

---

## Pipeline Steps

```
Raw Data → Preprocess → TF-IDF Features → Train Models → Evaluate → Save Best Model
```

1. **Load Data** — merge `Fake.csv` + `True.csv`, label, shuffle (44,898 articles)
2. **Preprocess** — lowercase, remove URLs, non-alpha chars, stop-words, Porter stemming
3. **TF-IDF** — unigrams + bigrams, 5,000 features, sublinear TF scaling
4. **Train** — Logistic Regression & Naive Bayes on 80/20 stratified split
5. **Evaluate** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix PNG
6. **Save** — best model persisted to `models/` with joblib

---

## Tech Stack

- **ML**: scikit-learn, joblib
- **NLP**: NLTK (stopwords, stemmer), TF-IDF
- **Data**: pandas, numpy
- **Visualisation**: matplotlib, seaborn, wordcloud
- **Notebook**: Jupyter
- **Presentation**: python-pptx
- **CLI (Week 2)**: colorama, requests, BeautifulSoup4
- **Coming**: Streamlit, python-telegram-bot, Google Gemini API, Google Vision API

---

## Dataset

- **Source**: [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Credit**: Ahmed, Hadeer, et al. (2017)
- **Size**: 44,898 articles | Fake: 23,481 | Real: 21,417
- **Features**: title, text, subject, date, label
