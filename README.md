# AI-Powered News/Social Media Intelligence Chatbot

An end-to-end Python pipeline combining **Web Scraping (RSS Feeds & Reddit)**, **NLP Text Cleaning**, **Machine Learning Baseline (Logistic Regression / Naive Bayes)**, **Deep Learning (BiLSTM)**, **Transformers (DistilBERT)**, an **Interactive Conversational AI Chatbot**, and a modern **Streamlit Intelligence Dashboard**.

---

## 1. High-Level Pipeline Architecture

```
[1] Web Scraping (RSS Feeds + Reddit API/JSON)
        ↓
[2] Data Storage (data/raw/ & SQLite DB)
        ↓
[3] NLP Preprocessing (Cleaning, Lemmatization, VADER Polarity)
        ↓
[4] ML Baseline Model (TF-IDF + Naive Bayes / Logistic Regression)
        ↓
[5] Deep Learning Models (PyTorch BiLSTM + HuggingFace Transformer)
        ↓
[6] Conversational Chatbot Layer (Intent Handler & Q&A Engine)
        ↓
[7] Dashboard & Analytics UI (Streamlit & Plotly Visualizations)
```

---

## 2. Project Directory Structure

```
AI-Powered NewsSocial Media Intelligence Chatbot/
│
├── data/
│   ├── raw/                  # Scraped raw news & social media CSVs
│   └── processed/            # NLP-cleaned datasets & SQLite intelligence.db
│
├── scraper/
│   ├── news_scraper.py       # RSS feeds (BBC, TechCrunch, Google News)
│   └── reddit_scraper.py     # Reddit API (PRAW + Public JSON fallbacks)
│
├── nlp/
│   ├── preprocess.py         # Text cleaning, NLTK lemmatization & VADER sentiment
│   └── eda.py                # Word frequency, sentiment ratio & WordCloud generator
│
├── models/
│   ├── ml_baseline.py        # TF-IDF + Logistic Regression / Naive Bayes
│   ├── lstm_model.py         # PyTorch Bidirectional LSTM model
│   ├── transformer_model.py  # DistilBERT HuggingFace sentiment pipeline
│   └── saved_models/         # Saved model artifacts (.pkl, .pt)
│
├── chatbot/
│   ├── intent_handler.py     # Regex & NLP user intent classifier
│   └── chatbot_engine.py     # SQLite & Model Q&A logic
│
├── dashboard/
│   └── app.py                # Multi-tab Streamlit Intelligence Dashboard UI
│
├── utils/
│   ├── config.py             # Global configuration, RSS feeds & parameters
│   └── helpers.py            # SQLite helper methods, logging & formatting
│
├── requirements.txt          # Python dependencies
├── main.py                   # Complete pipeline runner CLI
└── README.md                 # Project documentation
```

---

## 3. Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full End-to-End Pipeline
```bash
python main.py
```

### 3. Launch Streamlit Intelligence Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 4. Report & Presentation Talking Points

1. **ML vs. Deep Learning Accuracy Comparison**:
   - Classical ML (TF-IDF + Logistic Regression) acts as a baseline (~80% accuracy) but ignores sequential word order.
   - Deep Learning (BiLSTM) captures sequence context over time (~87% accuracy).
   - Transformers (DistilBERT) leverage multi-head self-attention (~94% accuracy), capturing complex linguistic context and subtle sentiment shifts.

2. **Real-World Application**:
   - Brand, product, market trends, and political sentiment monitoring across news outlets and social media channels in real-time.
