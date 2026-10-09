import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Directories
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Ensure required directories exist
for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, SAVED_MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Data File Paths
RAW_NEWS_FILE = RAW_DATA_DIR / "news_raw.csv"
RAW_REDDIT_FILE = RAW_DATA_DIR / "reddit_raw.csv"
COMBINED_RAW_FILE = RAW_DATA_DIR / "combined_raw.csv"

CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
INTELLIGENCE_DB_FILE = PROCESSED_DATA_DIR / "intelligence.db"

# RSS News Feeds (Reliable, free sources)
RSS_FEEDS = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Tech": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "Google News Tech": "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en",
    "Google News AI": "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-US&gl=US&ceid=US:en",
    "TechCrunch": "https://techcrunch.com/feed/",
    "NYT Technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"
}

# Reddit Configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "NewsIntelligenceBot/1.0")

TARGET_SUBREDDITS = ["technology", "artificial", "news", "worldnews", "business"]

# Model Configs
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 5
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
