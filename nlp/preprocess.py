import re
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.config import (
    RAW_NEWS_FILE, RAW_REDDIT_FILE, COMBINED_RAW_FILE,
    CLEANED_DATA_FILE
)
from utils.helpers import get_logger, save_to_sqlite

logger = get_logger("NLPPreprocess")

# Ensure required NLTK datasets
for resource in ['stopwords', 'wordnet', 'punkt']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK resource '{resource}': {e}")

try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by"}

LEMMATIZER = WordNetLemmatizer()
VADER_ANALYZER = SentimentIntensityAnalyzer()

def clean_text(text: str) -> str:
    """Perform text normalization: lowercase, URL removal, punctuation removal, lemmatization."""
    if not isinstance(text, str) or not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization & stopword removal + lemmatization
    tokens = text.split()
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token) for token in tokens 
        if token not in STOP_WORDS and len(token) > 2
    ]
    
    return " ".join(cleaned_tokens)

def compute_sentiment_vader(text: str):
    """Calculate sentiment compound score and label (Positive, Negative, Neutral)."""
    if not text:
        return 0.0, "neutral", 1
    
    scores = VADER_ANALYZER.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        label = "positive"
        sentiment_idx = 2
    elif compound <= -0.05:
        label = "negative"
        sentiment_idx = 0
    else:
        label = "neutral"
        sentiment_idx = 1
        
    return compound, label, sentiment_idx

def run_preprocessing_pipeline() -> pd.DataFrame:
    """Read raw scraped data, apply NLP cleaning and sentiment labeling, save outputs."""
    raw_dfs = []
    
    if os.path.exists(RAW_NEWS_FILE):
        raw_dfs.append(pd.read_csv(RAW_NEWS_FILE))
    if os.path.exists(RAW_REDDIT_FILE):
        raw_dfs.append(pd.read_csv(RAW_REDDIT_FILE))
        
    if not raw_dfs:
        logger.warning("No raw data files found. Running scrapers first...")
        from scraper.news_scraper import fetch_rss_news
        from scraper.reddit_scraper import fetch_reddit_data
        raw_dfs.append(fetch_rss_news())
        raw_dfs.append(fetch_reddit_data())
        
    df_combined = pd.concat(raw_dfs, ignore_index=True)
    df_combined.to_csv(COMBINED_RAW_FILE, index=False)
    logger.info(f"Combined {len(df_combined)} raw entries.")
    
    # NLP Cleaning
    logger.info("Cleaning raw text data...")
    df_combined['cleaned_text'] = df_combined['text'].apply(clean_text)
    
    # Sentiment Calculation
    logger.info("Calculating baseline VADER sentiment scores...")
    sent_results = df_combined['text'].apply(compute_sentiment_vader)
    df_combined['sentiment_score'] = [r[0] for r in sent_results]
    df_combined['sentiment'] = [r[1] for r in sent_results]
    df_combined['label'] = [r[2] for r in sent_results]
    
    # Remove empty cleaned texts
    df_processed = df_combined[df_combined['cleaned_text'].str.strip() != ""].copy()
    
    # Save processed CSV & SQLite DB
    df_processed.to_csv(CLEANED_DATA_FILE, index=False)
    save_to_sqlite(df_processed, "intelligence_data")
    
    logger.info(f"NLP Preprocessing complete. Saved {len(df_processed)} processed rows.")
    return df_processed

if __name__ == "__main__":
    df_proc = run_preprocessing_pipeline()
    print("Processed Data Sample:")
    print(df_proc[['title', 'sentiment', 'sentiment_score', 'cleaned_text']].head(3))
