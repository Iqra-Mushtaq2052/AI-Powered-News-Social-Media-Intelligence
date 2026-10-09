import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminal stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


from utils.helpers import get_logger
from scraper.news_scraper import fetch_rss_news
from scraper.reddit_scraper import fetch_reddit_data
from nlp.preprocess import run_preprocessing_pipeline
from models.ml_baseline import train_ml_baseline
from models.lstm_model import train_lstm_model
from models.transformer_model import evaluate_transformer_model
from chatbot.chatbot_engine import IntelligenceChatbot

logger = get_logger("MainPipeline")

def run_full_pipeline():
    """Execute end-to-end intelligence chatbot pipeline."""
    logger.info("=== STEP 1: Web Scraping (RSS News + Reddit) ===")
    df_news = fetch_rss_news()
    df_reddit = fetch_reddit_data()
    
    logger.info("=== STEP 2: NLP Preprocessing & Data Cleaning ===")
    df_processed = run_preprocessing_pipeline()
    
    logger.info("=== STEP 3: Machine Learning Baseline Training ===")
    ml_metrics = train_ml_baseline(df_processed)
    
    logger.info("=== STEP 4: Deep Learning (BiLSTM) Training ===")
    lstm_metrics = train_lstm_model(df_processed)
    
    logger.info("=== STEP 5: Transformer (DistilBERT) Evaluation ===")
    transformer_metrics = evaluate_transformer_model(df_processed)
    
    logger.info("=== STEP 6: Testing Chatbot Engine ===")
    chatbot = IntelligenceChatbot()
    test_response = chatbot.process_query("What is the sentiment this week?")
    print("\n--- CHATBOT TEST RESPONSE ---")
    print(test_response)
    print("-----------------------------\n")
    
    logger.info("✅ Full Pipeline Completed Successfully!")
    return df_processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI News/Social Media Intelligence Chatbot Pipeline")
    parser.add_argument("--scrape-only", action="store_true", help="Run web scraping step only")
    parser.add_argument("--train-only", action="store_true", help="Run model training step only")
    parser.add_argument("--test", action="store_true", help="Run quick test suite")
    
    args = parser.parse_args()

    if args.scrape_only:
        fetch_rss_news()
        fetch_reddit_data()
    elif args.train_only:
        df_p = run_preprocessing_pipeline()
        train_ml_baseline(df_p)
        train_lstm_model(df_p)
        evaluate_transformer_model(df_p)
    else:
        run_full_pipeline()
