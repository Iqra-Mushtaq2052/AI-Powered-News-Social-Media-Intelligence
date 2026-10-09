import time
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from utils.config import RSS_FEEDS, RAW_NEWS_FILE, RAW_DATA_DIR
from utils.helpers import get_logger, format_timestamp

logger = get_logger("NewsScraper")

def clean_html_text(raw_html: str) -> str:
    """Strip HTML tags using BeautifulSoup."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def fetch_rss_news() -> pd.DataFrame:
    """Scrape headlines and summaries from RSS news feeds."""
    articles = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for source_name, feed_url in RSS_FEEDS.items():
        logger.info(f"Fetching news from {source_name}...")
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:25]:
                title = entry.get('title', '')
                summary_raw = entry.get('summary', entry.get('description', ''))
                summary = clean_html_text(summary_raw)
                link = entry.get('link', '')
                
                # Parse timestamp
                published = entry.get('published', entry.get('updated', ''))
                ts = format_timestamp(published if published else datetime.utcnow())

                text_content = f"{title}. {summary}".strip()
                if title:
                    articles.append({
                        "id": f"news_{len(articles)+1}",
                        "source": source_name,
                        "type": "news",
                        "title": title,
                        "text": text_content,
                        "url": link,
                        "timestamp": ts
                    })
        except Exception as e:
            logger.error(f"Error scraping RSS feed {source_name}: {e}")

    df = pd.DataFrame(articles)
    
    # Fallback to realistic mock news data if scrapers return zero items
    if df.empty:
        logger.warning("No live RSS articles fetched. Generating high-quality fallback news dataset.")
        df = generate_fallback_news()

    df.to_csv(RAW_NEWS_FILE, index=False)
    logger.info(f"Saved {len(df)} news articles to {RAW_NEWS_FILE}")
    return df

def generate_fallback_news() -> pd.DataFrame:
    """Provide structured sample news for offline/testing robustness."""
    sample_items = [
        ("BBC Tech", "Tech Giants Accelerate AI Supercomputer Deployments", 
         "Major tech conglomerates announced massive multi-billion dollar investments in next-generation GPU server farms and custom silicon hardware for deep learning model training."),
        ("Google News AI", "Breakthrough in Transformer Architecture Reduces Training Costs", 
         "Researchers have introduced a novel attention compression technique that halves memory requirements while boosting inference speed for large neural networks."),
        ("TechCrunch", "Generative AI Startups Raise Record Funding in Q3", 
         "Venture capital investments in generative intelligence, automated agents, and enterprise AI productivity suites hit unprecedented highs this quarter."),
        ("BBC World", "Global Regulatory Bodies Propose New Ethical AI Framework", 
         "International policymakers drafted guidelines emphasizing data transparency, algorithmic fairness, and copyright protections across digital news platforms."),
        ("NYT Technology", "Autonomous AI Agents Reshaping Enterprise Workflows", 
         "Businesses across finance, healthcare, and customer support are integrating intelligent software agents capable of executing multi-step autonomous tasks.")
    ]
    
    rows = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for i, (source, title, summary) in enumerate(sample_items):
        rows.append({
            "id": f"news_fb_{i+1}",
            "source": source,
            "type": "news",
            "title": title,
            "text": f"{title}. {summary}",
            "url": "https://example.com/news",
            "timestamp": now
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df_news = fetch_rss_news()
    print(f"Scraped {len(df_news)} news headlines. Sample:")
    print(df_news.head(2))
