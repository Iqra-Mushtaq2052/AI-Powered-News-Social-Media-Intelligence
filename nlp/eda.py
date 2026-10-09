import pandas as pd
from collections import Counter
from typing import Dict, Any, List

def get_word_frequencies(df: pd.DataFrame, top_n: int = 20) -> List[tuple]:
    """Extract top N frequent words from cleaned text."""
    if df.empty or 'cleaned_text' not in df.columns:
        return []
    
    all_words = " ".join(df['cleaned_text'].dropna()).split()
    return Counter(all_words).most_common(top_n)

def get_sentiment_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Get count of positive, negative, and neutral items."""
    if df.empty or 'sentiment' not in df.columns:
        return {"positive": 0, "neutral": 0, "negative": 0}
    
    counts = df['sentiment'].value_counts().to_dict()
    return {
        "positive": counts.get("positive", 0),
        "neutral": counts.get("neutral", 0),
        "negative": counts.get("negative", 0)
    }

def get_source_breakdown(df: pd.DataFrame) -> Dict[str, int]:
    """Get count of posts/articles per source."""
    if df.empty or 'source' not in df.columns:
        return {}
    return df['source'].value_counts().to_dict()

def generate_wordcloud(df: pd.DataFrame):
    """Generate WordCloud object for Streamlit or Matplotlib plotting."""
    try:
        from wordcloud import WordCloud
        text = " ".join(df['cleaned_text'].dropna())
        if not text.strip():
            text = "AI News Intelligence Social Media Analytics Sentiment"
        wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(text)
        return wc
    except Exception as e:
        print(f"Error generating WordCloud: {e}")
        return None
