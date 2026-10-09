import logging
import sqlite3
import pandas as pd
from datetime import datetime
from utils.config import INTELLIGENCE_DB_FILE

def get_logger(name: str) -> logging.Logger:
    """Set up and return a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def save_to_sqlite(df: pd.DataFrame, table_name: str = "intelligence_data"):
    """Save dataframe to SQLite database for easy query access by Chatbot."""
    try:
        conn = sqlite3.connect(INTELLIGENCE_DB_FILE)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        logging.info(f"Successfully saved {len(df)} rows to SQLite table '{table_name}'.")
    except Exception as e:
        logging.error(f"Error saving to SQLite: {e}")

def load_from_sqlite(table_name: str = "intelligence_data") -> pd.DataFrame:
    """Load data from SQLite database."""
    try:
        conn = sqlite3.connect(INTELLIGENCE_DB_FILE)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Error loading from SQLite: {e}")
        return pd.DataFrame()

def format_timestamp(ts=None) -> str:
    """Return formatted timestamp string."""
    if ts is None:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)
