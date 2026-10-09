import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from utils.config import CLEANED_DATA_FILE, SAVED_MODELS_DIR
from utils.helpers import get_logger

logger = get_logger("MLBaseline")

def train_ml_baseline(df: pd.DataFrame = None):
    """Train Naive Bayes and Logistic Regression baseline classifiers using TF-IDF."""
    if df is None or df.empty:
        if os.path.exists(CLEANED_DATA_FILE):
            df = pd.read_csv(CLEANED_DATA_FILE)
        else:
            logger.error("No cleaned data available to train ML baseline.")
            return {}

    # Ensure label column exists
    if 'label' not in df.columns or 'cleaned_text' not in df.columns:
        logger.error("Required columns ('cleaned_text', 'label') missing from DataFrame.")
        return {}

    X = df['cleaned_text'].fillna("")
    y = df['label']

    if len(df) < 5:
        # Duplicate data for minimal split demonstration if tiny dataset
        X = pd.concat([X]*5, ignore_index=True)
        y = pd.concat([y]*5, ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 1. Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_preds = nb_model.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_preds)

    # 2. Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_preds)

    best_model = lr_model if lr_acc >= nb_acc else nb_model
    best_name = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"
    best_acc = max(lr_acc, nb_acc)

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, lr_preds if lr_acc >= nb_acc else nb_preds, average='macro', zero_division=0
    )

    metrics = {
        "model_type": f"ML Baseline ({best_name})",
        "accuracy": round(best_acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4)
    }

    # Save model and vectorizer artifact
    model_save_path = SAVED_MODELS_DIR / "ml_baseline.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump({"model": best_model, "vectorizer": vectorizer, "metrics": metrics}, f)

    logger.info(f"ML Baseline ({best_name}) Trained. Accuracy: {best_acc:.4f}, F1: {f1:.4f}")
    return metrics

def predict_ml_baseline(text: str):
    """Predict sentiment label for input text using trained ML baseline."""
    model_save_path = SAVED_MODELS_DIR / "ml_baseline.pkl"
    if not os.path.exists(model_save_path):
        train_ml_baseline()

    try:
        with open(model_save_path, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        vectorizer = data["vectorizer"]
        
        vec = vectorizer.transform([text])
        pred_label_idx = model.predict(vec)[0]
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        return label_map.get(pred_label_idx, "neutral")
    except Exception as e:
        logger.error(f"Prediction error in ML Baseline: {e}")
        return "neutral"

if __name__ == "__main__":
    metrics = train_ml_baseline()
    print("Baseline Metrics:", metrics)
    print("Test Prediction:", predict_ml_baseline("AI technology is making huge progress today!"))
