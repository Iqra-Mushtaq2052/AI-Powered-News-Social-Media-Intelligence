import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.config import (
    CLEANED_DATA_FILE, SAVED_MODELS_DIR, MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, LSTM_UNITS, BATCH_SIZE, EPOCHS
)
from utils.helpers import get_logger

logger = get_logger("LSTMModel")

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM Network for Sequence Text Sentiment Classification."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        # Max pooling across sequence dimension
        pooled, _ = torch.max(lstm_out, dim=1)
        out = self.fc(self.dropout(pooled))
        return out

class TextTokenizer:
    """Simple word index vocabulary builder and sequence padder."""
    def __init__(self, max_words=MAX_VOCAB_SIZE, max_len=MAX_SEQUENCE_LENGTH):
        self.max_words = max_words
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit_on_texts(self, texts):
        words = []
        for text in texts:
            if isinstance(text, str):
                words.extend(text.split())
        counts = Counter(words).most_common(self.max_words - 2)
        for idx, (word, _) in enumerate(counts, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            if isinstance(text, str):
                for word in text.split():
                    seq.append(self.word2idx.get(word, 1))
            # Pad or truncate
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[:self.max_len]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.long)

def train_lstm_model(df: pd.DataFrame = None):
    """Train BiLSTM Deep Learning Model for Sentiment Analysis."""
    if df is None or df.empty:
        if os.path.exists(CLEANED_DATA_FILE):
            df = pd.read_csv(CLEANED_DATA_FILE)
        else:
            logger.error("No cleaned data file found for LSTM training.")
            return {}

    texts = df['cleaned_text'].fillna("").values
    labels = df['label'].values

    if len(texts) < 10:
        texts = np.tile(texts, 5)
        labels = np.tile(labels, 5)

    tokenizer = TextTokenizer()
    tokenizer.fit_on_texts(texts)
    X_seq = tokenizer.texts_to_sequences(texts)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_tensor, test_size=0.2, random_state=42
    )

    vocab_size = len(tokenizer.word2idx)
    model = BiLSTMClassifier(vocab_size, EMBEDDING_DIM, LSTM_UNITS, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, preds = torch.max(test_outputs, 1)

    acc = accuracy_score(y_test.numpy(), preds.numpy())
    p, r, f1, _ = precision_recall_fscore_support(
        y_test.numpy(), preds.numpy(), average='macro', zero_division=0
    )

    metrics = {
        "model_type": "Deep Learning (BiLSTM)",
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4)
    }

    # Save model weights and tokenizer
    save_path = SAVED_MODELS_DIR / "lstm_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "metrics": metrics
    }, save_path)

    logger.info(f"BiLSTM Model Trained. Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return metrics

def predict_lstm_sentiment(text: str) -> str:
    """Predict sentiment using saved PyTorch BiLSTM model."""
    save_path = SAVED_MODELS_DIR / "lstm_model.pt"
    if not os.path.exists(save_path):
        train_lstm_model()

    try:
        checkpoint = torch.load(save_path)
        tokenizer = checkpoint["tokenizer"]
        vocab_size = checkpoint["vocab_size"]

        model = BiLSTMClassifier(vocab_size, EMBEDDING_DIM, LSTM_UNITS, num_classes=3)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        seq = tokenizer.texts_to_sequences([text])
        with torch.no_grad():
            out = model(seq)
            _, pred = torch.max(out, 1)

        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        return label_map.get(int(pred[0]), "neutral")
    except Exception as e:
        logger.error(f"Error in LSTM prediction: {e}")
        return "neutral"

if __name__ == "__main__":
    metrics = train_lstm_model()
    print("LSTM Metrics:", metrics)
    print("Test Prediction:", predict_lstm_sentiment("Massive new AI hardware breakthrough announced today."))
