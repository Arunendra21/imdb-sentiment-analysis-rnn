"""Model and vocabulary loading for the IMDB sentiment classifier."""
from __future__ import annotations

from pathlib import Path

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

from .preprocessing import preprocess_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "simple_rnn_imdb.keras"


def load_sentiment_model(model_path: Path = MODEL_PATH):
    """Load the trained SimpleRNN model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Train the model first with `python train.py`."
        )
    return load_model(model_path)


def load_word_index() -> dict:
    """Download (once) and return the Keras IMDB word-to-index mapping."""
    return imdb.get_word_index()


def predict_sentiment(text: str, model, word_index: dict) -> tuple[str, float]:
    """Return (sentiment label, positive-class probability) for a review."""
    inputs = preprocess_text(text, word_index)
    score = float(model.predict(inputs, verbose=0)[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score
