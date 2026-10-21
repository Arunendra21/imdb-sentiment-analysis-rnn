"""Text preprocessing for the IMDB sentiment model.

Encoding follows the Keras IMDB convention used during training:
indices 0, 1 and 2 are reserved (padding, start-of-sequence, unknown),
and every known word index is shifted by +3. Sequences are padded or
truncated to ``MAX_LEN`` tokens.
"""
from __future__ import annotations

import numpy as np
from tensorflow.keras.preprocessing import sequence

MAX_LEN = 500
INDEX_OFFSET = 3
UNKNOWN_INDEX = 2


def encode_review(text: str, word_index: dict) -> list[int]:
    """Encode raw review text into a list of IMDB word indices.

    Matches the original project code exactly: words missing from the
    vocabulary fall back to index 2 before the +3 offset is applied.
    """
    words = text.lower().split()
    return [word_index.get(word, UNKNOWN_INDEX) + INDEX_OFFSET for word in words]


def preprocess_text(text: str, word_index: dict) -> np.ndarray:
    """Encode and pad raw review text into a model-ready array of shape (1, MAX_LEN)."""
    encoded = encode_review(text, word_index)
    return sequence.pad_sequences([encoded], maxlen=MAX_LEN)


def decode_review(encoded_review, reverse_word_index: dict) -> str:
    """Decode a sequence of IMDB word indices back into readable text."""
    return " ".join(
        reverse_word_index.get(i - INDEX_OFFSET, "?") for i in encoded_review
    )
