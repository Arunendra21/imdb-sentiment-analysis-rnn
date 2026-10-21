"""Train the SimpleRNN sentiment model on the Keras IMDB dataset.

Script version of ``notebooks/training.ipynb`` — same architecture,
hyperparameters and training procedure as the original project:

    Embedding(10000, 128) -> SimpleRNN(128, relu) -> Dense(1, sigmoid)
    Adam, binary cross-entropy, up to 10 epochs, batch size 32,
    20% validation split, early stopping on val_loss (patience 5).

Usage:  python train.py [--epochs 10] [--output models/simple_rnn_imdb.keras]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

MAX_FEATURES = 10000
MAX_LEN = 500
EMBEDDING_DIM = 128
RNN_UNITS = 128


def build_model() -> tf.keras.Model:
    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        SimpleRNN(RNN_UNITS, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "models" / "simple_rnn_imdb.keras")
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    model = build_model()
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=5,
                                   restore_best_weights=True)
    model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
