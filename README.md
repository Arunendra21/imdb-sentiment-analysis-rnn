# IMDB Movie Review Sentiment Analysis with SimpleRNN

A deep learning project that classifies IMDB movie reviews as **positive** or
**negative** using a Simple Recurrent Neural Network (SimpleRNN) built with
Keras/TensorFlow. It includes the training pipeline, the trained model, and a
Streamlit web app for interactive predictions.

## Problem Statement

Sentiment analysis of movie reviews is a classic binary text-classification
problem. Given the raw text of a review, the goal is to predict whether the
reviewer’s overall sentiment is positive or negative. This project solves it
with a sequence model that reads the review word by word, using the Keras
IMDB dataset (25,000 training and 25,000 test reviews).

## Model Architecture

| Layer | Details |
|---|---|
| Embedding | Vocabulary 10,000 → 128-dimensional vectors, input length 500 |
| SimpleRNN | 128 units, ReLU activation |
| Dense | 1 unit, sigmoid activation (positive-class probability) |

- **Total parameters:** 1,313,025
- **Loss / optimiser:** Binary cross-entropy / Adam
- **Training:** Up to 10 epochs, batch size 32, 20% validation split,
  early stopping on validation loss (patience 5, restore best weights)
- **Measured test accuracy:** 80.3% (evaluated with the
  shipped model on the IMDB test set — see *Evaluation* below)

## How It Works

1. **Encoding** — review text is lowercased, split into words and mapped to
   integer indices using the Keras IMDB word index (+3 offset; indices 0–2
   are reserved).
2. **Padding** — every sequence is padded/truncated to 500 tokens.
3. **Prediction** — the SimpleRNN reads the sequence and outputs a single
   probability; scores above 0.5 are classified as *Positive*.

The preprocessing used at inference (`src/preprocessing.py`) is exactly the
procedure the model was trained with.

## Project Structure

```
.
├── app.py                    # Streamlit entry point
├── train.py                  # Training script (script version of the notebook)
├── src/
│   ├── model.py              # Model + vocabulary loading, prediction helper
│   └── preprocessing.py      # Text encoding / padding / decoding
├── models/
│   └── simple_rnn_imdb.keras # Trained model (native Keras format)
├── notebooks/
│   ├── training.ipynb        # Original training notebook
│   └── prediction.ipynb      # Original inference notebook
├── tests/
│   └── test_preprocessing.py # Unit tests for preprocessing
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Arunendra21/imdb-sentiment-analysis-rnn.git
cd imdb-sentiment-analysis-rnn
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.11 or 3.12 (TensorFlow 2.16). The first run downloads the
IMDB vocabulary (~1.5 MB) from Keras; the trained model is included in
`models/`, so no training is needed to use the app.

## Usage

### Streamlit app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, paste a movie review, and click
**Classify**.

### Retrain the model

```bash
python train.py            # trains and writes models/simple_rnn_imdb.keras
python train.py --epochs 5 # optional overrides
```

### Run the tests

```bash
pip install pytest
pytest
```

## Evaluation

The original training notebook did not record a final test-set score, so the
shipped model was evaluated directly for this repository: the figure above
is the accuracy of `models/simple_rnn_imdb.h5` on the full 25,000-review
IMDB test set, measured with the dependency versions in `requirements.txt`.
SimpleRNNs with ReLU activation are known to be less stable than LSTM/GRU
variants; the training log in the notebook shows the resulting fluctuation
in validation accuracy across epochs.

## Deployment

The app is ready for [Streamlit Community Cloud](https://share.streamlit.io):

1. Fork or push this repository to GitHub.
2. In Streamlit Cloud, create a new app pointing at this repo with
   **Main file path: `app.py`** and Python 3.12.
3. No secrets or API keys are required.

Live app: STREAMLIT_URL_PLACEHOLDER

## Notes and Limitations

- The original project saved the model in the legacy `.h5` format with
  TensorFlow 2.15. That file no longer loads directly under Keras 3
  (TensorFlow 2.16 rejects a legacy SimpleRNN config argument), so the
  weights were transferred losslessly into the identical architecture in
  the native `.keras` format. Equivalence was verified against an
  independent NumPy forward pass of the original H5 weights
  (max prediction difference ~1.5e-06, i.e. float32 rounding), and the
  file also shrinks from 16 MB to 5.3 MB because the H5 copy carried
  Adam optimiser state that inference does not need.
- Predictions use the fixed 10,000-word IMDB vocabulary; words outside it
  are handled by the original project’s fallback encoding.
- `notebooks/` preserves the original project notebooks (outputs cleared);
  `embedding.ipynb` from the original upload was a standalone embedding
  tutorial unrelated to this model and is not included.
