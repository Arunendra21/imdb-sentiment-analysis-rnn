"""Streamlit app: IMDB Movie Review Sentiment Analysis.

Run locally with:  streamlit run app.py
"""
import streamlit as st

from src.model import load_sentiment_model, load_word_index, predict_sentiment

st.set_page_config(page_title="IMDB Movie Review Sentiment Analysis", page_icon="🎬")


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_sentiment_model()


@st.cache_resource(show_spinner="Loading vocabulary…")
def get_word_index():
    return load_word_index()


st.title("IMDB Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review and the trained SimpleRNN model will classify "
    "it as positive or negative."
)

try:
    model = get_model()
    word_index = get_word_index()
except Exception as exc:  # model file missing or vocabulary download failed
    st.error(f"Could not load the model assets: {exc}")
    st.stop()

with st.form("review_form"):
    user_input = st.text_area("Movie Review", height=160,
                              placeholder="This movie was fantastic! The acting was great…")
    submitted = st.form_submit_button("Classify")

if submitted:
    if not user_input.strip():
        st.warning("Please enter a movie review first.")
    else:
        with st.spinner("Analysing…"):
            sentiment, score = predict_sentiment(user_input, model, word_index)
        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")
        st.write(f"Prediction score: {score:.4f} "
                 f"(probability that the review is positive)")
        st.progress(score)
