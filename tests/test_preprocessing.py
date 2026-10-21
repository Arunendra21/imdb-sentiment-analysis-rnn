"""Unit tests for the text preprocessing pipeline (no model file needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import MAX_LEN, decode_review, encode_review, preprocess_text

WORD_INDEX = {"this": 10, "movie": 25, "was": 7, "fantastic": 900}


def test_encode_known_words_applies_offset():
    assert encode_review("this movie", WORD_INDEX) == [13, 28]


def test_encode_is_case_insensitive():
    assert encode_review("THIS Movie", WORD_INDEX) == [13, 28]


def test_encode_unknown_word_matches_original_behaviour():
    # Original project code: word_index.get(word, 2) + 3  ->  5 for unknown words.
    assert encode_review("unseenword", WORD_INDEX) == [5]


def test_preprocess_pads_to_max_len():
    result = preprocess_text("this movie was fantastic", WORD_INDEX)
    assert result.shape == (1, MAX_LEN)
    assert list(result[0, -4:]) == [13, 28, 10, 903]


def test_decode_roundtrip():
    reverse = {v: k for k, v in WORD_INDEX.items()}
    encoded = encode_review("this movie", WORD_INDEX)
    assert decode_review(encoded, reverse) == "this movie"


def test_decode_unknown_index_is_question_mark():
    assert decode_review([123456], {}) == "?"
