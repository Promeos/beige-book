"""
Tests for src/sentiment.py — VADER scoring and DataFrame integration.

FinBERT tests are skipped to avoid downloading the model in CI.
"""

import pandas as pd

from src.sentiment import score_sentiment, add_sentiment_scores


# ---------------------------------------------------------------------------
# score_sentiment (VADER)
# ---------------------------------------------------------------------------


class TestScoreSentiment:
    def test_positive_text(self):
        scores = score_sentiment("The economy is booming and growth is excellent.")
        assert scores["compound"] > 0

    def test_negative_text(self):
        scores = score_sentiment("The economy collapsed and unemployment surged.")
        assert scores["compound"] < 0

    def test_neutral_text(self):
        scores = score_sentiment("The report was released on Tuesday.")
        assert -0.3 < scores["compound"] < 0.3

    def test_returns_all_keys(self):
        scores = score_sentiment("Some text.")
        assert set(scores.keys()) >= {"compound", "pos", "neg", "neu"}

    def test_compound_range(self):
        scores = score_sentiment("Terrible, horrible, awful economic collapse!")
        assert -1.0 <= scores["compound"] <= 1.0

    def test_empty_string(self):
        scores = score_sentiment("")
        assert scores["compound"] == 0.0
        assert scores["neu"] == 1.0

    def test_non_string_input(self):
        scores = score_sentiment(None)
        assert scores["compound"] == 0.0

    def test_whitespace_only(self):
        scores = score_sentiment("   ")
        assert scores["compound"] == 0.0


# ---------------------------------------------------------------------------
# add_sentiment_scores
# ---------------------------------------------------------------------------


class TestAddSentimentScores:
    def test_adds_vader_columns(self, sample_beige_df):
        result = add_sentiment_scores(sample_beige_df)
        for col in ["vader_compound", "vader_pos", "vader_neg", "vader_neu"]:
            assert col in result.columns

    def test_preserves_original_columns(self, sample_beige_df):
        result = add_sentiment_scores(sample_beige_df)
        assert "date" in result.columns
        assert "district" in result.columns
        assert "summary" in result.columns

    def test_same_number_of_rows(self, sample_beige_df):
        result = add_sentiment_scores(sample_beige_df)
        assert len(result) == len(sample_beige_df)

    def test_does_not_mutate_input(self, sample_beige_df):
        original_cols = list(sample_beige_df.columns)
        add_sentiment_scores(sample_beige_df)
        assert list(sample_beige_df.columns) == original_cols

    def test_compound_values_in_range(self, sample_beige_df):
        result = add_sentiment_scores(sample_beige_df)
        assert result["vader_compound"].between(-1, 1).all()

    def test_custom_text_column(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01"]),
                "district": ["Boston"],
                "notes": ["Activity expanded strongly."],
            }
        )
        result = add_sentiment_scores(df, text_col="notes")
        assert "vader_compound" in result.columns
        assert result["vader_compound"].iloc[0] > 0
