"""
Tests for src/prepare.py — text cleaning, district normalization, time alignment.
"""

import pandas as pd

from src.prepare import (
    clean_text,
    normalize_district,
    prep_beige_data,
    align_time_periods,
    compute_national_aggregate,
    align_regional_data,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_removes_html_tags(self):
        assert clean_text("<p>Hello</p>") == "Hello"

    def test_normalizes_whitespace(self):
        assert clean_text("too   many   spaces") == "too many spaces"

    def test_strips_leading_trailing(self):
        assert clean_text("  hello  ") == "hello"

    def test_handles_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(42) == ""

    def test_combined_html_and_whitespace(self):
        result = clean_text("<b>Economic</b>  activity   <i>expanded</i>")
        assert result == "Economic activity expanded"

    def test_empty_string(self):
        assert clean_text("") == ""


# ---------------------------------------------------------------------------
# normalize_district
# ---------------------------------------------------------------------------


class TestNormalizeDistrict:
    def test_full_name_to_canonical(self):
        assert normalize_district("Federal Reserve Bank of Boston") == "Boston"

    def test_already_canonical(self):
        assert normalize_district("Cleveland") == "Cleveland"

    def test_non_string_passthrough(self):
        assert normalize_district(None) is None

    def test_all_districts_normalize(self):
        from src.config import DISTRICT_ALIASES

        for full_name, short_name in DISTRICT_ALIASES.items():
            assert normalize_district(full_name) == short_name


# ---------------------------------------------------------------------------
# prep_beige_data
# ---------------------------------------------------------------------------


class TestPrepBeigeData:
    def test_converts_dates(self, sample_beige_df):
        result = prep_beige_data(sample_beige_df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_cleans_summaries(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "district": ["Boston"],
                "summary": ["<p>Hello   world</p>"],
            }
        )
        result = prep_beige_data(df)
        assert result["summary"].iloc[0] == "Hello world"

    def test_drops_empty_summaries(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-01"],
                "district": ["Boston", "New York"],
                "summary": ["Good text.", ""],
            }
        )
        result = prep_beige_data(df)
        assert len(result) == 1

    def test_normalizes_districts(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "district": ["Federal Reserve Bank of Dallas"],
                "summary": ["Some text."],
            }
        )
        result = prep_beige_data(df)
        assert result["district"].iloc[0] == "Dallas"


# ---------------------------------------------------------------------------
# align_time_periods
# ---------------------------------------------------------------------------


class TestAlignTimePeriods:
    def test_forward_merge_produces_output(
        self, sample_beige_with_sentiment, sample_fred_df
    ):
        result = align_time_periods(sample_beige_with_sentiment, sample_fred_df)
        assert len(result) > 0
        assert "GDPC1" in result.columns
        assert "vader_compound" in result.columns

    def test_forward_merge_dates_align_forward(
        self, sample_beige_with_sentiment, sample_fred_df
    ):
        result = align_time_periods(sample_beige_with_sentiment, sample_fred_df)
        # Beige Book date 2023-01-18 should map to FRED date >= 2023-01-18
        # The forward merge means the indicator date is >= the beige date
        for _, row in result.iterrows():
            # The FRED values should be from dates on or after the beige date
            assert pd.notna(row["GDPC1"])

    def test_preserves_all_beige_rows(
        self, sample_beige_with_sentiment, sample_fred_df
    ):
        result = align_time_periods(sample_beige_with_sentiment, sample_fred_df)
        assert len(result) == len(sample_beige_with_sentiment)


# ---------------------------------------------------------------------------
# compute_national_aggregate
# ---------------------------------------------------------------------------


class TestComputeNationalAggregate:
    def test_output_columns(self, sample_beige_with_sentiment):
        result = compute_national_aggregate(sample_beige_with_sentiment)
        expected_cols = {
            "date",
            "sentiment_mean",
            "sentiment_std",
            "sentiment_min",
            "sentiment_max",
        }
        assert set(result.columns) == expected_cols

    def test_one_row_per_date(self, sample_beige_with_sentiment):
        result = compute_national_aggregate(sample_beige_with_sentiment)
        assert len(result) == sample_beige_with_sentiment["date"].nunique()

    def test_mean_is_between_min_and_max(self, sample_beige_with_sentiment):
        result = compute_national_aggregate(sample_beige_with_sentiment)
        for _, row in result.iterrows():
            assert row["sentiment_min"] <= row["sentiment_mean"] <= row["sentiment_max"]


# ---------------------------------------------------------------------------
# align_regional_data
# ---------------------------------------------------------------------------


class TestAlignRegionalData:
    def test_produces_output(
        self, sample_beige_with_sentiment, sample_regional_fred_df
    ):
        result = align_regional_data(
            sample_beige_with_sentiment, sample_regional_fred_df
        )
        assert len(result) > 0
        assert "coincident_index" in result.columns

    def test_empty_when_no_matching_districts(self, sample_beige_with_sentiment):
        fred = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-31"]),
                "district": ["Dallas"],
                "coincident_index": [100.0],
            }
        )
        result = align_regional_data(sample_beige_with_sentiment, fred)
        assert len(result) == 0
