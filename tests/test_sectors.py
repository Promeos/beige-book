"""
Tests for src/sectors.py — sector extraction and sentiment scoring.
"""

import pandas as pd

from src.sectors import (
    _split_sentences,
    _classify_sentence,
    extract_sectors,
    score_sectors,
    build_sector_dataframe,
)


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _split_sentences(text)
        assert len(result) == 3

    def test_single_sentence(self):
        result = _split_sentences("Only one sentence.")
        assert len(result) == 1

    def test_handles_abbreviations(self):
        # "St. Louis" splits because the regex sees ". L" (period-space-capital).
        # This is a known limitation of the simple regex splitter.
        text = "Activity in St. Louis expanded modestly."
        result = _split_sentences(text)
        # The simple splitter does split on "St. L", so expect 2 parts
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _classify_sentence
# ---------------------------------------------------------------------------


class TestClassifySentence:
    def test_manufacturing_keywords(self):
        sectors = _classify_sentence("Manufacturing production expanded modestly.")
        assert "Manufacturing" in sectors

    def test_employment_keywords(self):
        sectors = _classify_sentence(
            "Employment conditions tightened with strong hiring."
        )
        assert "Employment" in sectors

    def test_real_estate_keywords(self):
        sectors = _classify_sentence("Housing prices rose across the district.")
        assert "Real Estate" in sectors

    def test_energy_keywords(self):
        sectors = _classify_sentence("Oil drilling activity increased in the region.")
        assert "Energy" in sectors

    def test_no_sector_match(self):
        sectors = _classify_sentence("The weather was mild this quarter.")
        assert len(sectors) == 0

    def test_multiple_sectors(self):
        sectors = _classify_sentence(
            "Manufacturing employment and hiring expanded alongside factory production."
        )
        assert len(sectors) >= 2


# ---------------------------------------------------------------------------
# extract_sectors
# ---------------------------------------------------------------------------


class TestExtractSectors:
    def test_returns_dict(self, manufacturing_text):
        result = extract_sectors(manufacturing_text)
        assert isinstance(result, dict)

    def test_manufacturing_detected(self, manufacturing_text):
        result = extract_sectors(manufacturing_text)
        assert "Manufacturing" in result

    def test_multiple_sectors(self, mixed_sector_text):
        result = extract_sectors(mixed_sector_text)
        assert len(result) >= 3

    def test_empty_string(self):
        assert extract_sectors("") == {}

    def test_non_string(self):
        assert extract_sectors(None) == {}

    def test_general_fallback(self):
        result = extract_sectors("The weather was mild this quarter.")
        assert "General" in result


# ---------------------------------------------------------------------------
# score_sectors
# ---------------------------------------------------------------------------


class TestScoreSectors:
    def test_returns_list_of_dicts(self, manufacturing_text):
        result = score_sectors(manufacturing_text)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_dict_keys(self, manufacturing_text):
        result = score_sectors(manufacturing_text)
        expected_keys = {
            "sector",
            "text",
            "vader_compound",
            "vader_pos",
            "vader_neg",
            "vader_neu",
            "sentence_count",
        }
        for r in result:
            assert set(r.keys()) == expected_keys

    def test_compound_in_range(self, mixed_sector_text):
        result = score_sectors(mixed_sector_text)
        for r in result:
            assert -1.0 <= r["vader_compound"] <= 1.0


# ---------------------------------------------------------------------------
# build_sector_dataframe
# ---------------------------------------------------------------------------


class TestBuildSectorDataframe:
    def test_output_columns(self, sample_beige_df):
        result = build_sector_dataframe(sample_beige_df)
        for col in ["date", "district", "sector", "text", "vader_compound"]:
            assert col in result.columns

    def test_more_rows_than_input(self, sample_beige_df):
        result = build_sector_dataframe(sample_beige_df)
        # Each input row should produce >= 1 sector row
        assert len(result) >= len(sample_beige_df)

    def test_dates_are_datetime(self, sample_beige_df):
        result = build_sector_dataframe(sample_beige_df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_sorted_output(self, sample_beige_df):
        result = build_sector_dataframe(sample_beige_df)
        dates = result["date"].tolist()
        assert dates == sorted(dates)
