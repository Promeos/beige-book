"""
Tests for src/acquire.py — scraping and FRED data fetching.

All external calls (HTTP requests, FRED API) are mocked.
"""

import sys
import types

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Ensure fredapi is available as a mock module so fetch_fred_data can import it
_fredapi_mock_module = types.ModuleType("fredapi")
_fredapi_mock_module.Fred = MagicMock
if "fredapi" not in sys.modules:
    sys.modules["fredapi"] = _fredapi_mock_module

from src.acquire import (
    _parse_date_from_url,
    _is_district_name,
    _normalize_district,
    _extract_district_summaries,
    fetch_fred_data,
)


# ---------------------------------------------------------------------------
# _parse_date_from_url
# ---------------------------------------------------------------------------


class TestParseDateFromUrl:
    def test_standard_url(self):
        url = "https://www.federalreserve.gov/monetarypolicy/beigebook202301.htm"
        assert _parse_date_from_url(url) == "2023-01-01"

    def test_summary_url_format(self):
        url = (
            "https://www.federalreserve.gov/monetarypolicy/beigebook202601-summary.htm"
        )
        assert _parse_date_from_url(url) == "2026-01-01"

    def test_no_match(self):
        assert _parse_date_from_url("https://example.com/page.htm") is None


# ---------------------------------------------------------------------------
# _is_district_name
# ---------------------------------------------------------------------------


class TestIsDistrictName:
    def test_canonical_names(self):
        assert _is_district_name("Boston") is True
        assert _is_district_name("San Francisco") is True
        assert _is_district_name("Kansas City") is True

    def test_full_bank_name(self):
        assert _is_district_name("Federal Reserve Bank of Cleveland") is True

    def test_not_a_district(self):
        assert _is_district_name("Summary of Economic Activity") is False
        assert _is_district_name("Labor Markets") is False


# ---------------------------------------------------------------------------
# _normalize_district
# ---------------------------------------------------------------------------


class TestNormalizeDistrictAcquire:
    def test_full_to_canonical(self):
        assert _normalize_district("Federal Reserve Bank of Boston") == "Boston"

    def test_already_canonical(self):
        assert _normalize_district("Dallas") == "Dallas"

    def test_unknown_name(self):
        assert _normalize_district("Unknown Place") == "Unknown Place"


# ---------------------------------------------------------------------------
# _extract_district_summaries
# ---------------------------------------------------------------------------


class TestExtractDistrictSummaries:
    def test_h5_format(self):
        html = """
        <html><body>
        <h5>Boston</h5>
        <p>Economic activity expanded modestly in recent weeks.</p>
        <h5>New York</h5>
        <p>Growth slowed amid rising uncertainty.</p>
        <h5>Philadelphia</h5><p>Activity was flat.</p>
        <h5>Cleveland</h5><p>Output rose.</p>
        <h5>Richmond</h5><p>Conditions improved.</p>
        <h5>Atlanta</h5><p>Growth was modest.</p>
        <h5>Chicago</h5><p>Activity expanded.</p>
        <h5>St. Louis</h5><p>Conditions held steady.</p>
        <h5>Minneapolis</h5><p>Activity grew slightly.</p>
        <h5>Kansas City</h5><p>Growth was moderate.</p>
        <h5>Dallas</h5><p>Activity expanded at a modest pace.</p>
        <h5>San Francisco</h5><p>Economic conditions improved.</p>
        </body></html>
        """
        results = _extract_district_summaries(html)
        assert len(results) == 12

    def test_empty_html(self):
        results = _extract_district_summaries("<html><body></body></html>")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# fetch_fred_data (mocked)
# ---------------------------------------------------------------------------


class TestFetchFredData:
    @patch("src.acquire.FRED_API_KEY", "fake_key")
    @patch("src.acquire.FRED_SERIES", {"GDPC1": "Real GDP", "UNRATE": "Unemployment"})
    def test_returns_dataframe(self):
        mock_fred = MagicMock()
        dates = pd.date_range("2023-01-01", periods=12, freq="ME")
        mock_fred.get_series.return_value = pd.Series(range(12), index=dates)

        with patch("fredapi.Fred", return_value=mock_fred):
            result = fetch_fred_data(start_date="2023-01-01", end_date="2023-12-31")

        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns

    @patch("src.acquire.FRED_API_KEY", None)
    def test_raises_without_api_key(self):
        with patch("fredapi.Fred"):
            with pytest.raises(ValueError, match="FRED_API_KEY"):
                fetch_fred_data()

    @patch("src.acquire.FRED_API_KEY", "fake_key")
    @patch("src.acquire.FRED_SERIES", {"GDPC1": "Real GDP"})
    def test_handles_api_error(self):
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = Exception("API error")

        with patch("fredapi.Fred", return_value=mock_fred):
            result = fetch_fred_data()

        assert isinstance(result, pd.DataFrame)
