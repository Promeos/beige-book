"""
Tests for robustness analysis module.
"""

import numpy as np
import pandas as pd
import pytest

from src.robustness import (
    run_adf_tests,
    compute_differenced_correlations,
    run_differenced_granger_tests,
    run_exclude_covid_oos,
    apply_fdr_correction,
    run_sector_fdr_correction,
)


@pytest.fixture
def merged_df_large():
    """Merged national dataset large enough for all robustness tests."""
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2013-01-01", periods=n, freq="QS")
    sentiment = np.random.uniform(-0.3, 0.6, n)
    return pd.DataFrame(
        {
            "date": dates,
            "sentiment_mean": sentiment,
            "GDPC1": 18000 + np.cumsum(np.random.randn(n) * 50),
            "UNRATE": 5.0 + np.cumsum(np.random.randn(n) * 0.1),
            "CPIAUCSL": 250 + np.cumsum(np.random.randn(n) * 0.5),
            "SP500": 3000 + np.cumsum(np.random.randn(n) * 30),
        }
    )


@pytest.fixture
def sector_district_corr_df():
    """Sample sector-district correlation results for FDR testing."""
    np.random.seed(42)
    sectors = ["Manufacturing", "Employment", "Real Estate"]
    districts = ["Boston", "Cleveland", "Chicago", "Dallas"]
    rows = []
    for s in sectors:
        for d in districts:
            rows.append(
                {
                    "sector": s,
                    "district": d,
                    "correlation": np.random.uniform(-0.5, 0.7),
                    "p_value": np.random.uniform(0.001, 0.2),
                    "n_obs": 80,
                }
            )
    return pd.DataFrame(rows)


class TestADFTests:
    """Tests for Augmented Dickey-Fuller unit root tests."""

    def test_returns_dataframe(self, merged_df_large):
        result = run_adf_tests(merged_df_large)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_expected_columns(self, merged_df_large):
        result = run_adf_tests(merged_df_large)
        expected = {"series", "adf_stat", "p_value", "stationary", "n_obs"}
        assert expected.issubset(set(result.columns))

    def test_includes_levels_and_differenced(self, merged_df_large):
        result = run_adf_tests(merged_df_large)
        series_names = result["series"].tolist()
        # Should have both levels and differenced
        assert "sentiment_mean" in series_names
        assert "Δsentiment_mean" in series_names

    def test_random_walk_is_nonstationary(self):
        """A random walk should be detected as non-stationary."""
        np.random.seed(99)
        n = 100
        df = pd.DataFrame(
            {
                "sentiment_mean": np.cumsum(np.random.randn(n)),
                "UNRATE": np.cumsum(np.random.randn(n)),
            }
        )
        result = run_adf_tests(df, indicator_cols=["UNRATE"])
        levels = result[~result["series"].str.startswith("Δ")]
        # Random walks should generally be non-stationary
        assert not levels["stationary"].all()

    def test_stationary_series_detected(self):
        """White noise should be detected as stationary."""
        np.random.seed(99)
        n = 200
        df = pd.DataFrame(
            {
                "sentiment_mean": np.random.randn(n),
                "UNRATE": np.random.randn(n),
            }
        )
        result = run_adf_tests(df, indicator_cols=["UNRATE"])
        levels = result[~result["series"].str.startswith("Δ")]
        assert levels["stationary"].all()


class TestDifferencedCorrelations:
    """Tests for first-differenced correlations."""

    def test_returns_dataframe(self, merged_df_large):
        result = compute_differenced_correlations(merged_df_large)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, merged_df_large):
        result = compute_differenced_correlations(merged_df_large)
        expected = {"indicator", "lag", "pearson_r", "pearson_p"}
        assert expected.issubset(set(result.columns))

    def test_multiple_lags(self, merged_df_large):
        result = compute_differenced_correlations(merged_df_large, max_lag=2)
        for col in ["UNRATE", "CPIAUCSL", "SP500"]:
            col_results = result[result["indicator"] == col]
            lags = col_results["lag"].tolist()
            assert 0 in lags
            assert 1 in lags
            assert 2 in lags

    def test_correlations_bounded(self, merged_df_large):
        result = compute_differenced_correlations(merged_df_large)
        assert (result["pearson_r"].abs() <= 1.0).all()


class TestDifferencedGranger:
    """Tests for first-differenced Granger causality tests."""

    def test_returns_dict(self, merged_df_large):
        result = run_differenced_granger_tests(merged_df_large, max_lag=2)
        assert isinstance(result, dict)

    def test_has_indicator_keys(self, merged_df_large):
        result = run_differenced_granger_tests(merged_df_large, max_lag=2)
        assert len(result) > 0
        for key in result:
            assert key in ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]

    def test_lag_structure(self, merged_df_large):
        result = run_differenced_granger_tests(merged_df_large, max_lag=2)
        for indicator, lags in result.items():
            assert 1 in lags
            assert 2 in lags
            for lag, vals in lags.items():
                assert "f_stat" in vals
                assert "p_value" in vals
                assert "significant" in vals


class TestExcludeCovidOOS:
    """Tests for exclude-COVID out-of-sample evaluation."""

    def test_returns_dict(self):
        np.random.seed(42)
        n = 120
        dates = pd.date_range("2012-01-01", periods=n, freq="MS")
        df = pd.DataFrame(
            {
                "date": dates,
                "sentiment_mean": np.random.uniform(-0.3, 0.6, n),
                "UNRATE": 5.0 + np.cumsum(np.random.randn(n) * 0.05),
            }
        )
        result = run_exclude_covid_oos(df, indicator_cols=["UNRATE"])
        assert isinstance(result, dict)

    def test_skips_insufficient_data(self):
        """Should handle cases with too few test observations."""
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", periods=30, freq="MS")
        df = pd.DataFrame(
            {
                "date": dates,
                "sentiment_mean": np.random.uniform(-0.3, 0.6, 30),
                "UNRATE": 5.0 + np.random.randn(30) * 0.1,
            }
        )
        # All data before COVID cutoff — test set will be empty
        result = run_exclude_covid_oos(df, indicator_cols=["UNRATE"])
        assert len(result) == 0

    def test_metrics_structure(self):
        np.random.seed(42)
        n = 150
        dates = pd.date_range("2012-01-01", periods=n, freq="MS")
        df = pd.DataFrame(
            {
                "date": dates,
                "sentiment_mean": np.random.uniform(-0.3, 0.6, n),
                "UNRATE": 5.0 + np.cumsum(np.random.randn(n) * 0.05),
            }
        )
        result = run_exclude_covid_oos(df, indicator_cols=["UNRATE"])
        if "UNRATE" in result:
            assert "baseline" in result["UNRATE"]
            assert "sentiment_model" in result["UNRATE"]
            assert "rmse" in result["UNRATE"]["baseline"]


class TestFDRCorrection:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_adds_columns(self):
        df = pd.DataFrame(
            {
                "sector": ["A", "B", "C", "D"],
                "p_value": [0.001, 0.01, 0.04, 0.5],
            }
        )
        result = apply_fdr_correction(df)
        assert "p_adjusted" in result.columns
        assert "significant_fdr" in result.columns

    def test_adjusted_p_geq_raw(self):
        """FDR-adjusted p-values should be >= raw p-values."""
        df = pd.DataFrame(
            {
                "p_value": [0.001, 0.01, 0.03, 0.05, 0.1, 0.5],
            }
        )
        result = apply_fdr_correction(df)
        assert (result["p_adjusted"] >= result["p_value"] - 1e-10).all()

    def test_fewer_or_equal_significant(self):
        """FDR correction should reject fewer or same number of hypotheses."""
        df = pd.DataFrame(
            {
                "p_value": [0.001, 0.02, 0.04, 0.06, 0.1, 0.3],
            }
        )
        result = apply_fdr_correction(df)
        n_raw = (result["p_value"] < 0.05).sum()
        n_fdr = result["significant_fdr"].sum()
        assert n_fdr <= n_raw

    def test_empty_dataframe(self):
        df = pd.DataFrame({"p_value": []})
        result = apply_fdr_correction(df)
        assert len(result) == 0

    def test_all_significant_remain(self):
        """Very small p-values should survive FDR correction."""
        df = pd.DataFrame(
            {
                "p_value": [0.0001, 0.0002, 0.0003],
            }
        )
        result = apply_fdr_correction(df)
        assert result["significant_fdr"].all()


class TestSectorFDRCorrection:
    """Tests for sector-district FDR correction wrapper."""

    def test_returns_dataframe(self, sector_district_corr_df):
        result = run_sector_fdr_correction(sector_district_corr_df)
        assert isinstance(result, pd.DataFrame)
        assert "p_adjusted" in result.columns
        assert "significant_fdr" in result.columns

    def test_preserves_original_columns(self, sector_district_corr_df):
        result = run_sector_fdr_correction(sector_district_corr_df)
        for col in ["sector", "district", "correlation", "p_value"]:
            assert col in result.columns
