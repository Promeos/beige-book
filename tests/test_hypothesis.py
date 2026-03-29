"""
Tests for src/hypothesis.py — correlation and Granger causality tests.
"""

import numpy as np
import pandas as pd

from src.hypothesis import (
    evaluate_p_value,
    compute_lagged_correlations,
    run_granger_tests,
    compute_regional_correlations,
    compute_sector_correlations,
    compute_sector_district_correlations,
    compute_sector_indicator_correlations,
    run_sector_granger_tests,
)


# ---------------------------------------------------------------------------
# evaluate_p_value
# ---------------------------------------------------------------------------


class TestEvaluatePValue:
    def test_significant(self):
        assert evaluate_p_value(0.01) is True

    def test_not_significant(self):
        assert evaluate_p_value(0.10) is False

    def test_boundary_at_alpha(self):
        # p == ALPHA (0.05) should NOT be significant (strict <)
        assert evaluate_p_value(0.05) is False


# ---------------------------------------------------------------------------
# compute_lagged_correlations
# ---------------------------------------------------------------------------


class TestComputeLaggedCorrelations:
    def test_returns_dataframe(self, sample_national_agg):
        result = compute_lagged_correlations(sample_national_agg)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, sample_national_agg):
        result = compute_lagged_correlations(sample_national_agg)
        expected = {
            "indicator",
            "lag",
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
        }
        assert set(result.columns) == expected

    def test_correlations_in_range(self, sample_national_agg):
        result = compute_lagged_correlations(sample_national_agg)
        assert result["pearson_r"].between(-1, 1).all()
        assert result["spearman_r"].between(-1, 1).all()

    def test_p_values_in_range(self, sample_national_agg):
        result = compute_lagged_correlations(sample_national_agg)
        assert (result["pearson_p"] >= 0).all()
        assert (result["pearson_p"] <= 1).all()

    def test_skips_missing_columns(self):
        df = pd.DataFrame(
            {
                "sentiment_mean": np.random.randn(20),
                "GDPC1": np.random.randn(20),
            }
        )
        result = compute_lagged_correlations(df, indicator_cols=["GDPC1", "MISSING"])
        indicators = result["indicator"].unique()
        assert "GDPC1" in indicators
        assert "MISSING" not in indicators

    def test_skips_short_series(self):
        df = pd.DataFrame(
            {
                "sentiment_mean": np.random.randn(5),
                "GDPC1": np.random.randn(5),
            }
        )
        result = compute_lagged_correlations(df, indicator_cols=["GDPC1"], max_lag=0)
        # Only 5 observations < threshold of 10
        assert len(result) == 0

    def test_custom_max_lag(self, sample_national_agg):
        result = compute_lagged_correlations(
            sample_national_agg, indicator_cols=["GDPC1"], max_lag=2
        )
        assert result["lag"].max() <= 2


# ---------------------------------------------------------------------------
# run_granger_tests
# ---------------------------------------------------------------------------


class TestRunGrangerTests:
    def test_returns_dict(self, sample_national_agg):
        result = run_granger_tests(
            sample_national_agg, indicator_cols=["GDPC1"], max_lag=2
        )
        assert isinstance(result, dict)

    def test_result_structure(self, sample_national_agg):
        result = run_granger_tests(
            sample_national_agg, indicator_cols=["GDPC1"], max_lag=2
        )
        if "GDPC1" in result:
            for lag in [1, 2]:
                entry = result["GDPC1"][lag]
                assert "f_stat" in entry
                assert "p_value" in entry
                assert "significant" in entry
                assert 0 <= entry["p_value"] <= 1

    def test_skips_insufficient_data(self):
        df = pd.DataFrame(
            {
                "sentiment_mean": np.random.randn(8),
                "GDPC1": np.random.randn(8),
            }
        )
        result = run_granger_tests(df, indicator_cols=["GDPC1"], max_lag=4)
        # 8 < max_lag + 10 = 14, so should skip
        assert "GDPC1" not in result


# ---------------------------------------------------------------------------
# compute_regional_correlations
# ---------------------------------------------------------------------------


class TestComputeRegionalCorrelations:
    def test_returns_dataframe(self):
        np.random.seed(0)
        df = pd.DataFrame(
            {
                "district": ["Boston"] * 20 + ["New York"] * 20,
                "vader_compound": np.random.randn(40),
                "coincident_index": np.random.randn(40),
            }
        )
        result = compute_regional_correlations(df)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self):
        np.random.seed(0)
        df = pd.DataFrame(
            {
                "district": ["Boston"] * 15,
                "vader_compound": np.random.randn(15),
                "coincident_index": np.random.randn(15),
            }
        )
        result = compute_regional_correlations(df)
        expected = {"district", "correlation", "p_value", "n_obs"}
        assert set(result.columns) == expected

    def test_skips_small_districts(self):
        df = pd.DataFrame(
            {
                "district": ["Boston"] * 5,
                "vader_compound": np.random.randn(5),
                "coincident_index": np.random.randn(5),
            }
        )
        result = compute_regional_correlations(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# sector-level statistical tests
# ---------------------------------------------------------------------------


class TestSectorCorrelationFunctions:
    def test_sector_correlations_return_expected_sectors(self, sample_sector_pipeline_inputs):
        result = compute_sector_correlations(
            sample_sector_pipeline_inputs["sector_df"],
            sample_sector_pipeline_inputs["regional_df"],
        )
        assert set(result["sector"]) == {"Employment", "Manufacturing"}
        assert result["correlation"].gt(0.9).all()

    def test_sector_district_correlations_cover_each_pair(
        self, sample_sector_pipeline_inputs
    ):
        result = compute_sector_district_correlations(
            sample_sector_pipeline_inputs["sector_df"],
            sample_sector_pipeline_inputs["regional_df"],
        )
        pairs = set(zip(result["sector"], result["district"]))
        expected = {
            ("Employment", "Boston"),
            ("Employment", "New York"),
            ("Manufacturing", "Boston"),
            ("Manufacturing", "New York"),
        }
        assert pairs == expected
        assert result["p_value"].between(0, 1).all()

    def test_sector_indicator_correlations_include_requested_lags(
        self, sample_sector_pipeline_inputs
    ):
        result = compute_sector_indicator_correlations(
            sample_sector_pipeline_inputs["sector_merged_df"], max_lag=2
        )
        grouped = result.groupby("sector")["lag"].apply(list).to_dict()
        assert grouped == {
            "Employment": [0, 1, 2],
            "Manufacturing": [0, 1, 2],
        }

    def test_sector_granger_tests_find_signal(self, sample_sector_pipeline_inputs):
        result = run_sector_granger_tests(
            sample_sector_pipeline_inputs["sector_merged_df"], max_lag=2
        )
        assert set(result) == {"Employment", "Manufacturing"}
        assert result["Employment"][1]["significant"]
        assert result["Manufacturing"][1]["significant"]
