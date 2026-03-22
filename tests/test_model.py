"""
Tests for src/model.py — OLS regression and out-of-sample evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from src.model import (
    run_ols_regression,
    run_all_regressions,
    out_of_sample_test,
    _directional_accuracy,
)


# ---------------------------------------------------------------------------
# run_ols_regression
# ---------------------------------------------------------------------------


class TestRunOlsRegression:
    def test_returns_model(self, sample_national_agg):
        result = run_ols_regression(sample_national_agg, "GDPC1")
        assert result is not None
        assert hasattr(result, "rsquared")

    def test_rsquared_in_range(self, sample_national_agg):
        result = run_ols_regression(sample_national_agg, "GDPC1")
        assert 0.0 <= result.rsquared <= 1.0

    def test_returns_none_for_insufficient_data(self):
        df = pd.DataFrame(
            {
                "sentiment_mean": np.random.randn(10),
                "GDPC1": np.random.randn(10),
            }
        )
        result = run_ols_regression(df, "GDPC1")
        assert result is None

    def test_with_controls(self, sample_national_agg):
        result = run_ols_regression(sample_national_agg, "GDPC1", controls=["UNRATE"])
        assert result is not None
        # Should have 3 params: const, sentiment_mean, UNRATE
        assert len(result.params) == 3

    def test_coefficients_exist(self, sample_national_agg):
        result = run_ols_regression(sample_national_agg, "GDPC1")
        assert "sentiment_mean" in result.params.index
        assert "const" in result.params.index


# ---------------------------------------------------------------------------
# run_all_regressions
# ---------------------------------------------------------------------------


class TestRunAllRegressions:
    def test_returns_dict(self, sample_national_agg):
        result = run_all_regressions(sample_national_agg, indicator_cols=["GDPC1"])
        assert isinstance(result, dict)
        assert "GDPC1" in result

    def test_simple_and_controlled(self, sample_national_agg):
        result = run_all_regressions(sample_national_agg, indicator_cols=["GDPC1"])
        assert "simple" in result["GDPC1"]
        assert "controlled" in result["GDPC1"]


# ---------------------------------------------------------------------------
# out_of_sample_test
# ---------------------------------------------------------------------------


class TestOutOfSampleTest:
    def test_returns_metrics(self):
        np.random.seed(42)
        n = 80
        dates = pd.date_range("2010-01-01", periods=n, freq="QS")
        df = pd.DataFrame(
            {
                "date": dates,
                "sentiment_mean": np.random.uniform(-0.5, 0.5, n),
                "GDPC1": 18000 + np.cumsum(np.random.randn(n) * 50),
            }
        )
        result = out_of_sample_test(df, "GDPC1", train_end="2018-12-31")
        assert result is not None
        assert "baseline" in result
        assert "sentiment_model" in result

    def test_metrics_keys(self, sample_national_agg):
        result = out_of_sample_test(
            sample_national_agg, "GDPC1", train_end="2018-12-31"
        )
        if result is not None:
            for model_key in ["baseline", "sentiment_model"]:
                assert "rmse" in result[model_key]
                assert "mae" in result[model_key]
                assert "directional_accuracy" in result[model_key]

    def test_rmse_non_negative(self, sample_national_agg):
        result = out_of_sample_test(
            sample_national_agg, "GDPC1", train_end="2018-12-31"
        )
        if result is not None:
            assert result["baseline"]["rmse"] >= 0
            assert result["sentiment_model"]["rmse"] >= 0

    def test_returns_none_insufficient_data(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="QS"),
                "sentiment_mean": np.random.randn(10),
                "GDPC1": np.random.randn(10),
            }
        )
        result = out_of_sample_test(df, "GDPC1", train_end="2020-06-01")
        assert result is None


# ---------------------------------------------------------------------------
# _directional_accuracy
# ---------------------------------------------------------------------------


class TestDirectionalAccuracy:
    def test_perfect_accuracy(self):
        actual = [1, 2, 3, 4, 5]
        predicted = [1, 2, 3, 4, 5]
        assert _directional_accuracy(actual, predicted) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        actual = [1, 2, 3, 4, 5]
        predicted = [5, 4, 3, 2, 1]
        assert _directional_accuracy(actual, predicted) == pytest.approx(0.0)

    def test_empty_arrays(self):
        assert _directional_accuracy([], []) == 0.0

    def test_single_element(self):
        assert _directional_accuracy([1], [1]) == 0.0

    def test_range(self):
        actual = np.random.randn(50)
        predicted = np.random.randn(50)
        result = _directional_accuracy(actual, predicted)
        assert 0.0 <= result <= 1.0
