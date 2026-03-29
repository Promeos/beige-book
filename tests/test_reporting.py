"""
Tests for reproducible reporting artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting import (
    filter_date_range,
    summarize_split_sample_stability,
    write_analysis_artifact,
)


class TestFilterDateRange:
    def test_inclusive_bounds(self):
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])}
        )

        result = filter_date_range(df, start="2020-02-01", end="2020-03-01")

        assert result["date"].tolist() == list(
            pd.to_datetime(["2020-02-01", "2020-03-01"])
        )


class TestSummarizeSplitSampleStability:
    def test_returns_pre_post_and_full_periods(self):
        np.random.seed(42)
        n = 72
        dates = pd.date_range("2007-01-01", periods=n, freq="MS")
        sentiment = np.sin(np.arange(n) / 4) + np.random.randn(n) * 0.05
        unrate = 5 + np.cumsum(np.random.randn(n) * 0.03) - sentiment * 0.1

        df = pd.DataFrame(
            {
                "date": dates,
                "sentiment_mean": sentiment,
                "UNRATE": unrate,
            }
        )

        result = summarize_split_sample_stability(df, split_date="2010-01-01")

        assert result["target"] == "UNRATE"
        assert result["split_date"] == "2010-01-01"
        assert set(result["periods"]) == {"pre_split", "post_split", "full_sample"}
        assert result["periods"]["full_sample"]["n_obs"] > 20
        assert "pearson_r" in result["periods"]["post_split"]["diff_correlation"]


class TestWriteAnalysisArtifact:
    def test_writes_json_and_markdown(self, tmp_path):
        results = {
            "national": {
                "sample": {"start_date": "1996-10-30", "end_date": "2026-02-01", "n_dates": 235},
                "indicators": {
                    "UNRATE": {
                        "lag0": {"pearson_r": -0.45, "pearson_p": 0.0001},
                        "granger": {"significant_lags": [3, 4]},
                        "controlled_ols": {"sentiment_p_value": 0.0001},
                        "out_of_sample": {"rmse_delta": 0.05},
                    }
                },
            },
            "regional": {
                "full_available_sample": {
                    "top_districts": [{"district": "Atlanta", "correlation": 0.02}]
                },
                "post_2011_sample": {
                    "top_districts": [{"district": "Cleveland", "correlation": 0.66}]
                },
            },
            "sector": {
                "dataset": {
                    "source": "sector_sentiment.csv",
                    "start_date": "2011-01-01",
                    "end_date": "2025-11-01",
                    "n_dates": 120,
                    "n_rows": 9422,
                },
                "indicator_correlations": {
                    "top_lag0": [
                        {
                            "sector": "Employment",
                            "indicator_id": "PAYEMS",
                            "pearson_r": 0.51,
                            "pearson_p": 0.0001,
                        }
                    ]
                },
            },
            "robustness": {
                "exclude_covid_oos": {"UNRATE": {"rmse_delta": 0.0029}},
                "split_sample_unrate": {
                    "periods": {
                        "post_split": {"diff_correlation": {"pearson_r": -0.70}}
                    }
                },
            },
        }

        paths = write_analysis_artifact(results, tmp_path)

        assert Path(paths["json"]).exists()
        assert Path(paths["markdown"]).exists()
        assert "Analysis Results" in Path(paths["markdown"]).read_text(encoding="utf-8")
