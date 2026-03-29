"""
Tests for reproducible reporting artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting import (
    filter_date_range,
    summarize_national_analysis,
    summarize_regional_analysis,
    summarize_sector_analysis,
    summarize_robustness,
    summarize_split_sample_stability,
    build_analysis_artifact,
    render_markdown_summary,
    write_analysis_artifact,
)
from src.hypothesis import compute_lagged_correlations


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


class TestSummaryBuilders:
    def test_national_summary_matches_lagged_correlations(self, sample_reporting_merged_df):
        summary = summarize_national_analysis(sample_reporting_merged_df)
        corr_df = compute_lagged_correlations(
            sample_reporting_merged_df,
            indicator_cols=["GDPC1", "UNRATE", "CPIAUCSL", "SP500"],
        )

        lag0 = corr_df[
            (corr_df["indicator"] == "UNRATE") & (corr_df["lag"] == 0)
        ].iloc[0]
        assert summary["indicators"]["UNRATE"]["lag0"]["pearson_r"] == pytest.approx(
            lag0["pearson_r"]
        )
        assert summary["indicators"]["UNRATE"]["lag0"]["pearson_p"] == pytest.approx(
            lag0["pearson_p"]
        )

    def test_regional_summary_returns_both_windows(self, sample_sector_pipeline_inputs):
        summary = summarize_regional_analysis(
            sample_sector_pipeline_inputs["regional_merged_df"],
            start="2017-01-01",
        )
        assert set(summary) == {"full_available_sample", "post_2011_sample"}
        assert summary["full_available_sample"]["top_districts"]
        assert summary["post_2011_sample"]["top_districts"]

    def test_sector_summary_includes_regional_block(self, sample_sector_pipeline_inputs):
        summary = summarize_sector_analysis(
            sample_sector_pipeline_inputs["sector_df"],
            sample_sector_pipeline_inputs["sector_fred_df"],
            regional_merged_df=sample_sector_pipeline_inputs["regional_df"],
            source_name="synthetic_sector.csv",
        )
        assert summary["dataset"]["source"] == "synthetic_sector.csv"
        assert "regional_correlations" in summary
        assert summary["indicator_correlations"]["top_lag0"]

    def test_robustness_summary_returns_expected_sections(self, sample_reporting_merged_df):
        summary = summarize_robustness(sample_reporting_merged_df, split_date="2018-01-01")
        assert set(summary) == {
            "adf",
            "differenced",
            "exclude_covid_oos",
            "split_sample_unrate",
        }
        assert "post_split" in summary["split_sample_unrate"]["periods"]


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


class TestArtifactRendering:
    def test_build_analysis_artifact_and_render_markdown(
        self, sample_reporting_merged_df, sample_sector_pipeline_inputs
    ):
        artifact = build_analysis_artifact(
            merged_df=sample_reporting_merged_df,
            regional_merged_df=sample_sector_pipeline_inputs["regional_merged_df"],
            sector_df=sample_sector_pipeline_inputs["sector_df"],
            sector_fred_df=sample_sector_pipeline_inputs["sector_fred_df"],
            source_files={"sector": "synthetic_sector.csv"},
        )

        markdown = render_markdown_summary(artifact)

        assert set(artifact) >= {
            "generated_at",
            "method_notes",
            "source_files",
            "national",
            "regional",
            "sector",
            "robustness",
        }
        assert "## National" in markdown
        assert "## Robustness" in markdown

    def test_render_markdown_handles_missing_optional_metrics(self):
        results = {
            "national": {
                "sample": {
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-01",
                    "n_dates": 12,
                },
                "indicators": {
                    "UNRATE": {
                        "lag0": {"pearson_r": -0.2, "pearson_p": 0.2},
                        "granger": {"significant_lags": []},
                        "controlled_ols": {"sentiment_p_value": None},
                        "out_of_sample": {"rmse_delta": None},
                    }
                },
            },
            "regional": {
                "full_available_sample": {"top_districts": []},
                "post_2011_sample": {"top_districts": []},
            },
            "sector": {
                "dataset": {
                    "source": "synthetic.csv",
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-01",
                    "n_dates": 12,
                    "n_rows": 24,
                },
                "indicator_correlations": {"top_lag0": []},
            },
            "robustness": {
                "exclude_covid_oos": {"UNRATE": {"rmse_delta": None}},
                "split_sample_unrate": {"periods": {}},
            },
        }

        markdown = render_markdown_summary(results)
        assert "Exclude-COVID UNRATE RMSE delta: --" in markdown
        assert "Full available sample top district: --" in markdown
