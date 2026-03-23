"""
Beige Book Sentiment Analysis source package.

Exposes core functions for data acquisition, preparation, sentiment
scoring, visualization, and statistical testing.
"""

from .acquire import (
    get_beige_data,
    get_fred_data,
    get_regional_fred_data,
    get_sector_fred_data,
)
from .prepare import (
    prep_beige_data,
    align_time_periods,
    compute_national_aggregate,
    align_regional_data,
    compute_sector_national_aggregate,
    align_sector_with_indicators,
)
from .sentiment import add_sentiment_scores
from .explore import (
    plot_sentiment_timeseries,
    plot_regional_comparison,
    plot_sentiment_vs_indicator,
    plot_correlation_matrix,
    plot_regional_sentiment_vs_economy,
    plot_regional_correlation_bars,
    plot_district_timeseries_grid,
    plot_sector_heatmap,
    plot_sector_timeseries,
    plot_sector_district_grid,
    plot_sector_volatility,
    plot_sector_vs_indicator,
    plot_sector_predictive_grid,
)
from .hypothesis import (
    evaluate_p_value,
    compute_lagged_correlations,
    run_granger_tests,
    compute_regional_correlations,
    compute_sector_correlations,
    compute_sector_district_correlations,
    compute_sector_indicator_correlations,
    run_sector_granger_tests,
)
from .model import run_sector_regressions, sector_out_of_sample_test
from .sectors import (
    extract_sectors,
    score_sectors,
    build_sector_dataframe,
    build_sentence_sector_dataframe,
    aggregate_sentence_sector_scores,
)
from .maps import (
    plot_sector_map,
    plot_sector_map_grid,
    plot_dominant_sector_map,
    plot_sector_map_animated,
)

__all__ = [
    "get_beige_data",
    "get_fred_data",
    "get_regional_fred_data",
    "prep_beige_data",
    "align_time_periods",
    "align_regional_data",
    "compute_national_aggregate",
    "add_sentiment_scores",
    "plot_sentiment_timeseries",
    "plot_regional_comparison",
    "plot_sentiment_vs_indicator",
    "plot_correlation_matrix",
    "plot_regional_sentiment_vs_economy",
    "plot_regional_correlation_bars",
    "plot_district_timeseries_grid",
    "evaluate_p_value",
    "compute_lagged_correlations",
    "run_granger_tests",
    "compute_regional_correlations",
    "compute_sector_correlations",
    "compute_sector_district_correlations",
    "plot_sector_heatmap",
    "plot_sector_timeseries",
    "plot_sector_district_grid",
    "plot_sector_volatility",
    "extract_sectors",
    "score_sectors",
    "build_sector_dataframe",
    "plot_sector_map",
    "plot_sector_map_grid",
    "plot_dominant_sector_map",
    "plot_sector_map_animated",
    "get_sector_fred_data",
    "compute_sector_national_aggregate",
    "align_sector_with_indicators",
    "plot_sector_vs_indicator",
    "plot_sector_predictive_grid",
    "compute_sector_indicator_correlations",
    "run_sector_granger_tests",
    "run_sector_regressions",
    "sector_out_of_sample_test",
    "build_sentence_sector_dataframe",
    "aggregate_sentence_sector_scores",
]
