"""
Beige Book Sentiment Analysis source package.

Exposes core functions for data acquisition, preparation, sentiment
scoring, visualization, and statistical testing.
"""

from .acquire import get_beige_data, get_fred_data, get_regional_fred_data
from .prepare import (
    prep_beige_data, align_time_periods, compute_national_aggregate,
    align_regional_data,
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
)
from .hypothesis import (
    evaluate_p_value,
    compute_lagged_correlations,
    run_granger_tests,
    compute_regional_correlations,
    compute_sector_correlations,
    compute_sector_district_correlations,
)
from .sectors import extract_sectors, score_sectors, build_sector_dataframe
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
]
