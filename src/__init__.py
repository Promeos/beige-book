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
)
from .hypothesis import (
    evaluate_p_value,
    compute_lagged_correlations,
    run_granger_tests,
    compute_regional_correlations,
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
]
