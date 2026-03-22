"""
Beige Book Sentiment Analysis source package.

Exposes core functions for data acquisition, preparation, sentiment
scoring, visualization, and statistical testing.
"""

from .acquire import get_beige_data, get_fred_data
from .prepare import prep_beige_data, align_time_periods, compute_national_aggregate
from .sentiment import add_sentiment_scores
from .explore import (
    plot_sentiment_timeseries,
    plot_regional_comparison,
    plot_sentiment_vs_indicator,
    plot_correlation_matrix,
)
from .hypothesis import (
    evaluate_p_value,
    compute_lagged_correlations,
    run_granger_tests,
)

__all__ = [
    "get_beige_data",
    "get_fred_data",
    "prep_beige_data",
    "align_time_periods",
    "compute_national_aggregate",
    "add_sentiment_scores",
    "plot_sentiment_timeseries",
    "plot_regional_comparison",
    "plot_sentiment_vs_indicator",
    "plot_correlation_matrix",
    "evaluate_p_value",
    "compute_lagged_correlations",
    "run_granger_tests",
]
