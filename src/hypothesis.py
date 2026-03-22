"""
Statistical tests for Beige Book sentiment predictive analysis.

Tests whether Beige Book sentiment Granger-causes economic indicators
and computes lagged correlations.
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests

from src.config import ALPHA


def evaluate_p_value(p):
    """
    Determine if a p-value is statistically significant.

    Parameters
    ----------
    p : float

    Returns
    -------
    significant : bool
    """
    significant = p < ALPHA
    if significant:
        print(f"  p = {p:.4f} < {ALPHA:.2f} → Reject H₀")
    else:
        print(f"  p = {p:.4f} >= {ALPHA:.2f} → Fail to reject H₀")
    return significant


def compute_lagged_correlations(
    df, sentiment_col="sentiment_mean", indicator_cols=None, max_lag=3
):
    """
    Compute Pearson and Spearman correlations between sentiment
    and economic indicators at various lags.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Must have sentiment_col and indicator columns, sorted by date.
    sentiment_col : str
    indicator_cols : list of str
    max_lag : int
        Maximum number of periods to lag.

    Returns
    -------
    results : pandas.core.frame.DataFrame
        Rows are (indicator, lag), columns are pearson_r, pearson_p,
        spearman_r, spearman_p.
    """
    if indicator_cols is None:
        indicator_cols = ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]

    rows = []
    for col in indicator_cols:
        if col not in df.columns:
            continue
        for lag in range(max_lag + 1):
            # Shift indicator forward by lag periods (sentiment leads)
            shifted = df[col].shift(-lag)
            mask = df[sentiment_col].notna() & shifted.notna()

            if mask.sum() < 10:
                continue

            s = df.loc[mask, sentiment_col].values
            i = shifted[mask].values

            pr, pp = pearsonr(s, i)
            sr, sp = spearmanr(s, i)

            rows.append(
                {
                    "indicator": col,
                    "lag": lag,
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": sp,
                }
            )

    return pd.DataFrame(rows)


def run_granger_tests(
    df, sentiment_col="sentiment_mean", indicator_cols=None, max_lag=4
):
    """
    Run Granger causality tests: does sentiment Granger-cause indicators?

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Must have sentiment_col and indicator columns, sorted by date.
    sentiment_col : str
    indicator_cols : list of str
    max_lag : int

    Returns
    -------
    results : dict
        Nested dict: results[indicator][lag] = {f_stat, p_value, significant}.
    """
    if indicator_cols is None:
        indicator_cols = ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]

    results = {}
    for col in indicator_cols:
        if col not in df.columns:
            continue

        # Granger test needs a 2-column array: [dependent, predictor]
        pair = df[[col, sentiment_col]].dropna()
        if len(pair) < max_lag + 10:
            print(f"  Skipping {col}: insufficient observations ({len(pair)})")
            continue

        print(f"\nGranger Causality: {sentiment_col} → {col}")
        print("-" * 50)

        try:
            test_result = grangercausalitytests(
                pair.values, maxlag=max_lag, verbose=False
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue

        results[col] = {}
        for lag in range(1, max_lag + 1):
            f_stat = test_result[lag][0]["ssr_ftest"][0]
            p_value = test_result[lag][0]["ssr_ftest"][1]
            significant = p_value < ALPHA

            results[col][lag] = {
                "f_stat": f_stat,
                "p_value": p_value,
                "significant": significant,
            }
            marker = "***" if significant else ""
            print(f"  Lag {lag}: F={f_stat:.3f}, p={p_value:.4f} {marker}")

    return results
