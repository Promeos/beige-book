"""
Robustness checks for Beige Book sentiment predictive analysis.

Addresses potential confounds: common trend bias (unit roots, first-differencing),
COVID structural break, Granger stationarity, and multiple testing correction.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.multitest import multipletests

from src.config import ALPHA


def run_adf_tests(merged_df, sentiment_col="sentiment_mean", indicator_cols=None):
    """
    Run Augmented Dickey-Fuller unit root tests on sentiment and indicator series.

    Reports whether each series is stationary (I(0)) or non-stationary (I(1)).

    Parameters
    ----------
    merged_df : pandas.core.frame.DataFrame
    sentiment_col : str
    indicator_cols : list of str

    Returns
    -------
    results : pandas.core.frame.DataFrame
        Columns: series, adf_stat, p_value, stationary, n_obs.
    """
    if indicator_cols is None:
        indicator_cols = ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]

    cols_to_test = [sentiment_col] + [
        c for c in indicator_cols if c in merged_df.columns
    ]

    rows = []
    print("\nAugmented Dickey-Fuller Unit Root Tests")
    print("=" * 60)
    print("  H₀: Series has a unit root (non-stationary)")
    print("  H₁: Series is stationary")
    print("-" * 60)

    for col in cols_to_test:
        series = merged_df[col].dropna()
        if len(series) < 20:
            continue

        result = adfuller(series, autolag="AIC")
        adf_stat, p_value, usedlag, nobs = result[0], result[1], result[2], result[3]
        stationary = p_value < ALPHA

        label = "STATIONARY" if stationary else "NON-STATIONARY"
        marker = "✓" if stationary else "✗"
        print(
            f"  {col:20s}  ADF={adf_stat:+.3f}  p={p_value:.4f}  "
            f"lags={usedlag}  n={nobs}  → {label} {marker}"
        )

        rows.append(
            {
                "series": col,
                "adf_stat": adf_stat,
                "p_value": p_value,
                "stationary": stationary,
                "n_obs": nobs,
            }
        )

    # Also test first-differenced series
    print("\nFirst-Differenced Series:")
    print("-" * 60)
    for col in cols_to_test:
        series = merged_df[col].dropna().diff().dropna()
        if len(series) < 20:
            continue

        result = adfuller(series, autolag="AIC")
        adf_stat, p_value = result[0], result[1]
        stationary = p_value < ALPHA

        label = "STATIONARY" if stationary else "NON-STATIONARY"
        marker = "✓" if stationary else "✗"
        print(f"  Δ{col:19s}  ADF={adf_stat:+.3f}  p={p_value:.4f}  → {label} {marker}")

        rows.append(
            {
                "series": f"Δ{col}",
                "adf_stat": adf_stat,
                "p_value": p_value,
                "stationary": stationary,
                "n_obs": len(series),
            }
        )

    return pd.DataFrame(rows)


def compute_differenced_correlations(
    merged_df, sentiment_col="sentiment_mean", indicator_cols=None, max_lag=3
):
    """
    Compute correlations on first-differenced series to remove common trend bias.

    Parameters
    ----------
    merged_df : pandas.core.frame.DataFrame
    sentiment_col : str
    indicator_cols : list of str
    max_lag : int

    Returns
    -------
    results : pandas.core.frame.DataFrame
        Columns: indicator, lag, pearson_r, pearson_p, spearman_r, spearman_p.
    """
    if indicator_cols is None:
        indicator_cols = ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]

    print("\nFirst-Differenced Lagged Correlations")
    print("=" * 60)
    print("  (Δsentiment vs. Δindicator — removes common trend bias)")
    print("-" * 60)

    rows = []
    for col in indicator_cols:
        if col not in merged_df.columns:
            continue

        # First-difference both series
        d_sent = merged_df[sentiment_col].diff()
        d_ind = merged_df[col].diff()

        print(f"\n  Δsentiment vs. Δ{col}:")
        for lag in range(max_lag + 1):
            shifted = d_ind.shift(-lag)
            mask = d_sent.notna() & shifted.notna()

            if mask.sum() < 10:
                continue

            s = d_sent[mask].values
            i = shifted[mask].values

            pr, pp = pearsonr(s, i)
            sr, sp = spearmanr(s, i)

            marker = "***" if pp < ALPHA else ""
            print(
                f"    Lag {lag}: Pearson r={pr:+.3f} (p={pp:.4f})  "
                f"Spearman r={sr:+.3f} (p={sp:.4f}) {marker}"
            )

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


def run_differenced_granger_tests(
    merged_df, sentiment_col="sentiment_mean", indicator_cols=None, max_lag=4
):
    """
    Run Granger causality tests on first-differenced series.

    First-differencing ensures stationarity, avoiding spurious Granger results
    from I(1) series.

    Parameters
    ----------
    merged_df : pandas.core.frame.DataFrame
    sentiment_col : str
    indicator_cols : list of str
    max_lag : int

    Returns
    -------
    results : dict
        results[indicator][lag] = {f_stat, p_value, significant}.
    """
    if indicator_cols is None:
        indicator_cols = ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]

    print("\nFirst-Differenced Granger Causality Tests")
    print("=" * 60)
    print("  (Tests on Δseries to ensure stationarity)")
    print("-" * 60)

    results = {}
    for col in indicator_cols:
        if col not in merged_df.columns:
            continue

        # First-difference both
        diff_df = pd.DataFrame(
            {
                col: merged_df[col].diff(),
                sentiment_col: merged_df[sentiment_col].diff(),
            }
        ).dropna()

        pair = diff_df[[col, sentiment_col]]

        if len(pair) < max_lag + 10:
            print(f"\n  Δ{col}: insufficient observations ({len(pair)})")
            continue

        print(f"\n  Δ{sentiment_col} → Δ{col}")

        try:
            test_result = grangercausalitytests(
                pair.values, maxlag=max_lag, verbose=False
            )
        except Exception as e:
            print(f"    Error: {e}")
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
            print(f"    Lag {lag}: F={f_stat:.3f}, p={p_value:.4f} {marker}")

    return results


def run_exclude_covid_oos(
    merged_df, indicator_cols=None, sentiment_col="sentiment_mean"
):
    """
    Out-of-sample evaluation excluding COVID period (March 2020 - June 2021).

    Train: 2011-2018. Test: 2021-07 onward (skipping COVID).

    Parameters
    ----------
    merged_df : pandas.core.frame.DataFrame
    indicator_cols : list of str
    sentiment_col : str

    Returns
    -------
    results : dict
        results[indicator] = {baseline: {rmse, mae}, sentiment_model: {rmse, mae}}.
    """
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    if indicator_cols is None:
        indicator_cols = ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]

    print("\nOut-of-Sample Evaluation (Excluding COVID)")
    print("=" * 60)
    print("  Train: ≤ 2018-12-31")
    print("  Test:  > 2021-06-30 (skipping March 2020 - June 2021)")
    print("-" * 60)

    results = {}
    for col in indicator_cols:
        if col not in merged_df.columns:
            continue

        df = merged_df.copy()
        df[f"{col}_lag1"] = df[col].shift(1)
        lagged_col = f"{col}_lag1"

        data = df[["date", col, sentiment_col, lagged_col]].dropna()
        train = data[data["date"] <= "2018-12-31"]
        test = data[data["date"] > "2021-06-30"]

        if len(train) < 20 or len(test) < 5:
            print(
                f"\n  {col}: insufficient data (train={len(train)}, test={len(test)})"
            )
            continue

        y_train = train[col]
        y_test = test[col]

        # Baseline: lagged indicator only
        X_train_base = sm.add_constant(train[[lagged_col]])
        X_test_base = sm.add_constant(test[[lagged_col]])
        baseline = sm.OLS(y_train, X_train_base).fit()
        pred_base = baseline.predict(X_test_base)

        # Full: sentiment + lagged indicator
        X_train_full = sm.add_constant(train[[sentiment_col, lagged_col]])
        X_test_full = sm.add_constant(test[[sentiment_col, lagged_col]])
        full_model = sm.OLS(y_train, X_train_full).fit()
        pred_full = full_model.predict(X_test_full)

        base_rmse = np.sqrt(mean_squared_error(y_test, pred_base))
        full_rmse = np.sqrt(mean_squared_error(y_test, pred_full))
        improvement = base_rmse - full_rmse

        results[col] = {
            "baseline": {
                "rmse": base_rmse,
                "mae": mean_absolute_error(y_test, pred_base),
            },
            "sentiment_model": {
                "rmse": full_rmse,
                "mae": mean_absolute_error(y_test, pred_full),
            },
        }

        marker = "✓" if improvement > 0 else "✗"
        print(f"\n  {col}:")
        print(f"    Baseline RMSE:  {base_rmse:.4f}")
        print(f"    Sentiment RMSE: {full_rmse:.4f}")
        print(f"    Improvement:    {improvement:+.4f} {marker}")

    return results


def apply_fdr_correction(corr_df, p_col="p_value", alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction to a DataFrame of correlations.

    Parameters
    ----------
    corr_df : pandas.core.frame.DataFrame
        Must have a p-value column.
    p_col : str
        Name of the p-value column.
    alpha : float
        FDR threshold.

    Returns
    -------
    corr_df : pandas.core.frame.DataFrame
        Original DataFrame with added columns: p_adjusted, significant_fdr.
    """
    corr_df = corr_df.copy()

    if len(corr_df) == 0:
        corr_df["p_adjusted"] = []
        corr_df["significant_fdr"] = []
        return corr_df

    reject, p_adj, _, _ = multipletests(corr_df[p_col], alpha=alpha, method="fdr_bh")
    corr_df["p_adjusted"] = p_adj
    corr_df["significant_fdr"] = reject

    n_tests = len(corr_df)
    n_raw = (corr_df[p_col] < alpha).sum()
    n_fdr = reject.sum()

    print(f"\n  Benjamini-Hochberg FDR Correction (α={alpha})")
    print(f"  Total tests: {n_tests}")
    print(f"  Significant (uncorrected): {n_raw}")
    print(f"  Significant (FDR-corrected): {n_fdr}")
    print(f"  False discoveries avoided: {n_raw - n_fdr}")

    return corr_df


def run_all_robustness_checks(merged_df):
    """
    Run the full suite of robustness checks on the merged national dataset.

    Parameters
    ----------
    merged_df : pandas.core.frame.DataFrame
        Output of align_time_periods() — national sentiment + FRED indicators.

    Returns
    -------
    results : dict
        Keys: adf, differenced_corr, differenced_granger, exclude_covid.
    """
    print("\n" + "#" * 60)
    print("# ROBUSTNESS CHECKS")
    print("#" * 60)

    # 1. ADF unit root tests
    adf_df = run_adf_tests(merged_df)

    # 2. First-differenced correlations
    diff_corr_df = compute_differenced_correlations(merged_df)

    # 3. First-differenced Granger tests
    diff_granger = run_differenced_granger_tests(merged_df)

    # 4. Exclude-COVID out-of-sample
    exclude_covid = run_exclude_covid_oos(merged_df)

    return {
        "adf": adf_df,
        "differenced_corr": diff_corr_df,
        "differenced_granger": diff_granger,
        "exclude_covid": exclude_covid,
    }


def run_sector_fdr_correction(sector_district_corr_df):
    """
    Apply FDR correction to sector-district correlation results.

    Parameters
    ----------
    sector_district_corr_df : pandas.core.frame.DataFrame
        Output of compute_sector_district_correlations().
        Must have columns: sector, district, correlation, p_value.

    Returns
    -------
    corrected_df : pandas.core.frame.DataFrame
        With added columns: p_adjusted, significant_fdr.
    """
    print("\nSector-District Multiple Testing Correction")
    print("=" * 60)

    corrected = apply_fdr_correction(sector_district_corr_df)

    # Show top results that survive FDR
    survivors = corrected[corrected["significant_fdr"]].sort_values(
        "correlation", ascending=False
    )

    if len(survivors) > 0:
        print("\n  Top sector-district pairs surviving FDR correction:")
        print(f"  {'Sector':25s}  {'District':15s}  {'r':>7s}  {'p_adj':>8s}")
        print("  " + "-" * 60)
        for _, row in survivors.head(10).iterrows():
            print(
                f"  {row['sector']:25s}  {row['district']:15s}  "
                f"{row['correlation']:+.3f}  {row['p_adjusted']:.4f}"
            )
    else:
        print("\n  No sector-district pairs survive FDR correction.")

    return corrected
