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

    return results


def compute_regional_correlations(
    df, sentiment_col="vader_compound", indicator_col="coincident_index"
):
    """
    Compute per-district correlation between sentiment and a regional indicator.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Must have columns: district, sentiment_col, indicator_col.
    sentiment_col : str
    indicator_col : str

    Returns
    -------
    results : pandas.core.frame.DataFrame
        One row per district with columns: district, correlation, p_value, n_obs.
    """
    rows = []
    for district in sorted(df["district"].unique()):
        subset = df[df["district"] == district][[sentiment_col, indicator_col]].dropna()
        if len(subset) < 10:
            continue

        r, p = pearsonr(subset[sentiment_col], subset[indicator_col])
        rows.append(
            {
                "district": district,
                "correlation": r,
                "p_value": p,
                "n_obs": len(subset),
            }
        )

    results = pd.DataFrame(rows)

    print("\nRegional Correlations: Sentiment vs. Coincident Index")
    print("-" * 60)
    for _, row in results.iterrows():
        marker = "***" if row["p_value"] < ALPHA else ""
        print(
            f"  {row['district']:15s}  r={row['correlation']:+.3f}  "
            f"p={row['p_value']:.4f}  n={row['n_obs']:.0f} {marker}"
        )

    return results


def compute_sector_correlations(
    sector_df,
    regional_df,
    sentiment_col="vader_compound",
    indicator_col="coincident_index",
):
    """
    Compute correlation between each sector's sentiment and regional economic
    activity, aggregated across all districts.

    Parameters
    ----------
    sector_df : pandas.core.frame.DataFrame
        Must have columns: date, district, sector, vader_compound.
    regional_df : pandas.core.frame.DataFrame
        Must have columns: date, district, coincident_index.
    sentiment_col : str
    indicator_col : str

    Returns
    -------
    results : pandas.core.frame.DataFrame
        One row per sector with columns: sector, correlation, p_value, n_obs.
    """
    sector_df = sector_df.copy()
    sector_df["date"] = pd.to_datetime(sector_df["date"])
    regional_df = regional_df.copy()
    regional_df["date"] = pd.to_datetime(regional_df["date"])

    merged = pd.merge_asof(
        sector_df.sort_values("date"),
        regional_df[["date", "district", indicator_col]].sort_values("date"),
        on="date",
        by="district",
        direction="forward",
    )

    rows = []
    for sector in sorted(merged["sector"].unique()):
        subset = merged[merged["sector"] == sector][
            [sentiment_col, indicator_col]
        ].dropna()
        if len(subset) < 10:
            continue

        r, p = pearsonr(subset[sentiment_col], subset[indicator_col])
        rows.append(
            {
                "sector": sector,
                "correlation": r,
                "p_value": p,
                "n_obs": len(subset),
            }
        )

    results = pd.DataFrame(rows).sort_values("correlation", ascending=False)

    print("\nSector Correlations: Sentiment vs. Coincident Index")
    print("-" * 60)
    for _, row in results.iterrows():
        marker = "***" if row["p_value"] < ALPHA else ""
        print(
            f"  {row['sector']:25s}  r={row['correlation']:+.3f}  "
            f"p={row['p_value']:.4f}  n={row['n_obs']:.0f} {marker}"
        )

    return results


def compute_sector_district_correlations(
    sector_df,
    regional_df,
    sentiment_col="vader_compound",
    indicator_col="coincident_index",
):
    """
    Compute correlation for each (sector, district) pair.

    Parameters
    ----------
    sector_df : pandas.core.frame.DataFrame
        Must have columns: date, district, sector, vader_compound.
    regional_df : pandas.core.frame.DataFrame
        Must have columns: date, district, coincident_index.
    sentiment_col : str
    indicator_col : str

    Returns
    -------
    results : pandas.core.frame.DataFrame
        One row per (sector, district) with columns: sector, district,
        correlation, p_value, n_obs.
    """
    sector_df = sector_df.copy()
    sector_df["date"] = pd.to_datetime(sector_df["date"])
    regional_df = regional_df.copy()
    regional_df["date"] = pd.to_datetime(regional_df["date"])

    merged = pd.merge_asof(
        sector_df.sort_values("date"),
        regional_df[["date", "district", indicator_col]].sort_values("date"),
        on="date",
        by="district",
        direction="forward",
    )

    rows = []
    for (sector, district), group in merged.groupby(["sector", "district"]):
        subset = group[[sentiment_col, indicator_col]].dropna()
        if len(subset) < 10:
            continue

        r, p = pearsonr(subset[sentiment_col], subset[indicator_col])
        rows.append(
            {
                "sector": sector,
                "district": district,
                "correlation": r,
                "p_value": p,
                "n_obs": len(subset),
            }
        )

    return pd.DataFrame(rows)


def compute_sector_indicator_correlations(sector_merged_df, max_lag=3):
    """
    Compute lagged correlations between each sector's sentiment and its
    matched FRED indicator.

    Parameters
    ----------
    sector_merged_df : pandas.core.frame.DataFrame
        Output of prepare.align_sector_with_indicators().
        Columns: date, sector, sentiment_mean, indicator_value, indicator_id.
    max_lag : int

    Returns
    -------
    results : pandas.core.frame.DataFrame
        Columns: sector, indicator_id, lag, pearson_r, pearson_p, spearman_r, spearman_p.
    """
    rows = []
    print("\nSector-Indicator Lagged Correlations")
    print("=" * 70)

    for sector in sorted(sector_merged_df["sector"].unique()):
        subset = sector_merged_df[sector_merged_df["sector"] == sector].sort_values(
            "date"
        )
        indicator_id = subset["indicator_id"].iloc[0]

        print(f"\n{sector} → {indicator_id}")
        print("-" * 50)

        for lag in range(max_lag + 1):
            shifted = subset[["sentiment_mean", "indicator_value"]].copy()
            shifted["indicator_value"] = shifted["indicator_value"].shift(-lag)
            clean = shifted.dropna()

            if len(clean) < 10:
                continue

            pr, pp = pearsonr(clean["sentiment_mean"], clean["indicator_value"])
            sr, sp = spearmanr(clean["sentiment_mean"], clean["indicator_value"])

            marker = "***" if pp < ALPHA else ""
            print(
                f"  Lag {lag}: Pearson r={pr:+.3f} (p={pp:.4f})  "
                f"Spearman r={sr:+.3f} (p={sp:.4f}) {marker}"
            )

            rows.append(
                {
                    "sector": sector,
                    "indicator_id": indicator_id,
                    "lag": lag,
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": sp,
                }
            )

    return pd.DataFrame(rows)


def run_sector_granger_tests(sector_merged_df, max_lag=3):
    """
    Run Granger causality tests: does sector sentiment Granger-cause its
    matched economic indicator?

    Parameters
    ----------
    sector_merged_df : pandas.core.frame.DataFrame
        Output of prepare.align_sector_with_indicators().
    max_lag : int

    Returns
    -------
    results : dict
        results[sector][lag] = {f_stat, p_value, significant}.
    """
    results = {}

    print("\nSector Granger Causality Tests")
    print("=" * 70)

    for sector in sorted(sector_merged_df["sector"].unique()):
        subset = sector_merged_df[sector_merged_df["sector"] == sector].sort_values(
            "date"
        )
        indicator_id = subset["indicator_id"].iloc[0]
        pair = subset[["indicator_value", "sentiment_mean"]].dropna()

        if len(pair) < max_lag + 10:
            print(f"\n{sector} → {indicator_id}: insufficient data (n={len(pair)})")
            continue

        print(f"\n{sector} → {indicator_id} (n={len(pair)})")
        print("-" * 50)

        try:
            test_result = grangercausalitytests(
                pair.values, maxlag=max_lag, verbose=False
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue

        results[sector] = {}
        for lag in range(1, max_lag + 1):
            f_stat = test_result[lag][0]["ssr_ftest"][0]
            p_value = test_result[lag][0]["ssr_ftest"][1]
            significant = p_value < ALPHA

            results[sector][lag] = {
                "f_stat": f_stat,
                "p_value": p_value,
                "significant": significant,
            }
            marker = "***" if significant else ""
            print(f"  Lag {lag}: F={f_stat:.3f}, p={p_value:.4f} {marker}")

    return results
