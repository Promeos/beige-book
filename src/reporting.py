"""
Reproducible summary artifacts for the Beige Book analysis pipeline.

This module turns the pipeline's computed datasets into a canonical JSON
and Markdown summary so README claims can be traced back to a single source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from src.config import ALPHA
from src.hypothesis import compute_lagged_correlations
from src.model import run_ols_regression
from src.prepare import compute_sector_national_aggregate, align_sector_with_indicators


DEFAULT_REGIONAL_START = "2011-01-01"
DEFAULT_SPLIT_DATE = "2011-01-01"


def filter_date_range(df, start=None, end=None, date_col="date"):
    """
    Return a copy of rows inside an optional inclusive date window.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    start : str or None
    end : str or None
    date_col : str

    Returns
    -------
    pandas.core.frame.DataFrame
    """
    if df.empty:
        return df.copy()

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])

    if start is not None:
        result = result[result[date_col] >= pd.to_datetime(start)]
    if end is not None:
        result = result[result[date_col] <= pd.to_datetime(end)]

    return result.reset_index(drop=True)


def summarize_national_analysis(merged_df):
    """
    Build reproducible national headline statistics from an aligned dataset.
    """
    indicator_cols = ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]
    corr_df = compute_lagged_correlations(merged_df, indicator_cols=indicator_cols)

    indicators = {}
    for indicator in indicator_cols:
        indicator_corr = corr_df[corr_df["indicator"] == indicator].copy()
        if indicator_corr.empty:
            continue

        lag0 = indicator_corr[indicator_corr["lag"] == 0].iloc[0]
        best_abs = indicator_corr.iloc[indicator_corr["pearson_r"].abs().argmax()]
        granger = _run_granger_summary(merged_df, indicator)
        controlled = _controlled_model_summary(merged_df, indicator)
        oos = _out_of_sample_summary(merged_df, indicator)

        indicators[indicator] = {
            "lag0": _row_to_dict(
                lag0, ["lag", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
            ),
            "best_absolute_correlation": _row_to_dict(
                best_abs,
                ["lag", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
            ),
            "granger": granger,
            "controlled_ols": controlled,
            "out_of_sample": oos,
        }

    return {
        "sample": _sample_metadata(merged_df),
        "indicators": indicators,
    }


def summarize_regional_analysis(regional_merged_df, start=DEFAULT_REGIONAL_START):
    """
    Build regional correlation summaries for full and post-2011 samples.
    """
    full_corr = _regional_correlation_frame(regional_merged_df)
    focused = filter_date_range(regional_merged_df, start=start)
    focused_corr = _regional_correlation_frame(focused)

    return {
        "full_available_sample": _regional_summary_block(regional_merged_df, full_corr),
        "post_2011_sample": _regional_summary_block(focused, focused_corr),
    }


def summarize_sector_analysis(
    sector_df,
    sector_fred_df,
    regional_merged_df=None,
    source_name="sector_sentiment.csv",
):
    """
    Summarize the canonical sector predictive panel used in the pipeline.
    """
    sector_national = compute_sector_national_aggregate(sector_df)
    sector_merged = align_sector_with_indicators(sector_national, sector_fred_df)
    indicator_corr = _sector_indicator_correlation_frame(sector_merged)
    indicator_granger = _sector_granger_summary(sector_merged)

    results = {
        "dataset": {
            "source": source_name,
            "description": (
                "Summary-derived, sentence-classified sector panel used for "
                "predictive tests."
            ),
            **_sample_metadata(sector_df),
            "n_rows": int(len(sector_df)),
            "n_sectors": int(sector_df["sector"].nunique()),
        },
        "indicator_correlations": {
            "top_lag0": _records(
                indicator_corr[indicator_corr["lag"] == 0].sort_values(
                    "pearson_r", ascending=False
                ),
                [
                    "sector",
                    "indicator_id",
                    "lag",
                    "pearson_r",
                    "pearson_p",
                    "spearman_r",
                    "spearman_p",
                ],
                limit=10,
            ),
            "granger_significant": indicator_granger,
        },
    }

    if regional_merged_df is not None and not regional_merged_df.empty:
        sector_regional = _sector_regional_correlation_frame(
            sector_df, regional_merged_df
        )
        sector_district = _sector_district_correlation_frame(
            sector_df, regional_merged_df
        )
        corrected = sector_district.copy()
        reject, p_adj, _, _ = multipletests(
            corrected["p_value"], alpha=ALPHA, method="fdr_bh"
        )
        corrected["p_adjusted"] = p_adj
        corrected["significant_fdr"] = reject

        results["regional_correlations"] = {
            "by_sector": _records(
                sector_regional.sort_values("correlation", ascending=False),
                ["sector", "correlation", "p_value", "n_obs"],
            ),
            "sector_district_fdr": {
                "n_tests": int(len(corrected)),
                "n_raw_significant": int((corrected["p_value"] < ALPHA).sum()),
                "n_fdr_significant": int(corrected["significant_fdr"].sum()),
                "top_survivors": _records(
                    corrected[corrected["significant_fdr"]].sort_values(
                        "correlation", ascending=False
                    ),
                    ["sector", "district", "correlation", "p_adjusted", "n_obs"],
                    limit=10,
                ),
            },
        }

    return results


def summarize_robustness(merged_df, split_date=DEFAULT_SPLIT_DATE):
    """
    Summarize the repo's main robustness checks.
    """
    return {
        "adf": _adf_summary(merged_df),
        "differenced": _differenced_summary(merged_df),
        "exclude_covid_oos": _exclude_covid_summary(merged_df),
        "split_sample_unrate": summarize_split_sample_stability(
            merged_df, target_col="UNRATE", split_date=split_date
        ),
    }


def summarize_split_sample_stability(
    merged_df, target_col="UNRATE", split_date=DEFAULT_SPLIT_DATE
):
    """
    Summarize differenced correlation and Granger results pre/post split.
    """
    diff_df = merged_df[["date", "sentiment_mean", target_col]].copy()
    diff_df["date"] = pd.to_datetime(diff_df["date"])
    diff_df["d_sentiment"] = diff_df["sentiment_mean"].diff()
    diff_df["d_target"] = diff_df[target_col].diff()

    blocks = {}
    periods = {
        "pre_split": diff_df["date"] < pd.to_datetime(split_date),
        "post_split": diff_df["date"] >= pd.to_datetime(split_date),
        "full_sample": diff_df["date"].notna(),
    }

    for label, mask in periods.items():
        subset = diff_df.loc[mask, ["d_sentiment", "d_target"]].dropna()
        if len(subset) < 10:
            continue

        corr_r, corr_p = pearsonr(subset["d_sentiment"], subset["d_target"])
        granger = _granger_from_frame(
            subset.rename(
                columns={"d_target": target_col, "d_sentiment": "sentiment_mean"}
            ),
            dependent_col=target_col,
            predictor_col="sentiment_mean",
            max_lag=4,
        )

        blocks[label] = {
            "n_obs": int(len(subset)),
            "diff_correlation": {
                "pearson_r": _clean_number(corr_r),
                "pearson_p": _clean_number(corr_p),
            },
            "granger": granger,
        }

    return {
        "target": target_col,
        "split_date": str(pd.to_datetime(split_date).date()),
        "periods": blocks,
    }


def build_analysis_artifact(
    merged_df,
    regional_merged_df,
    sector_df,
    sector_fred_df,
    source_files=None,
    national=None,
    regional=None,
    sector=None,
    robustness=None,
):
    """
    Build the canonical results dictionary for JSON/Markdown export.

    Parameters
    ----------
    merged_df : pandas.core.frame.DataFrame
    regional_merged_df : pandas.core.frame.DataFrame
    sector_df : pandas.core.frame.DataFrame
    sector_fred_df : pandas.core.frame.DataFrame
    source_files : dict or None
    national : dict or None
        Pre-computed national summary. Recomputed from *merged_df* when None.
    regional : dict or None
        Pre-computed regional summary. Recomputed from *regional_merged_df* when None.
    sector : dict or None
        Pre-computed sector summary. Recomputed from *sector_df* / *sector_fred_df* when None.
    robustness : dict or None
        Pre-computed robustness summary. Recomputed from *merged_df* when None.

    Returns
    -------
    dict
    """
    artifact = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "method_notes": {
            "national_alignment": (
                "Each indicator is aligned independently to the next non-null "
                "release date via pandas.merge_asof(direction='forward')."
            ),
            "regional_scope": (
                "Regional results are reported for both the full available sample "
                "and the post-2011 subset used in the portfolio summary."
            ),
            "sector_scope": (
                "The predictive sector panel uses data/sector_sentiment.csv; "
                "data/beige_book_sectors.csv remains a raw scraped corpus with "
                "heterogeneous section labels."
            ),
        },
        "source_files": source_files or {},
        "national": national
        if national is not None
        else summarize_national_analysis(merged_df),
        "regional": regional
        if regional is not None
        else summarize_regional_analysis(regional_merged_df),
        "sector": sector
        if sector is not None
        else summarize_sector_analysis(
            sector_df,
            sector_fred_df,
            regional_merged_df=filter_date_range(
                regional_merged_df, start=DEFAULT_REGIONAL_START
            ),
        ),
        "robustness": robustness
        if robustness is not None
        else summarize_robustness(merged_df),
    }
    return artifact


def write_analysis_artifact(results, output_dir):
    """
    Write canonical results to JSON and Markdown.

    Returns
    -------
    dict
        Paths written to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "analysis_results.json"
    md_path = output_dir / "analysis_results.md"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_summary(results), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown_summary(results):
    """
    Render a concise Markdown summary from the canonical results artifact.
    """
    lines = [
        "# Analysis Results",
        "",
        "This file is generated by `run_pipeline.py`. Treat it as the canonical",
        "source for published summary numbers in this repository.",
        "",
        "## National",
        "",
    ]

    national = results["national"]
    lines.extend(
        [
            f"- Sample: {national['sample']['start_date']} to {national['sample']['end_date']} "
            f"({national['sample']['n_dates']} report dates)",
            "- Lag-0 correlations, Granger significance, controlled OLS p-values, and "
            "out-of-sample RMSE deltas:",
            "",
            "| Indicator | Lag-0 r | Lag-0 p | Granger Lags | Controlled p | OOS RMSE Delta |",
            "|-----------|--------:|--------:|--------------|-------------:|---------------:|",
        ]
    )
    for indicator, values in national["indicators"].items():
        granger_lags = (
            ", ".join(str(l) for l in values["granger"]["significant_lags"]) or "--"
        )
        controlled_p = values["controlled_ols"]["sentiment_p_value"]
        oos_delta = values["out_of_sample"]["rmse_delta"]
        lines.append(
            f"| {indicator} | {values['lag0']['pearson_r']:.3f} | "
            f"{values['lag0']['pearson_p']:.4f} | {granger_lags} | "
            f"{_fmt_optional(controlled_p, '.4f')} | {_fmt_optional(oos_delta, '.4f')} |"
        )

    regional = results["regional"]
    lines.extend(
        [
            "",
            "## Regional",
            "",
            f"- Full available sample top district: "
            f"{regional['full_available_sample']['top_districts'][0]['district']} "
            f"({regional['full_available_sample']['top_districts'][0]['correlation']:.3f})",
            f"- Post-2011 top district: "
            f"{regional['post_2011_sample']['top_districts'][0]['district']} "
            f"({regional['post_2011_sample']['top_districts'][0]['correlation']:.3f})",
            "",
            "## Sector",
            "",
        ]
    )

    sector = results["sector"]
    lines.extend(
        [
            f"- Dataset: `{sector['dataset']['source']}`",
            f"- Sample: {sector['dataset']['start_date']} to {sector['dataset']['end_date']} "
            f"({sector['dataset']['n_dates']} dates, {sector['dataset']['n_rows']} rows)",
            "",
            "| Sector | Indicator | Lag-0 r | Lag-0 p |",
            "|--------|-----------|--------:|--------:|",
        ]
    )
    for row in sector["indicator_correlations"]["top_lag0"][:5]:
        lines.append(
            f"| {row['sector']} | {row['indicator_id']} | {row['pearson_r']:.3f} | "
            f"{row['pearson_p']:.4f} |"
        )

    robustness = results["robustness"]
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            f"- Exclude-COVID UNRATE RMSE delta: "
            f"{robustness['exclude_covid_oos']['UNRATE']['rmse_delta']:.4f}",
            f"- Split-sample UNRATE post-2011 differenced r: "
            f"{robustness['split_sample_unrate']['periods']['post_split']['diff_correlation']['pearson_r']:.3f}",
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def _sample_metadata(df):
    if df.empty:
        return {"start_date": None, "end_date": None, "n_dates": 0}
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    return {
        "start_date": str(data["date"].min().date()),
        "end_date": str(data["date"].max().date()),
        "n_dates": int(data["date"].nunique()),
    }


def _row_to_dict(row, keys):
    return {key: _clean_number(row[key]) for key in keys}


def _records(df, columns, limit=None):
    if limit is not None:
        df = df.head(limit)
    return [
        {col: _clean_number(row[col]) for col in columns} for _, row in df.iterrows()
    ]


def _clean_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _fmt_optional(value, fmt):
    if value is None:
        return "--"
    return format(value, fmt)


def _run_granger_summary(df, indicator, max_lag=4):
    frame = df[["sentiment_mean", indicator]].copy().dropna()
    frame = frame.rename(columns={indicator: "indicator_value"})
    return _granger_from_frame(
        frame,
        dependent_col="indicator_value",
        predictor_col="sentiment_mean",
        max_lag=max_lag,
    )


def _granger_from_frame(frame, dependent_col, predictor_col, max_lag=4):
    if len(frame) < max_lag + 10:
        return {
            "n_obs": int(len(frame)),
            "significant_lags": [],
            "best_p_value": None,
        }

    test_result = grangercausalitytests(
        frame[[dependent_col, predictor_col]].values, maxlag=max_lag, verbose=False
    )
    lag_rows = []
    for lag in range(1, max_lag + 1):
        f_stat, p_value = test_result[lag][0]["ssr_ftest"][:2]
        lag_rows.append(
            {
                "lag": lag,
                "f_stat": _clean_number(f_stat),
                "p_value": _clean_number(p_value),
                "significant": bool(p_value < ALPHA),
            }
        )

    significant_lags = [row["lag"] for row in lag_rows if row["significant"]]
    best_p = min(row["p_value"] for row in lag_rows) if lag_rows else None
    return {
        "n_obs": int(len(frame)),
        "lags": lag_rows,
        "significant_lags": significant_lags,
        "best_p_value": _clean_number(best_p),
    }


def _controlled_model_summary(merged_df, indicator):
    df = merged_df.copy()
    lagged_col = f"{indicator}_lag1"
    df[lagged_col] = df[indicator].shift(1)
    model = run_ols_regression(
        df, indicator, sentiment_col="sentiment_mean", controls=[lagged_col]
    )
    if model is None:
        return {"n_obs": 0, "sentiment_coef": None, "sentiment_p_value": None}
    return {
        "n_obs": int(len(model.model.endog)),
        "sentiment_coef": _clean_number(model.params.get("sentiment_mean")),
        "sentiment_p_value": _clean_number(model.pvalues.get("sentiment_mean")),
        "rsquared": _clean_number(model.rsquared),
    }


def _out_of_sample_summary(merged_df, indicator, train_end="2018-12-31"):
    df = merged_df.copy()
    lagged_col = f"{indicator}_lag1"
    df[lagged_col] = df[indicator].shift(1)

    data = df[["date", indicator, "sentiment_mean", lagged_col]].dropna()
    train = data[data["date"] <= train_end]
    test = data[data["date"] > train_end]
    if len(train) < 20 or len(test) < 5:
        return {
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "baseline_rmse": None,
            "sentiment_rmse": None,
            "rmse_delta": None,
        }

    y_train = train[indicator]
    y_test = test[indicator]

    x_train_base = sm.add_constant(train[[lagged_col]])
    x_test_base = sm.add_constant(test[[lagged_col]])
    baseline = sm.OLS(y_train, x_train_base).fit()
    pred_base = baseline.predict(x_test_base)

    x_train_full = sm.add_constant(train[["sentiment_mean", lagged_col]])
    x_test_full = sm.add_constant(test[["sentiment_mean", lagged_col]])
    full = sm.OLS(y_train, x_train_full).fit()
    pred_full = full.predict(x_test_full)

    baseline_rmse = float(np.sqrt(((y_test - pred_base) ** 2).mean()))
    sentiment_rmse = float(np.sqrt(((y_test - pred_full) ** 2).mean()))

    return {
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "baseline_rmse": baseline_rmse,
        "sentiment_rmse": sentiment_rmse,
        "rmse_delta": baseline_rmse - sentiment_rmse,
    }


def _regional_correlation_frame(regional_merged_df):
    rows = []
    for district in sorted(regional_merged_df["district"].dropna().unique()):
        subset = regional_merged_df[regional_merged_df["district"] == district][
            ["vader_compound", "coincident_index"]
        ].dropna()
        if len(subset) < 10:
            continue
        corr, p_value = pearsonr(subset["vader_compound"], subset["coincident_index"])
        rows.append(
            {
                "district": district,
                "correlation": corr,
                "p_value": p_value,
                "n_obs": len(subset),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )


def _regional_summary_block(df, corr_df):
    eligible = (
        df.dropna(subset=["coincident_index"])
        if "coincident_index" in df.columns
        else df
    )
    return {
        "sample": _sample_metadata(eligible),
        "top_districts": _records(
            corr_df.sort_values("correlation", ascending=False),
            ["district", "correlation", "p_value", "n_obs"],
            limit=5,
        ),
        "all_districts": _records(
            corr_df.sort_values("correlation", ascending=False),
            ["district", "correlation", "p_value", "n_obs"],
        ),
    }


def _sector_indicator_correlation_frame(sector_merged_df, max_lag=3):
    rows = []
    for sector in sorted(sector_merged_df["sector"].dropna().unique()):
        subset = sector_merged_df[sector_merged_df["sector"] == sector].sort_values(
            "date"
        )
        indicator_id = subset["indicator_id"].iloc[0]

        for lag in range(max_lag + 1):
            shifted = subset[["sentiment_mean", "indicator_value"]].copy()
            shifted["indicator_value"] = shifted["indicator_value"].shift(-lag)
            clean = shifted.dropna()
            if len(clean) < 10:
                continue

            pr, pp = pearsonr(clean["sentiment_mean"], clean["indicator_value"])
            sr, spearman_p = spearmanr(
                clean["sentiment_mean"], clean["indicator_value"]
            )

            rows.append(
                {
                    "sector": sector,
                    "indicator_id": indicator_id,
                    "lag": lag,
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": spearman_p,
                }
            )

    return pd.DataFrame(rows)


def _sector_granger_summary(sector_merged_df, max_lag=3):
    rows = []
    for sector in sorted(sector_merged_df["sector"].dropna().unique()):
        subset = sector_merged_df[sector_merged_df["sector"] == sector].sort_values(
            "date"
        )
        indicator_id = subset["indicator_id"].iloc[0]
        granger = _granger_from_frame(
            subset[["indicator_value", "sentiment_mean"]].dropna(),
            dependent_col="indicator_value",
            predictor_col="sentiment_mean",
            max_lag=max_lag,
        )
        if granger["significant_lags"]:
            rows.append(
                {
                    "sector": sector,
                    "indicator_id": indicator_id,
                    "significant_lags": granger["significant_lags"],
                    "best_p_value": granger["best_p_value"],
                    "n_obs": granger["n_obs"],
                }
            )
    return rows


def _sector_regional_correlation_frame(sector_df, regional_merged_df):
    sector = sector_df.copy()
    sector["date"] = pd.to_datetime(sector["date"])
    regional = regional_merged_df.copy()
    regional["date"] = pd.to_datetime(regional["date"])

    merged = pd.merge_asof(
        sector.sort_values("date"),
        regional[["date", "district", "coincident_index"]].sort_values("date"),
        on="date",
        by="district",
        direction="forward",
    )

    rows = []
    for sector_name in sorted(merged["sector"].dropna().unique()):
        subset = merged[merged["sector"] == sector_name][
            ["vader_compound", "coincident_index"]
        ].dropna()
        if len(subset) < 10:
            continue
        corr, p_value = pearsonr(subset["vader_compound"], subset["coincident_index"])
        rows.append(
            {
                "sector": sector_name,
                "correlation": corr,
                "p_value": p_value,
                "n_obs": len(subset),
            }
        )
    return pd.DataFrame(rows)


def _sector_district_correlation_frame(sector_df, regional_merged_df):
    sector = sector_df.copy()
    sector["date"] = pd.to_datetime(sector["date"])
    regional = regional_merged_df.copy()
    regional["date"] = pd.to_datetime(regional["date"])

    merged = pd.merge_asof(
        sector.sort_values("date"),
        regional[["date", "district", "coincident_index"]].sort_values("date"),
        on="date",
        by="district",
        direction="forward",
    )

    rows = []
    for (sector_name, district), group in merged.groupby(["sector", "district"]):
        subset = group[["vader_compound", "coincident_index"]].dropna()
        if len(subset) < 10:
            continue
        corr, p_value = pearsonr(subset["vader_compound"], subset["coincident_index"])
        rows.append(
            {
                "sector": sector_name,
                "district": district,
                "correlation": corr,
                "p_value": p_value,
                "n_obs": len(subset),
            }
        )
    return pd.DataFrame(rows)


def _adf_summary(merged_df):
    rows = []
    for col in ["sentiment_mean", "GDPC1", "UNRATE", "CPIAUCSL", "SP500"]:
        if col not in merged_df.columns:
            continue
        series = merged_df[col].dropna()
        if len(series) < 20:
            continue
        adf_stat, p_value = adfuller(series, autolag="AIC")[:2]
        rows.append(
            {
                "series": col,
                "adf_stat": _clean_number(adf_stat),
                "p_value": _clean_number(p_value),
                "stationary": bool(p_value < ALPHA),
                "n_obs": int(len(series)),
            }
        )
    return rows


def _differenced_summary(merged_df):
    rows = {}
    d_sent = merged_df["sentiment_mean"].diff()

    for col in ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]:
        if col not in merged_df.columns:
            continue
        d_ind = merged_df[col].diff()
        mask = d_sent.notna() & d_ind.notna()
        if mask.sum() < 10:
            continue

        corr_r, corr_p = pearsonr(d_sent[mask], d_ind[mask])
        granger = _granger_from_frame(
            pd.DataFrame({col: d_ind, "sentiment_mean": d_sent}).dropna(),
            dependent_col=col,
            predictor_col="sentiment_mean",
            max_lag=4,
        )
        rows[col] = {
            "lag0": {
                "pearson_r": _clean_number(corr_r),
                "pearson_p": _clean_number(corr_p),
            },
            "granger": granger,
        }

    return rows


def _exclude_covid_summary(merged_df):
    rows = {}
    for col in ["UNRATE", "CPIAUCSL", "SP500"]:
        rows[col] = _exclude_covid_for_indicator(merged_df, col)
    return rows


def _exclude_covid_for_indicator(merged_df, indicator):
    df = merged_df.copy()
    lagged_col = f"{indicator}_lag1"
    df[lagged_col] = df[indicator].shift(1)

    data = df[["date", indicator, "sentiment_mean", lagged_col]].dropna()
    train = data[data["date"] <= "2018-12-31"]
    test = data[data["date"] > "2021-06-30"]
    if len(train) < 20 or len(test) < 5:
        return {
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "baseline_rmse": None,
            "sentiment_rmse": None,
            "rmse_delta": None,
        }

    y_train = train[indicator]
    y_test = test[indicator]

    x_train_base = sm.add_constant(train[[lagged_col]])
    x_test_base = sm.add_constant(test[[lagged_col]])
    baseline = sm.OLS(y_train, x_train_base).fit()
    pred_base = baseline.predict(x_test_base)

    x_train_full = sm.add_constant(train[["sentiment_mean", lagged_col]])
    x_test_full = sm.add_constant(test[["sentiment_mean", lagged_col]])
    full = sm.OLS(y_train, x_train_full).fit()
    pred_full = full.predict(x_test_full)

    baseline_rmse = float(np.sqrt(((y_test - pred_base) ** 2).mean()))
    sentiment_rmse = float(np.sqrt(((y_test - pred_full) ** 2).mean()))

    return {
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "baseline_rmse": baseline_rmse,
        "sentiment_rmse": sentiment_rmse,
        "rmse_delta": baseline_rmse - sentiment_rmse,
    }
