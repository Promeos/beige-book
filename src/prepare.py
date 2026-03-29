"""
Prepare and align Beige Book text data with FRED economic indicators.
"""

import re
import pandas as pd

from src.config import DISTRICT_ALIASES


def prep_beige_data(df):
    """
    Clean and prepare scraped Beige Book data.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Raw long-format DataFrame with columns: date, district, summary.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["summary"] = df["summary"].apply(clean_text)
    df["district"] = df["district"].apply(normalize_district)

    # Drop rows with empty summaries
    df = df[df["summary"].str.len() > 0].reset_index(drop=True)
    return df


def clean_text(text):
    """
    Clean a raw text string from HTML scraping.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML artifacts
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_district(name):
    """
    Normalize a district name to its canonical short form.

    Parameters
    ----------
    name : str

    Returns
    -------
    str
    """
    if not isinstance(name, str):
        return name
    for alias, canonical in DISTRICT_ALIASES.items():
        if alias in name or canonical in name:
            return canonical
    return name.strip()


def align_time_periods(beige_df, fred_df):
    """
    Align Beige Book publication dates with the next available
    economic indicator readings using a forward merge.

    This creates the lead structure needed for predictive analysis:
    sentiment at time T maps to indicators at T+1.

    Parameters
    ----------
    beige_df : pandas.core.frame.DataFrame
        Beige Book data with at least columns: date, district, vader_compound.
    fred_df : pandas.core.frame.DataFrame
        FRED indicators with columns: date, GDPC1, UNRATE, CPIAUCSL, SP500.

    Returns
    -------
    merged : pandas.core.frame.DataFrame
        Combined DataFrame with sentiment and forward-looking indicators.
    """
    beige_df = beige_df.copy()
    fred_df = fred_df.copy()

    beige_df["date"] = pd.to_datetime(beige_df["date"])
    fred_df["date"] = pd.to_datetime(fred_df["date"])

    # Sort both by date for merge_asof
    beige_df = beige_df.sort_values("date").reset_index(drop=True)
    fred_df = fred_df.sort_values("date").reset_index(drop=True)

    merged = beige_df.copy()

    # Align each indicator to its own non-null release calendar.
    # A single wide merge can map Beige Book dates to a row where one
    # indicator is present but another is missing (for example quarterly GDP
    # on monthly FRED rows), which breaks the "next available observation"
    # logic described in the README.
    indicator_cols = [col for col in fred_df.columns if col != "date"]
    for col in indicator_cols:
        indicator = fred_df[["date", col]].dropna().sort_values("date")
        if indicator.empty:
            merged[col] = pd.NA
            continue

        aligned = pd.merge_asof(
            beige_df[["date"]].sort_values("date"),
            indicator,
            on="date",
            direction="forward",
        )
        merged[col] = aligned[col].values

    return merged


def compute_national_aggregate(df):
    """
    Compute national-level sentiment aggregates per report date.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Long-format DataFrame with columns: date, district, vader_compound.

    Returns
    -------
    agg : pandas.core.frame.DataFrame
        One row per date with mean, std, min, max sentiment across districts.
    """
    agg = (
        df.groupby("date")["vader_compound"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    agg.columns = [
        "date",
        "sentiment_mean",
        "sentiment_std",
        "sentiment_min",
        "sentiment_max",
    ]
    return agg


def align_regional_data(beige_df, regional_fred_df):
    """
    Align district-level Beige Book sentiment with regional economic indicators.

    For each (date, district) pair, find the next available coincident index reading.

    Parameters
    ----------
    beige_df : pandas.core.frame.DataFrame
        Must have columns: date, district, vader_compound.
    regional_fred_df : pandas.core.frame.DataFrame
        Must have columns: date, district, coincident_index.

    Returns
    -------
    merged : pandas.core.frame.DataFrame
        District-level data with sentiment and forward-looking regional indicator.
    """
    beige_df = beige_df.copy()
    regional_fred_df = regional_fred_df.copy()

    beige_df["date"] = pd.to_datetime(beige_df["date"])
    regional_fred_df["date"] = pd.to_datetime(regional_fred_df["date"])

    # Merge per district using forward-looking merge
    merged_parts = []
    for district in beige_df["district"].unique():
        beige_dist = beige_df[beige_df["district"] == district].sort_values("date")
        fred_dist = regional_fred_df[
            regional_fred_df["district"] == district
        ].sort_values("date")

        if fred_dist.empty:
            continue

        merged = pd.merge_asof(
            beige_dist,
            fred_dist[["date", "coincident_index"]],
            on="date",
            direction="forward",
        )
        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame()

    return pd.concat(merged_parts, ignore_index=True)


def compute_sector_national_aggregate(sector_df):
    """
    Compute national-level sentiment aggregates per (date, sector).

    Parameters
    ----------
    sector_df : pandas.core.frame.DataFrame
        Long-format with columns: date, district, sector, vader_compound.

    Returns
    -------
    agg : pandas.core.frame.DataFrame
        One row per (date, sector) with mean sentiment across districts.
    """
    agg = (
        sector_df.groupby(["date", "sector"])["vader_compound"]
        .mean()
        .reset_index()
        .rename(columns={"vader_compound": "sentiment_mean"})
    )
    return agg


def align_sector_with_indicators(sector_agg_df, sector_fred_df):
    """
    Align sector-level sentiment with sector-matched FRED indicators.

    For each sector, uses merge_asof(direction='forward') to map sentiment
    at time T to the sector's matched indicator at T+1.

    Parameters
    ----------
    sector_agg_df : pandas.core.frame.DataFrame
        Output of compute_sector_national_aggregate().
        Columns: date, sector, sentiment_mean.
    sector_fred_df : pandas.core.frame.DataFrame
        Wide-format FRED data with date column and one column per series.

    Returns
    -------
    merged : pandas.core.frame.DataFrame
        Long-format: date, sector, sentiment_mean, indicator_value, indicator_id.
    """
    from src.config import SECTOR_FRED_SERIES

    sector_agg_df = sector_agg_df.copy()
    sector_fred_df = sector_fred_df.copy()

    sector_agg_df["date"] = pd.to_datetime(sector_agg_df["date"])
    sector_fred_df["date"] = pd.to_datetime(sector_fred_df["date"])
    sector_fred_df = sector_fred_df.sort_values("date")

    merged_parts = []
    for sector, (series_id, description) in SECTOR_FRED_SERIES.items():
        if series_id not in sector_fred_df.columns:
            continue

        sector_slice = (
            sector_agg_df[sector_agg_df["sector"] == sector].sort_values("date").copy()
        )
        if sector_slice.empty:
            continue

        indicator = sector_fred_df[["date", series_id]].dropna().copy()
        indicator = indicator.rename(columns={series_id: "indicator_value"})

        merged = pd.merge_asof(
            sector_slice,
            indicator,
            on="date",
            direction="forward",
        )
        merged["indicator_id"] = series_id
        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame()

    return pd.concat(merged_parts, ignore_index=True)
