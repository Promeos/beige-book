"""
Visualization functions for Beige Book sentiment analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import DISTRICTS, OUTPUT_DIR


# Style defaults
sns.set_style("whitegrid")
FIGSIZE = (14, 6)
DPI = 300


def plot_sentiment_timeseries(df, save=True):
    """
    Plot national aggregate sentiment over time.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Must have columns: date, sentiment_mean. Optionally sentiment_std.
    save : bool
        Save plot to output directory.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(df["date"], df["sentiment_mean"], color="steelblue", linewidth=1.5)

    if "sentiment_std" in df.columns:
        ax.fill_between(
            df["date"],
            df["sentiment_mean"] - df["sentiment_std"],
            df["sentiment_mean"] + df["sentiment_std"],
            alpha=0.2,
            color="steelblue",
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("VADER Compound Score")
    ax.set_title("Beige Book National Sentiment Over Time")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        _save_fig(fig, "sentiment_timeseries.png")
    return fig


def plot_regional_comparison(df, save=True):
    """
    Heatmap of sentiment by district over time.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Long-format with columns: date, district, vader_compound.
    save : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    pivot = df.pivot_table(
        index="district", columns="date", values="vader_compound", aggfunc="mean"
    )
    # Reorder districts to match canonical order
    ordered = [d for d in DISTRICTS if d in pivot.index]
    pivot = pivot.loc[ordered]

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=0,
        ax=ax,
        xticklabels=10,
        cbar_kws={"label": "Sentiment Score"},
    )
    ax.set_title("Beige Book Sentiment by Federal Reserve District")
    ax.set_xlabel("Report Date")
    ax.set_ylabel("")
    plt.tight_layout()

    if save:
        _save_fig(fig, "regional_comparison.png")
    return fig


def plot_sentiment_vs_indicator(df, indicator_col, indicator_label=None, save=True):
    """
    Dual-axis time series of sentiment vs. an economic indicator.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Must have columns: date, sentiment_mean, and the indicator column.
    indicator_col : str
        Column name of the economic indicator.
    indicator_label : str
        Display label for the indicator axis.
    save : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    indicator_label = indicator_label or indicator_col

    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # Sentiment on left axis
    color1 = "steelblue"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sentiment (VADER Compound)", color=color1)
    ax1.plot(df["date"], df["sentiment_mean"], color=color1, linewidth=1.5)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Indicator on right axis
    ax2 = ax1.twinx()
    color2 = "firebrick"
    ax2.set_ylabel(indicator_label, color=color2)
    ax2.plot(df["date"], df[indicator_col], color=color2, linewidth=1.5, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"Beige Book Sentiment vs. {indicator_label}")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    fig.tight_layout()

    if save:
        _save_fig(fig, f"sentiment_vs_{indicator_col.lower()}.png")
    return fig


def plot_correlation_matrix(corr_df, save=True):
    """
    Heatmap of correlation between sentiment and indicators at various lags.

    Parameters
    ----------
    corr_df : pandas.core.frame.DataFrame
        Correlation matrix (e.g., from hypothesis.compute_lagged_correlations).
    save : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Sentiment-Indicator Correlations at Various Lags")
    plt.tight_layout()

    if save:
        _save_fig(fig, "correlation_matrix.png")
    return fig


def _save_fig(fig, filename):
    """
    Save a matplotlib figure to the output directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Output filename (e.g., 'sentiment_timeseries.png').
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {path}")
