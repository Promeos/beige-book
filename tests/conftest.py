"""
Shared pytest fixtures for Beige Book test suite.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_beige_df():
    """Minimal Beige Book DataFrame in long format."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-18",
                    "2023-01-18",
                    "2023-03-08",
                    "2023-03-08",
                    "2023-05-31",
                    "2023-05-31",
                ]
            ),
            "district": [
                "Boston",
                "New York",
                "Boston",
                "New York",
                "Boston",
                "New York",
            ],
            "summary": [
                "Economic activity expanded modestly in recent weeks.",
                "Growth slowed amid rising uncertainty and weak demand.",
                "Consumer spending declined sharply across the district.",
                "Manufacturing output increased and new orders rose steadily.",
                "Employment conditions tightened with strong hiring activity.",
                "Housing prices fell significantly due to higher mortgage rates.",
            ],
        }
    )


@pytest.fixture
def sample_fred_df():
    """Minimal FRED economic indicator DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-31",
                    "2023-02-28",
                    "2023-03-31",
                    "2023-04-30",
                    "2023-05-31",
                    "2023-06-30",
                ]
            ),
            "GDPC1": [20000.0, 20050.0, 20100.0, 20150.0, 20200.0, 20250.0],
            "UNRATE": [3.4, 3.5, 3.6, 3.5, 3.4, 3.3],
            "CPIAUCSL": [300.0, 301.0, 302.0, 303.0, 304.0, 305.0],
            "SP500": [4000.0, 4050.0, 4100.0, 4150.0, 4200.0, 4250.0],
        }
    )


@pytest.fixture
def sample_beige_with_sentiment():
    """Beige Book DataFrame with VADER sentiment already scored."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-18",
                    "2023-01-18",
                    "2023-03-08",
                    "2023-03-08",
                ]
            ),
            "district": ["Boston", "New York", "Boston", "New York"],
            "summary": [
                "Economic activity expanded modestly.",
                "Growth slowed amid rising uncertainty.",
                "Consumer spending declined sharply.",
                "Manufacturing output increased steadily.",
            ],
            "vader_compound": [0.3182, -0.3612, -0.5574, 0.2960],
        }
    )


@pytest.fixture
def sample_regional_fred_df():
    """Regional FRED data with coincident index per district."""
    dates = pd.to_datetime(
        [
            "2023-01-31",
            "2023-02-28",
            "2023-03-31",
            "2023-04-30",
            "2023-05-31",
            "2023-06-30",
        ]
    )
    rows = []
    for district in ["Boston", "New York"]:
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "district": district,
                    "coincident_index": 100.0 + np.random.randn() * 2,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_national_agg():
    """National aggregate sentiment time series (enough for regression)."""
    np.random.seed(42)
    n = 40
    dates = pd.date_range("2015-01-01", periods=n, freq="QS")
    sentiment = np.random.uniform(-0.5, 0.5, n)
    return pd.DataFrame(
        {
            "date": dates,
            "sentiment_mean": sentiment,
            "GDPC1": 18000 + np.cumsum(np.random.randn(n) * 50),
            "UNRATE": 5.0 + np.cumsum(np.random.randn(n) * 0.1),
            "SP500": 3000 + np.cumsum(np.random.randn(n) * 30),
        }
    )


@pytest.fixture
def sample_reporting_merged_df():
    """Longer national series spanning train/test windows used by reporting."""
    np.random.seed(123)
    n = 132
    dates = pd.date_range("2012-01-01", periods=n, freq="MS")
    sentiment = np.sin(np.arange(n) / 6.0) + np.random.randn(n) * 0.04

    unrate = np.zeros(n)
    cpiaucsl = np.zeros(n)
    gdpc1 = np.zeros(n)
    sp500 = np.zeros(n)

    unrate[0] = 5.0 - 0.4 * sentiment[0]
    cpiaucsl[0] = 250.0 + 0.3 * sentiment[0]
    gdpc1[0] = 18000.0 + 5.0 * sentiment[0]
    sp500[0] = 3000.0 + 20.0 * sentiment[0]

    for t in range(1, n):
        unrate[t] = 0.82 * unrate[t - 1] - 0.55 * sentiment[t] + np.random.randn() * 0.05
        cpiaucsl[t] = (
            cpiaucsl[t - 1]
            + 0.2
            + 0.18 * sentiment[t]
            + np.random.randn() * 0.03
        )
        gdpc1[t] = gdpc1[t - 1] + 8.0 + 1.8 * sentiment[t] + np.random.randn() * 0.6
        sp500[t] = sp500[t - 1] + 4.0 + 9.0 * sentiment[t] + np.random.randn() * 2.0

    return pd.DataFrame(
        {
            "date": dates,
            "sentiment_mean": sentiment,
            "GDPC1": gdpc1,
            "UNRATE": unrate,
            "CPIAUCSL": cpiaucsl,
            "SP500": sp500,
        }
    )


@pytest.fixture
def sample_sector_pipeline_inputs():
    """Deterministic sector/regional inputs for sector and reporting tests."""
    np.random.seed(7)
    n = 48
    dates = pd.date_range("2016-01-01", periods=n, freq="MS")
    districts = ["Boston", "New York"]

    sector_rows = []
    beige_rows = []
    regional_rows = []

    for district in districts:
        offset = 0.15 if district == "Boston" else -0.10
        manufacturing = (
            np.sin(np.arange(n) / 4.0) + offset + np.random.randn(n) * 0.03
        )
        employment = (
            np.cos(np.arange(n) / 5.0) + offset / 2 + np.random.randn(n) * 0.03
        )

        overall_sentiment = (manufacturing + employment) / 2
        coincident_index = (
            100
            + 2.5 * manufacturing
            + 1.8 * employment
            + np.random.randn(n) * 0.05
        )

        for i, date in enumerate(dates):
            sector_rows.append(
                {
                    "date": date,
                    "district": district,
                    "sector": "Manufacturing",
                    "vader_compound": manufacturing[i],
                }
            )
            sector_rows.append(
                {
                    "date": date,
                    "district": district,
                    "sector": "Employment",
                    "vader_compound": employment[i],
                }
            )
            beige_rows.append(
                {
                    "date": date,
                    "district": district,
                    "vader_compound": overall_sentiment[i],
                }
            )
            regional_rows.append(
                {
                    "date": date,
                    "district": district,
                    "coincident_index": coincident_index[i],
                }
            )

    sector_df = pd.DataFrame(sector_rows)
    beige_df = pd.DataFrame(beige_rows)
    regional_df = pd.DataFrame(regional_rows)
    regional_merged_df = beige_df.merge(regional_df, on=["date", "district"])

    sector_mean = (
        sector_df.groupby(["date", "sector"])["vader_compound"]
        .mean()
        .reset_index()
        .rename(columns={"vader_compound": "sentiment_mean"})
    )

    manufacturing_mean = sector_mean[sector_mean["sector"] == "Manufacturing"][
        "sentiment_mean"
    ].to_numpy()
    employment_mean = sector_mean[sector_mean["sector"] == "Employment"][
        "sentiment_mean"
    ].to_numpy()

    ipman = np.zeros(n)
    payems = np.zeros(n)
    ipman[0] = 100.0 + 1.8 * manufacturing_mean[0]
    payems[0] = 200.0 + 2.2 * employment_mean[0]

    for t in range(1, n):
        ipman[t] = (
            0.65 * ipman[t - 1]
            + 1.8 * manufacturing_mean[t]
            + np.random.randn() * 0.05
        )
        payems[t] = (
            0.60 * payems[t - 1]
            + 2.2 * employment_mean[t]
            + np.random.randn() * 0.05
        )

    sector_fred_df = pd.DataFrame({"date": dates, "IPMAN": ipman, "PAYEMS": payems})

    sector_merged_df = sector_mean.copy()
    sector_merged_df["indicator_value"] = np.nan
    sector_merged_df["indicator_id"] = ""
    sector_merged_df.loc[
        sector_merged_df["sector"] == "Manufacturing", "indicator_value"
    ] = ipman
    sector_merged_df.loc[
        sector_merged_df["sector"] == "Manufacturing", "indicator_id"
    ] = "IPMAN"
    sector_merged_df.loc[
        sector_merged_df["sector"] == "Employment", "indicator_value"
    ] = payems
    sector_merged_df.loc[
        sector_merged_df["sector"] == "Employment", "indicator_id"
    ] = "PAYEMS"

    return {
        "sector_df": sector_df,
        "regional_df": regional_df,
        "regional_merged_df": regional_merged_df,
        "sector_fred_df": sector_fred_df,
        "sector_merged_df": sector_merged_df,
    }


@pytest.fixture
def manufacturing_text():
    """Text with clear manufacturing sector keywords."""
    return (
        "Manufacturing activity expanded modestly. "
        "New orders increased and shipments rose. "
        "Factory output was strong across the district."
    )


@pytest.fixture
def mixed_sector_text():
    """Text spanning multiple sectors."""
    return (
        "Manufacturing production expanded modestly in recent weeks. "
        "Retail sales declined slightly compared to last quarter. "
        "Employment conditions remained tight with strong hiring. "
        "Housing prices rose across the district. "
        "Oil drilling activity increased in the region."
    )
