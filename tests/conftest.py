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
