"""
Sentiment analysis for Beige Book economic summaries using VADER.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_analyzer = SentimentIntensityAnalyzer()


def score_sentiment(text):
    """
    Score sentiment of a text string using VADER.

    Parameters
    ----------
    text : str

    Returns
    -------
    scores : dict
        Keys: compound, pos, neg, neu (all floats).
    """
    if not isinstance(text, str) or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    return _analyzer.polarity_scores(text)


def add_sentiment_scores(df, text_col="summary"):
    """
    Add VADER sentiment scores to a DataFrame.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Must contain a text column.
    text_col : str
        Name of the column containing text to score.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Original DataFrame with added columns:
        vader_compound, vader_pos, vader_neg, vader_neu.
    """
    df = df.copy()
    scores = df[text_col].apply(score_sentiment).apply(pd.Series)
    df["vader_compound"] = scores["compound"]
    df["vader_pos"] = scores["pos"]
    df["vader_neg"] = scores["neg"]
    df["vader_neu"] = scores["neu"]
    return df
