"""
Sentiment analysis for Beige Book economic summaries.

Supports two models:
- VADER: Fast, rule-based, general-purpose
- FinBERT: Transformer-based, trained on financial text (more accurate for economic language)
"""
import logging

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()

# Lazy-loaded FinBERT pipeline (loaded on first use)
_finbert_pipeline = None


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


def score_finbert_sentence_level(text):
    """
    Score sentiment using FinBERT at the sentence level (Cleveland Fed approach).

    Splits text into sentences, classifies each as positive/neutral/negative,
    then computes a tone score:
        tone = (n_positive - n_negative) / (n_positive + n_negative)

    This avoids FinBERT's 512-token limit and captures mixed signals
    (e.g., "manufacturing expanded but spending weakened").

    Parameters
    ----------
    text : str

    Returns
    -------
    scores : dict
        Keys: finbert_score (-1 to +1), finbert_positive (count fraction),
        finbert_negative (count fraction), finbert_neutral (count fraction),
        finbert_n_sentences (int).
    """
    import re
    global _finbert_pipeline

    empty = {
        "finbert_score": 0.0,
        "finbert_positive": 0.0,
        "finbert_negative": 0.0,
        "finbert_neutral": 0.0,
        "finbert_n_sentences": 0,
    }

    if not isinstance(text, str) or not text.strip():
        return empty

    if _finbert_pipeline is None:
        _finbert_pipeline = _load_finbert()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return empty

    # Classify each sentence
    n_pos = 0
    n_neg = 0
    n_neu = 0

    for sent in sentences:
        result = _finbert_pipeline(sent, truncation=True, max_length=512)[0]
        label = result["label"].lower()
        if label == "positive":
            n_pos += 1
        elif label == "negative":
            n_neg += 1
        else:
            n_neu += 1

    total = n_pos + n_neg + n_neu
    polar = n_pos + n_neg

    # Cleveland Fed tone formula: (pos - neg) / (pos + neg)
    tone = (n_pos - n_neg) / polar if polar > 0 else 0.0

    return {
        "finbert_score": tone,
        "finbert_positive": n_pos / total,
        "finbert_negative": n_neg / total,
        "finbert_neutral": n_neu / total,
        "finbert_n_sentences": total,
    }


def _load_finbert():
    """
    Load the FinBERT sentiment-analysis pipeline.

    Downloads the yiyanghkust/finbert-tone model on first run (~400MB).

    Returns
    -------
    pipe : transformers.Pipeline
        A HuggingFace sentiment-analysis pipeline using FinBERT.
    """
    from transformers import pipeline
    logger.info("Loading FinBERT model (first run downloads ~400MB)...")
    pipe = pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
        tokenizer="yiyanghkust/finbert-tone",
    )
    logger.info("FinBERT loaded.")
    return pipe


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


def build_sentence_detail(df, text_col="summary"):
    """
    Score every sentence in every summary and return a detailed DataFrame.

    One row per sentence with its FinBERT classification. This lets you
    drill into any (date, district) pair and see exactly which sentences
    drove the sentiment score.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Must have columns: date, district, and text_col.
    text_col : str

    Returns
    -------
    detail_df : pandas.core.frame.DataFrame
        Columns: date, district, sentence_idx, sentence, label,
        confidence, finbert_positive, finbert_negative, finbert_neutral.
    """
    import re
    global _finbert_pipeline

    if _finbert_pipeline is None:
        _finbert_pipeline = _load_finbert()

    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        text = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        for j, sent in enumerate(sentences):
            # Get all label scores
            results = _finbert_pipeline(sent, truncation=True, max_length=512, top_k=None)
            label_scores = {r["label"].lower(): r["score"] for r in results}
            top_label = max(label_scores, key=label_scores.get)

            rows.append({
                "date": row["date"],
                "district": row["district"],
                "sentence_idx": j,
                "sentence": sent,
                "label": top_label,
                "confidence": label_scores[top_label],
                "finbert_positive": label_scores.get("positive", 0.0),
                "finbert_negative": label_scores.get("negative", 0.0),
                "finbert_neutral": label_scores.get("neutral", 0.0),
            })

        if (i + 1) % 100 == 0:
            logger.info("  Processed %d/%d summaries (%d sentences so far)",
                        i + 1, total, len(rows))

    detail_df = pd.DataFrame(rows)
    logger.info("Sentence detail complete: %d sentences from %d summaries", len(rows), total)
    return detail_df


def add_finbert_scores(df, text_col="summary"):
    """
    Add FinBERT sentiment scores to a DataFrame.

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
        finbert_score, finbert_positive, finbert_negative, finbert_neutral.
    """
    df = df.copy()
    logger.info("Scoring %d texts with FinBERT (sentence-level)...", len(df))
    scores = df[text_col].apply(score_finbert_sentence_level).apply(pd.Series)
    df["finbert_score"] = scores["finbert_score"]
    df["finbert_positive"] = scores["finbert_positive"]
    df["finbert_negative"] = scores["finbert_negative"]
    df["finbert_neutral"] = scores["finbert_neutral"]
    df["finbert_n_sentences"] = scores["finbert_n_sentences"]
    logger.info("FinBERT scoring complete. %d total sentences scored.",
                df["finbert_n_sentences"].sum())
    return df
