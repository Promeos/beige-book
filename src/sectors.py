"""
Sector extraction from Beige Book district summaries.

Splits each district's summary text into economic sectors (manufacturing,
real estate, employment, etc.) using keyword-based sentence classification.
"""

import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Sector definitions: name -> list of keyword patterns
# Each sentence is assigned to the sector with the most keyword matches.
SECTOR_KEYWORDS = {
    "Manufacturing": [
        r"\bmanufactur\w*\b",
        r"\bfactor(?:y|ies)\b",
        r"\bshipments?\b",
        r"\bnew orders?\b",
        r"\bproduction\b",
        r"\bindustrial\b",
        r"\binventory\b",
        r"\binventories\b",
        r"\bbacklog\b",
        r"\blead times?\b",
    ],
    "Consumer Spending": [
        r"\bretail\w*\b",
        r"\bconsumer\s+spend\w*\b",
        r"\bconsumer\s+demand\b",
        r"\bsame[- ]store\b",
        r"\bapparel\b",
        r"\bshopping\b",
        r"\bdiscretionary\b",
        r"\bauto\s+(sale|dealer)\w*\b",
        r"\bvehicle\s+sale\w*\b",
        r"\bgrocery\b",
        r"\brestaurant\w*\b",
        r"\bdining\b",
    ],
    "Real Estate": [
        r"\breal\s+estate\b",
        r"\bhousing\b",
        r"\bhome\s+(sale|price|build)\w*\b",
        r"\bresidential\b",
        r"\bcommercial\s+(real|property|space|lease|rent)\w*\b",
        r"\bmortgage\b",
        r"\bforeclosure\b",
        r"\bvacancy\b",
        r"\brental\b",
        r"\boffice\s+space\b",
        r"\bwarehousing\b",
    ],
    "Employment": [
        r"\bemploy\w*\b",
        r"\blabor\s+market\w*\b",
        r"\bhir(?:e|ing|ed)\b",
        r"\bjob\s+(open|post|gain|loss|growth)\w*\b",
        r"\bwork(?:er|force)\w*\b",
        r"\blayoff\w*\b",
        r"\bunemploy\w*\b",
        r"\bstaff(?:ing)?\b",
        r"\bretention\b",
        r"\bturnover\b",
        r"\bwage\w*\b",
        r"\bsalar(?:y|ies)\b",
        r"\bcompensation\b",
    ],
    "Financial Services": [
        r"\bbank(?:s|ing)?\b",
        r"\bloan\s+(demand|volume|growth|origination)\w*\b",
        r"\bcredit\b",
        r"\bdeposit\w*\b",
        r"\blending\b",
        r"\bdelinquenc\w*\b",
        r"\bfinancial\s+(service|sector|institution|condition)\w*\b",
        r"\binterest\s+rate\w*\b",
    ],
    "Construction": [
        r"\bconstruction\b",
        r"\bbuilding\s+(permit|activit)\w*\b",
        r"\bcontractor\w*\b",
        r"\binfrastructure\b",
        r"\bhomebuilder\w*\b",
        r"\bhousing\s+start\w*\b",
    ],
    "Agriculture": [
        r"\bagricult\w*\b",
        r"\bfarm(?:s|er|ing)?\b",
        r"\bcrop\w*\b",
        r"\blivestock\b",
        r"\bharvest\w*\b",
        r"\bcattle\b",
        r"\bdairy\b",
        r"\bcorn\b",
        r"\bsoybean\w*\b",
        r"\bwheat\b",
        r"\bcotton\b",
        r"\bpoultry\b",
        r"\bdrought\b",
    ],
    "Energy": [
        r"\benergy\b",
        r"\boil\b",
        r"\bnatural\s+gas\b",
        r"\bdrilling\b",
        r"\brig\s+count\w*\b",
        r"\bpetroleum\b",
        r"\bmining\b",
        r"\bcoal\b",
        r"\brenewable\w*\b",
        r"\bpipeline\b",
        r"\brefin(?:er|ing)\w*\b",
    ],
    "Tourism & Hospitality": [
        r"\btouris\w*\b",
        r"\bhospitality\b",
        r"\bhotel\w*\b",
        r"\blodging\b",
        r"\boccupancy\b",
        r"\btravel\b",
        r"\bvisitor\w*\b",
        r"\bconvention\w*\b",
        r"\bcasino\w*\b",
        r"\brecreation\w*\b",
    ],
    "Transportation": [
        r"\btransport\w*\b",
        r"\bfreight\b",
        r"\btrucking\b",
        r"\bshipping\b",
        r"\bport\s+(volume|activit|traffic)\w*\b",
        r"\brailroad\b",
        r"\blogistic\w*\b",
        r"\bsupply\s+chain\w*\b",
    ],
    "Prices": [
        r"\bprice\w*\b",
        r"\binflat\w*\b",
        r"\bcost\s+(increase|pressure|rise|decline|reduc)\w*\b",
        r"\binput\s+cost\w*\b",
        r"\braw\s+material\w*\b",
        r"\bcommodit(?:y|ies)\b",
        r"\bCPI\b",
    ],
}

# Pre-compile patterns for performance
_COMPILED_PATTERNS = {
    sector: [re.compile(p, re.IGNORECASE) for p in patterns]
    for sector, patterns in SECTOR_KEYWORDS.items()
}

_analyzer = SentimentIntensityAnalyzer()


def _split_sentences(text):
    """
    Split text into sentences, handling common abbreviations.

    Parameters
    ----------
    text : str

    Returns
    -------
    sentences : list of str
    """
    # Split on period/exclamation/question followed by space and capital letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in parts if s.strip()]


def _classify_sentence(sentence):
    """
    Classify a sentence into zero or more sectors based on keyword matches.

    Parameters
    ----------
    sentence : str

    Returns
    -------
    sectors : list of str
        Sector names that matched, sorted by number of matches (descending).
        Empty list if no sector matched.
    """
    scores = {}
    for sector, patterns in _COMPILED_PATTERNS.items():
        count = sum(1 for p in patterns if p.search(sentence))
        if count > 0:
            scores[sector] = count

    return sorted(scores, key=scores.get, reverse=True)


def extract_sectors(text):
    """
    Extract sector-level text segments from a Beige Book district summary.

    Each sentence is assigned to its best-matching sector. Sentences that
    match no sector are grouped under "General". Sentences matching multiple
    sectors are assigned to the one with the most keyword hits.

    Parameters
    ----------
    text : str
        Full district summary text.

    Returns
    -------
    sector_texts : dict
        Maps sector name -> concatenated sentences belonging to that sector.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    sentences = _split_sentences(text)
    sector_texts = {}

    for sentence in sentences:
        sectors = _classify_sentence(sentence)
        # Assign to top-matching sector, or "General" if none matched
        sector = sectors[0] if sectors else "General"

        if sector not in sector_texts:
            sector_texts[sector] = []
        sector_texts[sector].append(sentence)

    return {sector: " ".join(sents) for sector, sents in sector_texts.items()}


def score_sectors(text):
    """
    Extract sectors and score sentiment for each.

    Parameters
    ----------
    text : str
        Full district summary text.

    Returns
    -------
    sector_scores : list of dict
        Each dict has keys: sector, text, vader_compound, vader_pos,
        vader_neg, vader_neu, sentence_count.
    """
    sector_texts = extract_sectors(text)
    results = []

    for sector, sector_text in sector_texts.items():
        scores = _analyzer.polarity_scores(sector_text)
        results.append({
            "sector": sector,
            "text": sector_text,
            "vader_compound": scores["compound"],
            "vader_pos": scores["pos"],
            "vader_neg": scores["neg"],
            "vader_neu": scores["neu"],
            "sentence_count": len(sector_text.split(". ")),
        })

    return results


def build_sector_dataframe(df, text_col="summary"):
    """
    Expand a Beige Book DataFrame into sector-level rows with sentiment.

    Takes a DataFrame with (date, district, summary) and returns a long-format
    DataFrame with one row per (date, district, sector).

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Must contain columns: date, district, and the text column.
    text_col : str
        Name of the column containing summary text.

    Returns
    -------
    sector_df : pandas.core.frame.DataFrame
        Columns: date, district, sector, text, vader_compound, vader_pos,
        vader_neg, vader_neu, sentence_count.
    """
    rows = []
    for _, row in df.iterrows():
        sector_scores = score_sectors(row[text_col])
        for ss in sector_scores:
            ss["date"] = row["date"]
            ss["district"] = row["district"]
            rows.append(ss)

    sector_df = pd.DataFrame(rows)

    # Reorder columns
    col_order = [
        "date", "district", "sector", "text",
        "vader_compound", "vader_pos", "vader_neg", "vader_neu",
        "sentence_count",
    ]
    sector_df = sector_df[[c for c in col_order if c in sector_df.columns]]
    sector_df["date"] = pd.to_datetime(sector_df["date"])
    return sector_df.sort_values(["date", "district", "sector"]).reset_index(drop=True)
