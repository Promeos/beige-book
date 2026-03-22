"""
Scrape sector-level paragraphs from cached Beige Book HTML.

Extracts individual sector sections (Manufacturing, Employment, Prices, etc.)
for each district from each report, producing a long-format DataFrame:
    (date, district, sector, text)

This goes beyond the "Summary of Economic Activity" paragraph to capture
the full granular detail of each Beige Book report.
"""
import re
import logging
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.config import DATA_DIR, RAW_HTML_DIR, DISTRICT_ALIASES, START_YEAR, END_YEAR
from src.acquire import _parse_date_from_url, _normalize_district

logger = logging.getLogger(__name__)

# Sector names to skip (not real sectors)
SKIP_SECTORS = {
    "Summary of Economic Activity",
    "Overall Economic Activity",
    "Highlights by Federal Reserve District",
}


def get_sector_data(use_cache=True):
    """
    Load sector-level data from cache or scrape from cached HTML.

    Parameters
    ----------
    use_cache : bool

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Columns: date, district, sector, text.
    """
    csv_path = DATA_DIR / "beige_book_sectors.csv"
    if use_cache and csv_path.exists():
        logger.info("Loading cached sector data from %s", csv_path)
        return pd.read_csv(csv_path, parse_dates=["date"])

    df = scrape_all_sectors()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved sector data to %s (%d rows)", csv_path, len(df))
    return df


def scrape_all_sectors():
    """
    Extract sector-level paragraphs from all cached Beige Book HTML files.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Columns: date, district, sector, text.
    """
    html_files = sorted(RAW_HTML_DIR.glob("beigebook*.htm"))
    # Filter to individual reports (not index pages)
    html_files = [f for f in html_files if re.search(r"beigebook\d{6}", f.name)]

    logger.info("Processing %d cached HTML files for sector data", len(html_files))

    all_rows = []
    for filepath in html_files:
        date = _parse_date_from_url(filepath.name)
        if date is None:
            continue

        html = filepath.read_text(encoding="utf-8")

        if "-summary" in filepath.name:
            # 2024+ format: summary pages only have summary paragraphs, no sectors
            # Skip these — they don't have sector breakdowns
            continue

        rows = _extract_sectors_from_report(html, date)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    logger.info("Extracted %d sector entries from %d reports", len(df),
                df["date"].nunique() if not df.empty else 0)
    return df


def _extract_sectors_from_report(html, date):
    """
    Extract (district, sector, text) tuples from a single Beige Book report.

    Parameters
    ----------
    html : str
    date : str

    Returns
    -------
    rows : list of dict
    """
    clean_html = html.replace("<br />", "").replace("<br>", "")
    soup = BeautifulSoup(clean_html, "html.parser")

    rows = []

    # Find all district headings (h4 tags)
    district_headings = soup.find_all("h4")

    for i, heading in enumerate(district_headings):
        district_name = heading.get_text(strip=True)
        if not _is_district(district_name):
            continue

        canonical = _normalize_district(district_name)

        # Get all <p> tags between this h4 and the next h4
        next_heading = district_headings[i + 1] if i + 1 < len(district_headings) else None
        paragraphs = _get_paragraphs_between(heading, next_heading)

        current_sector = None
        current_text_parts = []

        for p in paragraphs:
            # Check if this paragraph starts a new sector (has a <strong> tag)
            strong = p.find("strong")
            if strong:
                # Save the previous sector
                if current_sector and current_text_parts:
                    text = " ".join(current_text_parts)
                    if current_sector not in SKIP_SECTORS and len(text) > 20:
                        rows.append({
                            "date": date,
                            "district": canonical,
                            "sector": _normalize_sector(current_sector),
                            "text": text,
                        })

                # Start new sector
                current_sector = strong.get_text(strip=True)
                # Get the rest of this paragraph (after the strong tag)
                full_text = p.get_text(strip=True)
                # Remove the sector name from the beginning
                sector_text = full_text[len(current_sector):].strip()
                current_text_parts = [sector_text] if sector_text else []
            else:
                # Continuation paragraph (no strong tag) — append to current sector
                text = p.get_text(strip=True)
                if text and current_sector:
                    # Stop if we hit boilerplate
                    if "For more information" in text or "www." in text:
                        break
                    current_text_parts.append(text)

        # Save the last sector
        if current_sector and current_text_parts:
            text = " ".join(current_text_parts)
            if current_sector not in SKIP_SECTORS and len(text) > 20:
                rows.append({
                    "date": date,
                    "district": canonical,
                    "sector": _normalize_sector(current_sector),
                    "text": text,
                })

    return rows


def _get_paragraphs_between(start_tag, end_tag):
    """Get all <p> tags between two heading tags."""
    paragraphs = []
    current = start_tag.find_next_sibling()
    while current:
        if current == end_tag:
            break
        if current.name == "p":
            paragraphs.append(current)
        elif current.name in ("h4", "h3", "h2"):
            break
        current = current.find_next_sibling()
    return paragraphs


def _is_district(text):
    """Check if text is a district heading."""
    district_keywords = [
        "Boston", "New York", "Philadelphia", "Cleveland", "Richmond",
        "Atlanta", "Chicago", "St. Louis", "Minneapolis", "Kansas City",
        "Dallas", "San Francisco",
    ]
    return any(kw in text for kw in district_keywords)


def _normalize_sector(name):
    """
    Normalize sector names to canonical forms.

    Different districts use slightly different names for the same sector.
    """
    name = name.strip().rstrip(":")

    # Common normalizations
    mappings = {
        "Employment and Wages": "Employment & Wages",
        "Labor Markets": "Employment & Wages",
        "Hiring and Wages": "Employment & Wages",
        "Wages and Prices": "Prices",
        "Prices and Wages": "Prices",
        "Consumer Spending": "Consumer Spending",
        "Retail, Travel, and Tourism": "Consumer Spending",
        "Retail and Tourism": "Consumer Spending",
        "Retail Trade": "Consumer Spending",
        "Real Estate and Construction": "Real Estate & Construction",
        "Construction and Real Estate": "Real Estate & Construction",
        "Residential Real Estate and Construction": "Real Estate & Construction",
        "Commercial Real Estate and Construction": "Real Estate & Construction",
        "Financial Services": "Financial Services",
        "Banking and Finance": "Financial Services",
        "Banking": "Financial Services",
        "Professional and Business Services": "Business Services",
        "Nonfinancial Services": "Business Services",
        "Services": "Business Services",
        "Transportation": "Transportation",
        "Freight": "Transportation",
        "Ports and Transportation": "Transportation",
        "Transportation and Warehousing": "Transportation",
        "Agriculture": "Agriculture",
        "Agriculture and Natural Resources": "Agriculture",
        "Agricultural Conditions": "Agriculture",
        "Natural Resources and Energy": "Energy",
        "Energy": "Energy",
        "Oil and Gas": "Energy",
    }

    return mappings.get(name, name)
