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

import pandas as pd
from bs4 import BeautifulSoup

from src.config import DATA_DIR, RAW_HTML_DIR, BASE_URL, DISTRICTS
from src.acquire import _parse_date_from_url, _normalize_district, _fetch_html

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
            # 2024+ format: summary pages don't have sectors
            # Fetch individual district pages instead
            rows = _scrape_district_pages(filepath.name, date)
            all_rows.extend(rows)
            continue

        rows = _extract_sectors_from_report(html, date)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    logger.info(
        "Extracted %d sector entries from %d reports",
        len(df),
        df["date"].nunique() if not df.empty else 0,
    )
    return df


def _scrape_district_pages(summary_filename, date):
    """
    Fetch and extract sectors from individual district pages (2024+ format).

    For 2024+, each district has its own page:
    beigebook202601-boston.htm, beigebook202601-cleveland.htm, etc.

    Parameters
    ----------
    summary_filename : str
        The summary page filename (e.g., beigebook202601-summary.htm)
    date : str

    Returns
    -------
    rows : list of dict
    """
    # Build district page URLs from the summary filename
    # beigebook202601-summary.htm → beigebook202601-{district}.htm
    base = summary_filename.replace("-summary.htm", "")

    district_slugs = {
        "Boston": "boston",
        "New York": "new-york",
        "Philadelphia": "philadelphia",
        "Cleveland": "cleveland",
        "Richmond": "richmond",
        "Atlanta": "atlanta",
        "Chicago": "chicago",
        "St. Louis": "st-louis",
        "Minneapolis": "minneapolis",
        "Kansas City": "kansas-city",
        "Dallas": "dallas",
        "San Francisco": "san-francisco",
    }

    all_rows = []
    for district, slug in district_slugs.items():
        page_url = f"{BASE_URL}/monetarypolicy/{base}-{slug}.htm"
        html = _fetch_html(page_url)
        if html is None:
            logger.warning("Could not fetch district page for %s (%s)", district, date)
            continue

        rows = _extract_sectors_from_district_page(html, date, district)
        all_rows.extend(rows)

    return all_rows


def _extract_sectors_from_district_page(html, date, district):
    """
    Extract sectors from an individual district page (2024+ format).

    These pages have sectors as <h4> headings with following <p> content.

    Parameters
    ----------
    html : str
    date : str
    district : str

    Returns
    -------
    rows : list of dict
    """
    clean_html = html.replace("<br />", "").replace("<br>", "")
    soup = BeautifulSoup(clean_html, "html.parser")

    rows = []
    # Sectors are h4 headings on individual district pages
    headings = soup.find_all("h4")

    for i, heading in enumerate(headings):
        sector_name = heading.get_text(strip=True)
        if not sector_name or sector_name in SKIP_SECTORS:
            continue
        if _is_district(sector_name):
            continue

        normalized = _normalize_sector(sector_name)
        if not normalized:
            continue

        # Get paragraphs until next h4
        next_h = headings[i + 1] if i + 1 < len(headings) else None
        paragraphs = _get_paragraphs_between(heading, next_h)
        text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        if len(text) > 20:
            rows.append({
                "date": date,
                "district": district,
                "sector": normalized,
                "text": text,
            })

    return rows


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

    # Try multiple heading tag formats:
    # 2017-2023: <h4>Federal Reserve Bank of Boston</h4>
    # 2011-2016: <h2>First District--Boston</h2>
    district_headings = soup.find_all("h4")
    district_headings = [
        h for h in district_headings if _is_district(h.get_text(strip=True))
    ]

    if not district_headings:
        # Try h2 format (2011-2016)
        district_headings = soup.find_all("h2")
        district_headings = [
            h for h in district_headings if _is_district(h.get_text(strip=True))
        ]

    if not district_headings:
        return rows

    for i, heading in enumerate(district_headings):
        district_name = heading.get_text(strip=True)
        if not _is_district(district_name):
            continue

        canonical = _normalize_district(district_name)

        # Get all <p> tags between this heading and the next district heading
        next_heading = (
            district_headings[i + 1] if i + 1 < len(district_headings) else None
        )
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
                    normalized = _normalize_sector(current_sector)
                    if current_sector not in SKIP_SECTORS and normalized and len(text) > 20:
                        rows.append({
                            "date": date,
                            "district": canonical,
                            "sector": normalized,
                            "text": text,
                        })

                # Start new sector
                current_sector = strong.get_text(strip=True)
                # Get the rest of this paragraph (after the strong tag)
                full_text = p.get_text(strip=True)
                # Remove the sector name from the beginning
                sector_text = full_text[len(current_sector) :].strip()
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
            normalized = _normalize_sector(current_sector)
            if current_sector not in SKIP_SECTORS and normalized and len(text) > 20:
                rows.append(
                    {
                        "date": date,
                        "district": canonical,
                        "sector": normalized,
                        "text": text,
                    }
                )

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
        elif current.name in ("h2", "h3", "h4") and current != start_tag:
            # Stop at next heading (but not our own tag level in case of nested structures)
            if _is_district(current.get_text(strip=True)):
                break
        current = current.find_next_sibling()
    return paragraphs


def _is_district(text):
    """Check if text is a district heading (handles both naming conventions)."""
    district_keywords = [
        "Boston",
        "New York",
        "Philadelphia",
        "Cleveland",
        "Richmond",
        "Atlanta",
        "Chicago",
        "St. Louis",
        "Minneapolis",
        "Kansas City",
        "Dallas",
        "San Francisco",
    ]
    # Skip section headers that contain district keywords
    skip_phrases = ["Summary of Commentary", "Highlights by", "Summary of Economic"]
    if any(skip in text for skip in skip_phrases):
        return False
    return any(kw in text for kw in district_keywords)


def _normalize_sector(name):
    """
    Normalize sector names to canonical forms.

    Different districts use slightly different names for the same sector.
    """
    name = name.strip().rstrip(":")

    # Normalize to canonical sector names
    name_lower = name.lower().rstrip(".").strip()

    # Map by keyword matching (order matters — more specific first)
    keyword_map = [
        # Employment & Wages
        (
            ["employment", "labor", "hiring", "staffing", "worker experience"],
            "Employment & Wages",
        ),
        # Prices
        (["price", "cost", "inflation"], "Prices"),
        # Manufacturing
        (["manufactur", "industrial production"], "Manufacturing"),
        # Consumer Spending
        (["consumer", "retail", "tourism", "hospitality"], "Consumer Spending"),
        # Real Estate & Construction
        (
            ["real estate", "construction", "residential", "commercial real"],
            "Real Estate & Construction",
        ),
        # Financial Services
        (["financial", "banking", "finance", "loan", "lending"], "Financial Services"),
        # Business Services
        (
            [
                "business service",
                "professional service",
                "nonfinancial service",
                "non-financial service",
                "software",
                "information technology",
                "it service",
                "selected business",
            ],
            "Business Services",
        ),
        # Transportation
        (["transport", "freight", "port", "shipping", "trucking"], "Transportation"),
        # Agriculture
        (["agricultur", "farm", "crop"], "Agriculture"),
        # Energy
        (["energy", "oil", "gas", "mining", "natural resource"], "Energy"),
        # Community
        (["community", "minority", "women-owned"], "Community"),
    ]

    for keywords, canonical in keyword_map:
        if any(kw in name_lower for kw in keywords):
            return canonical

    # Skip junk entries
    if len(name) <= 2 or name in (".", "across", "ervices"):
        return None

    return name
