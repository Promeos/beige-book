"""
Acquire Beige Book text data and FRED economic indicators.

Scrapes Federal Reserve Beige Book reports and fetches economic
indicator time series from the FRED API.
"""

import re
import logging
from datetime import datetime
from time import sleep

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scrapy import Selector

from src.config import (
    BASE_URL,
    INDEX_URL,
    SCRAPE_DELAY,
    REQUEST_HEADERS,
    START_YEAR,
    END_YEAR,
    DISTRICT_ALIASES,
    DATA_DIR,
    RAW_HTML_DIR,
    FRED_API_KEY,
    FRED_SERIES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beige Book scraping
# ---------------------------------------------------------------------------


def get_beige_data(use_cache=True):
    """
    Load Beige Book data from cache or scrape fresh.

    Parameters
    ----------
    use_cache : bool
        If True and beige_book.csv exists, read from disk.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Long-format DataFrame with columns: date, district, summary.
    """
    csv_path = DATA_DIR / "beige_book.csv"
    if use_cache and csv_path.exists():
        logger.info("Loading cached Beige Book data from %s", csv_path)
        return pd.read_csv(csv_path, parse_dates=["date"])

    df = scrape_beige_books(START_YEAR, END_YEAR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved Beige Book data to %s (%d rows)", csv_path, len(df))
    return df


def scrape_beige_books(start_year=None, end_year=None):
    """
    Scrape Beige Book economic summaries from the Federal Reserve website.

    Parameters
    ----------
    start_year : int
        First year to scrape (default from config).
    end_year : int
        Last year to scrape (default from config).

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Long-format DataFrame with columns: date, district, summary.
    """
    start_year = start_year or START_YEAR
    end_year = end_year or END_YEAR

    # Step 1: Collect all individual report URLs
    report_urls = _collect_report_urls(start_year, end_year)
    logger.info(
        "Found %d report URLs across %d-%d", len(report_urls), start_year, end_year
    )

    # Step 2: Scrape each report
    rows = []
    for url in report_urls:
        report_date = _parse_date_from_url(url)
        html = _fetch_html(url)
        if html is None:
            continue

        districts = _extract_district_summaries(html)
        if not districts:
            logger.warning("No district summaries found for %s (%s)", report_date, url)
            continue

        if len(districts) != 12:
            logger.warning(
                "Expected 12 districts, got %d for %s", len(districts), report_date
            )

        for district_name, summary_text in districts:
            canonical = _normalize_district(district_name)
            rows.append(
                {
                    "date": report_date,
                    "district": canonical,
                    "summary": summary_text,
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _collect_report_urls(start_year, end_year):
    """
    Fetch annual index pages and extract individual report URLs.

    Parameters
    ----------
    start_year : int
    end_year : int

    Returns
    -------
    urls : list of str
    """
    urls = []
    for year in range(start_year, end_year + 1):
        index_url = INDEX_URL.format(year=year)
        html = _fetch_html(index_url)
        if html is None:
            logger.warning("Could not fetch index for %d", year)
            continue

        sel = Selector(text=html, type="html")
        paths = sel.xpath('//a[contains(., "HTML")]/@href').getall()

        if not paths:
            # Fallback: some years use different link text
            paths = sel.xpath('//a[contains(@href, "beigebook")]/@href').getall()
            # Filter to individual reports (have YYYYMM pattern)
            paths = [p for p in paths if re.search(r"beigebook\d{6}", p)]

        for path in paths:
            full_url = BASE_URL + path if path.startswith("/") else path
            urls.append(full_url)

    return urls


def _extract_district_summaries(html):
    """
    Extract (district_name, summary_text) pairs from a Beige Book report page.

    Handles format variations across years by trying multiple extraction
    strategies:
    - 2024+: Districts in h5 tags, summaries in following p tags (no "Summary of Economic Activity" label)
    - 2011-2023: Districts in h4 tags as "Federal Reserve Bank of X", summaries labeled "Summary of Economic Activity"
    - Older: Various formats handled by BeautifulSoup fallback

    Parameters
    ----------
    html : str
        Raw HTML of a single Beige Book report page.

    Returns
    -------
    results : list of (str, str)
        Pairs of (district_name, summary_text).
    """
    clean_html = html.replace("<br />", "").replace("<br>", "")
    sel = Selector(text=clean_html, type="html")

    # Strategy 1: 2024+ format — district names in h5 tags
    results = _extract_h5_format(sel)
    if len(results) >= 10:
        return results

    # Strategy 2: 2011-2023 format — h4 tags with "Federal Reserve Bank of" names
    results = _extract_h4_format(sel)
    if len(results) >= 10:
        return results

    # Strategy 3: Try h3 headings (some older reports)
    results = _extract_with_heading_tag(sel, "h3")
    if results:
        return results

    # Strategy 4: BeautifulSoup fallback for unusual formats
    results = _extract_bs4_fallback(clean_html)
    return results


def _is_district_name(text):
    """
    Check if text looks like a Federal Reserve district name.

    Parameters
    ----------
    text : str
        Text to check against known district keywords.

    Returns
    -------
    bool
        True if text contains a recognized district keyword.
    """
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
        "Federal Reserve Bank",
    ]
    return any(kw in text for kw in district_keywords)


def _extract_h5_format(sel):
    """
    Extract district summaries from 2024+ format where districts are h5 tags.

    Parameters
    ----------
    sel : scrapy.Selector

    Returns
    -------
    results : list of (str, str)
    """
    district_names = sel.xpath("//h5/text()").getall()
    if not district_names:
        return []

    results = []
    for name in district_names:
        name_clean = name.strip()
        if not name_clean or not _is_district_name(name_clean):
            continue

        escaped_name = name_clean.replace('"', '\\"')
        # Grab the first paragraph after this h5
        first_p = sel.xpath(
            f'.//h5[contains(text(), "{escaped_name}")]/following-sibling::p[1]/text()'
        ).getall()
        summary = " ".join(s.replace("\n", " ").strip() for s in first_p if s.strip())

        if summary:
            results.append((name_clean, summary))

    return results


def _extract_h4_format(sel):
    """
    Extract district summaries from 2011-2023 format where districts
    are h4 tags like "Federal Reserve Bank of Boston".

    Filters out non-district h4 headings (e.g. "Highlights by Federal
    Reserve District", "Labor Markets", etc.).

    Parameters
    ----------
    sel : scrapy.Selector

    Returns
    -------
    results : list of (str, str)
    """
    district_names = sel.xpath("//h4/text()").getall()
    if not district_names:
        return []

    results = []
    seen_districts = set()

    for name in district_names:
        name_clean = name.strip()
        if not name_clean:
            continue

        # Only process headings that look like district names
        if not _is_district_name(name_clean):
            continue

        # Skip section headers like "Highlights by Federal Reserve District"
        if "Highlights" in name_clean or "District" in name_clean:
            continue

        # Deduplicate (some reports have districts in both highlights and detail sections)
        canonical = _normalize_district(name_clean)
        if canonical in seen_districts:
            continue
        seen_districts.add(canonical)

        escaped_name = name_clean.replace('"', '\\"')

        # Try to find "Summary of Economic Activity" paragraph
        summary_nodes = sel.xpath(
            f'.//h4[contains(text(), "{escaped_name}")]'
            '/following-sibling::p[contains(., "Summary of Economic Activity")]'
            "/text()"
        ).getall()

        if summary_nodes:
            summary = " ".join(
                s.replace("\n", " ").strip() for s in summary_nodes if s.strip()
            )
        else:
            # Fallback: grab the first paragraph after this h4
            first_p = sel.xpath(
                f'.//h4[contains(text(), "{escaped_name}")]'
                "/following-sibling::p[1]/text()"
            ).getall()
            summary = " ".join(
                s.replace("\n", " ").strip() for s in first_p if s.strip()
            )

        if summary:
            results.append((name_clean, summary))

    return results


def _extract_with_heading_tag(sel, tag):
    """
    Extract using a specified heading tag (h3, h5, etc.).

    Parameters
    ----------
    sel : scrapy.Selector
    tag : str

    Returns
    -------
    results : list of (str, str)
    """
    district_names = sel.xpath(f"//{tag}/text()").getall()
    if not district_names:
        return []

    results = []
    for name in district_names:
        name_clean = name.strip()
        if not name_clean:
            continue

        # Check if this looks like a district name
        if not any(alias in name_clean for alias in DISTRICT_ALIASES):
            if not any(
                d in name_clean
                for d in [
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
            ):
                continue

        escaped_name = name_clean.replace('"', '\\"')
        first_p = sel.xpath(
            f'.//{tag}[contains(text(), "{escaped_name}")]'
            "/following-sibling::p[1]/text()"
        ).getall()
        summary = " ".join(s.replace("\n", " ").strip() for s in first_p if s.strip())

        if summary:
            results.append((name_clean, summary))

    return results


def _extract_bs4_fallback(html):
    """
    BeautifulSoup fallback for pages with unusual HTML structure.

    Parameters
    ----------
    html : str

    Returns
    -------
    results : list of (str, str)
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # Look for any heading containing a district name
    for tag in soup.find_all(["h2", "h3", "h4", "h5", "strong"]):
        text = tag.get_text(strip=True)
        matched_district = None

        for alias, canonical in DISTRICT_ALIASES.items():
            if alias in text or canonical in text:
                matched_district = text
                break

        if matched_district is None:
            continue

        # Get the next paragraph sibling
        next_p = tag.find_next("p")
        if next_p:
            summary = next_p.get_text(strip=True).replace("\n", " ")
            if summary:
                results.append((matched_district, summary))

    return results


def _normalize_district(name):
    """
    Normalize a district bank name to its canonical short form.

    Parameters
    ----------
    name : str

    Returns
    -------
    str
    """
    for alias, canonical in DISTRICT_ALIASES.items():
        if alias in name or canonical in name:
            return canonical
    # If no alias matches, return cleaned original
    return name.strip()


def _parse_date_from_url(url):
    """
    Extract date from a Beige Book URL like beigebook202001.htm → 2020-01-01.

    Parameters
    ----------
    url : str

    Returns
    -------
    str
        Date string in YYYY-MM-DD format.
    """
    match = re.search(r"beigebook(\d{4})(\d{2})", url)
    if match:
        year, month = match.groups()
        return f"{year}-{month}-01"
    return None


def _fetch_html(url, use_cache=True):
    """
    Fetch HTML from a URL, using cached files when available.

    Parameters
    ----------
    url : str
    use_cache : bool

    Returns
    -------
    html : str or None
    """
    # Build a cache filename from the URL
    cache_name = url.split("/")[-1]
    cache_path = RAW_HTML_DIR / cache_name

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return None

    html = response.text

    # Cache the HTML
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")

    sleep(SCRAPE_DELAY)
    return html


# ---------------------------------------------------------------------------
# FRED economic indicators
# ---------------------------------------------------------------------------


def get_fred_data(use_cache=True):
    """
    Load FRED economic indicators from cache or fetch fresh.

    Parameters
    ----------
    use_cache : bool

    Returns
    -------
    df : pandas.core.frame.DataFrame
        DataFrame indexed by date with columns for each indicator.
    """
    csv_path = DATA_DIR / "fred_indicators.csv"
    if use_cache and csv_path.exists():
        logger.info("Loading cached FRED data from %s", csv_path)
        return pd.read_csv(csv_path, parse_dates=["date"])

    df = fetch_fred_data()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved FRED data to %s", csv_path)
    return df


def fetch_fred_data(start_date="2000-01-01", end_date=None):
    """
    Fetch economic indicator time series from the FRED API.

    Parameters
    ----------
    start_date : str
        Start date for the series.
    end_date : str
        End date (defaults to today).

    Returns
    -------
    df : pandas.core.frame.DataFrame
        DataFrame with date index and one column per indicator series.
    """
    from fredapi import Fred

    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY not found. Add it to your .env file.")

    fred = Fred(api_key=FRED_API_KEY)
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    series_data = {}
    for series_id, description in FRED_SERIES.items():
        logger.info("Fetching %s (%s)", series_id, description)
        try:
            data = fred.get_series(series_id, start_date, end_date)
            series_data[series_id] = data
        except Exception as e:
            logger.error("Failed to fetch %s: %s", series_id, e)

    df = pd.DataFrame(series_data)
    df.index.name = "date"

    # Resample S&P 500 from daily to monthly (end of month)
    if "SP500" in df.columns:
        sp500_monthly = df["SP500"].dropna().resample("ME").last()
        df = df.drop(columns=["SP500"]).resample("ME").last()
        df["SP500"] = sp500_monthly

    df = df.reset_index()
    return df
