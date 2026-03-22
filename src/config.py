"""
Configuration constants, file paths, and API keys for the Beige Book project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
OUTPUT_DIR = PROJECT_ROOT / "output"

# FRED API
FRED_API_KEY = os.getenv("FRED_API_KEY")

# FRED series to pull
FRED_SERIES = {
    "GDPC1": "Real GDP (quarterly)",
    "UNRATE": "Unemployment Rate (monthly)",
    "CPIAUCSL": "CPI (monthly)",
    "SP500": "S&P 500 (daily)",
}

# Beige Book scraping
BASE_URL = "https://www.federalreserve.gov"
INDEX_URL = BASE_URL + "/monetarypolicy/beigebook{year}.htm"
SCRAPE_DELAY = 5  # seconds between requests
REQUEST_HEADERS = {"User-Agent": "BeigeBookResearch/1.0"}
START_YEAR = 2011
END_YEAR = 2026

# The 12 Federal Reserve district banks (canonical short names)
DISTRICTS = [
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

# District name normalization mapping
# Maps variations found in Beige Book HTML to canonical short names
DISTRICT_ALIASES = {
    "Federal Reserve Bank of Boston": "Boston",
    "Federal Reserve Bank of New York": "New York",
    "Federal Reserve Bank of Philadelphia": "Philadelphia",
    "Federal Reserve Bank of Cleveland": "Cleveland",
    "Federal Reserve Bank of Richmond": "Richmond",
    "Federal Reserve Bank of Atlanta": "Atlanta",
    "Federal Reserve Bank of Chicago": "Chicago",
    "Federal Reserve Bank of St. Louis": "St. Louis",
    "Federal Reserve Bank of Minneapolis": "Minneapolis",
    "Federal Reserve Bank of Kansas City": "Kansas City",
    "Federal Reserve Bank of Dallas": "Dallas",
    "Federal Reserve Bank of San Francisco": "San Francisco",
}

# Regional FRED series — State Coincident Economic Activity Index
# (Philadelphia Fed, monthly, index 2007=100)
# Maps each district to its primary state's coincident index
REGIONAL_FRED_SERIES = {
    "Boston": "MAPHCI",
    "New York": "NYPHCI",
    "Philadelphia": "PAPHCI",
    "Cleveland": "OHPHCI",
    "Richmond": "VAPHCI",
    "Atlanta": "GAPHCI",
    "Chicago": "ILPHCI",
    "St. Louis": "MOPHCI",
    "Minneapolis": "MNPHCI",
    "Kansas City": "KSPHCI",
    "Dallas": "TXPHCI",
    "San Francisco": "CAPHCI",
}

# Federal Reserve district -> state FIPS code mapping
# Maps each district to the states it covers (using 2-letter postal codes)
# Based on the official Federal Reserve district boundaries
DISTRICT_STATES = {
    "Boston": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "New York": ["NY", "NJ"],
    "Philadelphia": ["PA", "DE"],
    "Cleveland": ["OH", "WV", "KY"],
    "Richmond": ["VA", "MD", "DC", "NC", "SC"],
    "Atlanta": ["GA", "FL", "AL", "TN", "LA", "MS"],
    "Chicago": ["IL", "IN", "IA", "MI", "WI"],
    "St. Louis": ["MO", "AR"],
    "Minneapolis": ["MN", "MT", "ND", "SD", "WI"],
    "Kansas City": ["KS", "NE", "OK", "CO", "WY", "NM"],
    "Dallas": ["TX"],
    "San Francisco": ["CA", "OR", "WA", "NV", "UT", "AZ", "ID", "HI", "AK"],
}

# Reverse mapping: state -> district (for states split between districts,
# the first listed district takes priority)
STATE_TO_DISTRICT = {}
for _district, _states in DISTRICT_STATES.items():
    for _state in _states:
        if _state not in STATE_TO_DISTRICT:
            STATE_TO_DISTRICT[_state] = _district

# Sentiment analysis
CONFIDENCE_INTERVAL = 0.95
ALPHA = 0.05
