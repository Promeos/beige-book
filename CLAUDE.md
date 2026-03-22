# Beige Book Sentiment Analysis

## Project Goal
Test whether Federal Reserve Beige Book sentiment has **predictive power** for real economic indicators (GDP, unemployment, CPI, S&P 500). Scrape regional economic summaries from the 12 Fed district banks, score sentiment, and correlate with FRED data.

# Custom Agents                                                                                                                 
Before starting work, check `.claude/agents/` for available specialized agents and consider delegating tasks that match their descriptions. Invoke agents by name when the task fits. 

## Project Structure
```
beige_book/
├── src/
│   ├── config.py      # Constants, paths, API keys, district names
│   ├── acquire.py     # Beige Book scraper + FRED data fetcher
│   ├── prepare.py     # Text cleaning, time alignment, merging
│   ├── sentiment.py   # VADER sentiment scoring
│   ├── explore.py     # Visualization functions
│   ├── hypothesis.py  # Statistical tests (correlation, Granger)
│   ├── model.py       # OLS regression, out-of-sample testing
│   ├── sectors.py     # Sector extraction from district summaries
│   └── maps.py        # Interactive choropleth maps (Plotly)
├── data/              # Scraped data + FRED CSVs (gitignored)
│   └── raw_html/      # Cached HTML pages
├── output/            # Generated plots and results
├── run_pipeline.py    # End-to-end pipeline runner
├── MVP.ipynb          # Original prototype (reference only)
└── .env               # FRED_API_KEY (gitignored)
```

## Code Conventions
- **Docstrings**: Triple-quoted with `Parameters` and `Returns` sections
- **No type hints** — keep it simple
- **Pandas-centric** — DataFrames are the primary data structure
- **Long format** for analysis data: one row per (date, district, text, sentiment)
- Use `src/config.py` for all constants and configuration

## Key Technical Decisions
- **Sentiment**: VADER (better for economic text than TextBlob)
- **Economic data**: FRED API via `fredapi` library
- **Predictive tests**: Granger causality + OLS regression with lead-lag structure
- **Time alignment**: `pd.merge_asof(direction='forward')` — Beige Book at time T maps to indicator at T+1
- **Scraping**: Start with 2011-present (consistent HTML), extend backward later

## Workflow Preferences
- **Show don't tell** — prefer working code over lengthy explanations
- **Iterative** — get something working first, then refine
- **Keep it simple** — avoid over-engineering; minimum complexity for the task

## Running the Project
```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Environment
- Python 3.x
- FRED API key required in `.env` file
