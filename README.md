# Does the Federal Reserve's Beige Book Predict the Economy?

A sentiment analysis of regional economic narratives from the 12 Federal Reserve district banks (2011--2026).

## Overview

This project scrapes 1,464 district-level summaries across 122 Beige Book reports (2011--2026), plus 10,728 sector-level paragraphs, scores sentiment using VADER (validated against FinBERT-FOMC and FinBERT-Tone), and tests whether that sentiment has **predictive power** for four economic indicators:

| FRED Series | Indicator | Frequency |
|-------------|-----------|-----------|
| `GDPC1` | Real GDP | Quarterly |
| `UNRATE` | Unemployment Rate | Monthly |
| `CPIAUCSL` | CPI | Monthly |
| `SP500` | S&P 500 | Daily |

Statistical methods: lagged Pearson/Spearman correlations, Granger causality tests, OLS regression with lead-lag structure, and out-of-sample evaluation.

## Key Findings

| Indicator | Granger Causes? | Best Lag | Correlation (r) | OLS Controlled (p) | Out-of-Sample RMSE |
|-----------|----------------|----------|-----------------|---------------------|---------------------|
| **Unemployment** | Yes (lags 3--4) | 3--4 | **-0.59** | **p < 0.001** | -0.0012 improvement |
| **GDP** | Yes (lags 1--2) | 1--2 | 0.15 | Insufficient quarterly data | -- |
| **CPI** | Yes (lags 3--4) | 3--4 | 0.05 | **p = 0.001** | -0.0791 improvement |
| **S&P 500** | No | -- | -0.28 | p = 0.074 | No improvement |

**Bottom line:** Beige Book sentiment has genuine predictive power for **unemployment** and **CPI/inflation**, even after controlling for each indicator's own history. It Granger-causes GDP at short lags. No predictive value for the S&P 500.

The strongest signal is **unemployment** (r = -0.59): when Beige Book narratives turn more positive, unemployment tends to fall in subsequent months. This makes intuitive sense — the district banks are directly surveying businesses about hiring conditions.

### Three-Model Comparison

We tested three sentiment models head-to-head across all 12 districts:

| Model | Type | Training Data | Wins |
|-------|------|---------------|------|
| **VADER** | Rule-based lexicon (7,500 words) | General-purpose | **10/12 districts** |
| **FinBERT-FOMC** | Transformer (fine-tuned) | FOMC meeting minutes | 0/12 |
| **FinBERT-Tone** | Transformer (fine-tuned) | 10K analyst report sentences | 2/12 |

VADER dominates because the Beige Book has a consistently optimistic baseline tone -- most summaries describe "moderate growth" or "steady expansion." VADER's positive bias aligns naturally with this tonal baseline, so deviations from it carry genuine signal.

The transformers only win in two cases:
- **San Francisco** -- FinBERT-Tone (r = +0.48), where the district's persistent pessimism needs a model that reads financial descriptions rather than general tone
- **Chicago** -- FinBERT-Tone (r = +0.23), where manufacturing-heavy language benefits from analyst-report training data

FinBERT-FOMC underperforms everywhere because it reads *policy tone* (hawkish/dovish), not economic descriptions. The Beige Book describes business conditions, not policy intent.

**Takeaway:** Simple beats sophisticated when the domain text has a consistent tonal baseline.

### Regional Analysis

Not all districts are equal. We tested each district's sentiment against its state's [Coincident Economic Activity Index](https://fred.stlouisfed.org/release?rid=109) (Philadelphia Fed):

![Regional Predictive Power by Federal Reserve District](output/regional_correlation_bars.png)

**Cleveland dominates** (r = 0.66) because the district covers Ohio's manufacturing heartland — auto plants (Honda, GM, Ford), steel, and chemicals. Manufacturing is highly cyclical, so when Cleveland's Beige Book turns positive or negative, Ohio's economy follows in lockstep. Diversified economies (Chicago, Dallas, Atlanta) show no significant correlation because competing signals dilute the sentiment.

| District | Correlation | p-value |
|----------|-----------|---------|
| **Cleveland** | **+0.66** | < 0.0001 |
| **Boston** | +0.35 | 0.0001 |
| **St. Louis** | +0.27 | 0.003 |
| **San Francisco** | +0.27 | 0.003 |
| **New York** | +0.21 | 0.022 |

### Sector Analysis

Each Beige Book report is broken into individual sector paragraphs (Manufacturing, Employment, Real Estate, etc.) scraped directly from the HTML structure. The dataset contains **10,728 sector-level observations** across 122 reports (2011--2026), covering 12 canonical sectors per district.

![Sector Sentiment Heatmap](output/sector_heatmap.png)

**Key findings:**

- **Sector predictive power**: Employment (r = +0.17) and Tourism (r = +0.18) positively predict economic activity. Financial Services (r = -0.27) has a *negative* correlation -- optimism in finance signals overheating, not genuine strength.
- **Geographic pessimism gradient**: Northeast > Midwest > South > West Coast. San Francisco is the most pessimistic district across nearly every sector, with three sectors (Financial Services, Transportation, Construction) averaging negative sentiment.
- **Cross-district synchronization**: Energy is nationally synchronized (r = 0.86) -- when oil moves, every district talks about it the same way. Manufacturing is the most locally driven (r = 0.59), making it the best sector for detecting regional variation.
- **Cleveland Employment** (r = 0.61) is the single most predictive sector-district pair, followed by Boston Manufacturing (r = 0.55) and San Francisco Employment (r = 0.55).
- **Rate hikes (2022-23) crushed Transportation** (-0.37 sentiment drop, the "freight recession") and Real Estate (-0.19), while Agriculture *improved* (+0.26) from high commodity prices.
- **COVID hit Employment universally** -- all 10 worst sector-district pairs were Employment -- but recovery was symmetric: the hardest-hit bounced back fastest.

For the full regional deep dive (district-by-district profiles, sector specialization, divergence analysis), see **[ANALYSIS.md](ANALYSIS.md)**.

## Project Structure

```
beige_book/
├── src/
│   ├── config.py          # Constants, paths, API keys, district names
│   ├── acquire.py         # Beige Book scraper + FRED data fetcher
│   ├── prepare.py         # Text cleaning, time alignment, merging
│   ├── sentiment.py       # VADER + FinBERT sentiment scoring
│   ├── explore.py         # Visualization functions
│   ├── hypothesis.py      # Statistical tests (correlation, Granger)
│   ├── model.py           # OLS regression, out-of-sample testing
│   ├── sectors.py         # Sector extraction via keyword classification
│   ├── scrape_sectors.py  # Sector-level paragraph scraper from cached HTML
│   └── maps.py            # Interactive choropleth maps (Plotly)
├── data/                  # Scraped data + FRED CSVs (gitignored)
│   └── raw_html/          # Cached HTML pages
├── output/                # Generated plots and results
├── run_pipeline.py        # End-to-end pipeline runner
├── Final-Report.ipynb     # Full analysis notebook with findings
├── ANALYSIS.md            # Regional deep dive and sector analysis
├── MVP.ipynb              # Original prototype (reference)
└── .env                   # FRED_API_KEY (gitignored)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a FRED API key

Request a free key at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).

Create a `.env` file in the project root:

```
FRED_API_KEY=your_key_here
```

## Usage

```bash
python run_pipeline.py
```

## Pipeline Steps

| Step | Module | What it does |
|------|--------|-------------|
| 1. Acquire | `src/acquire.py` | Scrapes Beige Book reports from federalreserve.gov (2011--2026) and fetches indicator series from the FRED API |
| 2. Prepare | `src/prepare.py` | Cleans text, normalizes district names, and aligns Beige Book dates to FRED reporting periods using `merge_asof(direction='forward')` |
| 3. Sentiment | `src/sentiment.py` | Scores each district summary with VADER compound sentiment |
| 4. Aggregate | `src/prepare.py` | Computes national sentiment aggregates (mean/std across 12 districts) and merges with FRED data |
| 5. Explore | `src/explore.py` | Generates time series plots, a regional heatmap, and dual-axis sentiment-vs-indicator charts |
| 6. Hypothesis | `src/hypothesis.py` | Runs lagged Pearson/Spearman correlations and Granger causality tests (up to 4 lags) |
| 7. Model | `src/model.py` | Fits OLS regressions (simple and controlled) and runs out-of-sample tests (train through 2018, test 2019+) comparing sentiment-augmented models against a lagged-indicator baseline |

## Output

**Plots** saved to `output/`:

- `sentiment_timeseries.png` -- National sentiment over time with +/- 1 std band
- `regional_comparison.png` -- Heatmap of sentiment across all 12 districts
- `sentiment_vs_gdpc1.png` -- Dual-axis: sentiment vs. Real GDP
- `sentiment_vs_unrate.png` -- Dual-axis: sentiment vs. Unemployment
- `sentiment_vs_cpiaucsl.png` -- Dual-axis: sentiment vs. CPI
- `sentiment_vs_sp500.png` -- Dual-axis: sentiment vs. S&P 500
- `regional_correlation_bars.png` -- Per-district predictive power
- `regional_sentiment_vs_economy.png` -- District sentiment vs. state economic activity scatter
- `district_timeseries_grid.png` -- 12-panel grid: sentiment + economic activity per district
- `sector_heatmap.png` -- Sector sentiment heatmap across districts
- `sector_timeseries.png` -- Sector sentiment over time
- `sector_manufacturing_grid.png` -- Manufacturing sentiment by district
- `sector_volatility.png` -- Sector sentiment volatility comparison

**Interactive maps** (HTML, open in browser for hover details):

- `map_sector_grid.html` -- 6-sector choropleth grid
- `map_dominant_strongest.html` -- Each district's strongest sector
- `map_dominant_weakest.html` -- Each district's weakest sector
- `map_manufacturing.html` / `map_employment.html` / `map_real_estate.html` / `map_energy.html` -- Individual sector choropleths
- `map_*_animated.html` -- Animated sentiment over time for each sector (play button + date slider)

**Console output** includes:

- Lagged correlation tables (Pearson r, Spearman r, p-values at lags 0--3)
- Granger causality F-statistics and p-values at lags 1--4
- OLS regression coefficient tables (simple model and controlled model with lagged indicator)
- Out-of-sample RMSE, MAE, and directional accuracy for baseline vs. sentiment-augmented model

## Key Technical Decisions

- **VADER over TextBlob and FinBERT** -- VADER wins 10/12 districts in head-to-head comparison; its positive bias aligns with the Beige Book's optimistic baseline
- **Time alignment** -- Beige Book at time T maps to indicator at T+1 (forward-looking test)
- **Controlled regressions** -- include lagged indicator as a control to isolate sentiment's marginal contribution
- **Out-of-sample split** -- train through 2018, test 2019+ to guard against overfitting

## Limitations

- **VADER** is a general-purpose sentiment tool, not tuned for economic language (e.g., "moderate" is positive in Fed-speak but neutral to VADER). However, its positive bias is actually an advantage here -- the Beige Book's optimistic baseline means VADER deviations carry real signal.
- **FinBERT-FOMC** reads policy tone (hawkish/dovish), not economic descriptions, which explains its underperformance on Beige Book text despite being trained on Fed language.
- The **2019--2026 test period** includes COVID and rapid rate hikes, which may inflate apparent predictive power
- **Time alignment** is approximate — Beige Book publication dates don't perfectly match indicator release dates

## Future Work

- **Sector-gated sentiment** — score each sector paragraph independently per district, then correlate sector-specific sentiment with corresponding economic indicators (e.g., Manufacturing sentiment vs. industrial production). The 10,728 sector-level paragraphs are scraped and ready for scoring.
- Train a **custom Beige Book-specific sentiment model** (three models tested: VADER won 10/12, but a model trained on Beige Book text with economic outcome labels could outperform all three)
- **Multi-state aggregation** — weight constituent states by GDP/employment for more accurate district-level indicators
- Extend data back to **1996** for more economic cycles
- Test **real-time forecasting** accuracy at the time of each publication
- Build a **dashboard** for browsing district and sector sentiment interactively
