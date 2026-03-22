# Does the Federal Reserve's Beige Book Predict the Economy?

A sentiment analysis of regional economic narratives from the 12 Federal Reserve district banks (2011–2025).

## Overview

This project scrapes "Summary of Economic Activity" text from each district's Beige Book report (1,440 district-level summaries across 120 reports), scores sentiment using VADER, and tests whether that sentiment has **predictive power** for four economic indicators:

| FRED Series | Indicator | Frequency |
|-------------|-----------|-----------|
| `GDPC1` | Real GDP | Quarterly |
| `UNRATE` | Unemployment Rate | Monthly |
| `CPIAUCSL` | CPI | Monthly |
| `SP500` | S&P 500 | Daily |

Statistical methods: lagged Pearson/Spearman correlations, Granger causality tests, OLS regression with lead-lag structure, and out-of-sample evaluation.

## Key Findings

| Indicator | Granger Causes? | Correlation (r) | OLS Controlled (p) | Out-of-Sample |
|-----------|----------------|-----------------|---------------------|---------------|
| **Unemployment** | Yes (lags 3–4) | **-0.59** | **p < 0.001** | Marginal RMSE improvement |
| **GDP** | Yes (lags 1–2) | 0.15 | Insufficient quarterly data | — |
| **CPI** | Yes (lags 3–4) | 0.05 | **p = 0.001** | RMSE improves by 0.08 |
| **S&P 500** | No | -0.28 | p = 0.074 | No improvement |

**Bottom line:** Beige Book sentiment has genuine predictive power for **unemployment** and **CPI/inflation**, even after controlling for each indicator's own history. It Granger-causes GDP at short lags. No predictive value for the S&P 500.

The strongest signal is **unemployment** (r = -0.59): when Beige Book narratives turn more positive, unemployment tends to fall in subsequent months. This makes intuitive sense — the district banks are directly surveying businesses about hiring conditions.

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
│   └── model.py       # OLS regression, out-of-sample testing
├── data/              # Scraped data + FRED CSVs (gitignored)
│   └── raw_html/      # Cached HTML pages
├── output/            # Generated plots and results
├── run_pipeline.py    # End-to-end pipeline runner
├── Final-Report.ipynb # Full analysis notebook with findings
├── MVP.ipynb          # Original prototype (reference)
└── .env               # FRED_API_KEY (gitignored)
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
| 1. Acquire | `src/acquire.py` | Scrapes Beige Book reports from federalreserve.gov (2011--2025) and fetches indicator series from the FRED API |
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

**Console output** includes:

- Lagged correlation tables (Pearson r, Spearman r, p-values at lags 0--3)
- Granger causality F-statistics and p-values at lags 1--4
- OLS regression coefficient tables (simple model and controlled model with lagged indicator)
- Out-of-sample RMSE, MAE, and directional accuracy for baseline vs. sentiment-augmented model

## Key Technical Decisions

- **VADER over TextBlob** -- better calibrated for economic/financial language
- **Time alignment** -- Beige Book at time T maps to indicator at T+1 (forward-looking test)
- **Controlled regressions** -- include lagged indicator as a control to isolate sentiment's marginal contribution
- **Out-of-sample split** -- train through 2018, test 2019+ to guard against overfitting

## Limitations

- **VADER** is a general-purpose sentiment tool, not tuned for economic language (e.g., "moderate" is positive in Fed-speak but neutral to VADER)
- The **2019–2025 test period** is dominated by COVID, which may inflate apparent predictive power
- **Time alignment** is approximate — Beige Book publication dates don't perfectly match indicator release dates

## Future Work

- Use **FinBERT** for domain-specific economic sentiment scoring
- Add **regional-level** analysis (district sentiment vs. district-level unemployment)
- Extend data back to **1996** for more economic cycles
- Test **real-time forecasting** accuracy at the time of each publication
