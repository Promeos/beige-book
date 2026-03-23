---
description: Quick summary stats on the current Beige Book dataset
user_invocable: true
---

# Explore Data

Load the project's datasets and print a quick health check.

## Steps

Run a Python script that:

1. Loads `data/beige_book.csv` and reports:
   - Row count, date range, number of unique districts
   - Missing values per column
   - Sample of 3 random rows

2. Loads `data/fred_indicators.csv` and reports:
   - Row count, date range
   - Which indicators are present (GDPC1, UNRATE, CPIAUCSL, SP500)
   - Missing values per column

3. If `data/sector_sentiment.csv` exists, report:
   - Row count, sectors covered, date range

4. Flag any data quality issues (gaps in dates, missing districts, null sentiment scores)
