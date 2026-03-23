---
description: Run the full Beige Book sentiment analysis pipeline and summarize results
user_invocable: true
---

# Run Pipeline

Run the end-to-end Beige Book pipeline and report results.

## Steps

1. Run: `python run_pipeline.py`
2. After completion, summarize:
   - Number of Beige Book reports scraped and districts covered
   - Sentiment score range
   - Key findings from lagged correlations (which indicators correlate most with sentiment?)
   - Granger causality results (does sentiment predict any indicators?)
   - Out-of-sample model performance (RMSE, R-squared)
   - Any errors or warnings that occurred
3. List the plots generated in `output/`
