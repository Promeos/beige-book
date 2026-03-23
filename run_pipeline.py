"""
Beige Book Sentiment Analysis Pipeline

End-to-end pipeline that scrapes Beige Book text, scores sentiment,
fetches FRED economic indicators, and tests whether sentiment has
predictive power for economic outcomes.

Usage:
    python run_pipeline.py
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Run the full Beige Book sentiment analysis pipeline.

    Steps: acquire data, clean text, score sentiment, align with FRED
    indicators, generate visualizations, run statistical tests, and
    evaluate predictive models.
    """
    # ---- Step 1: Acquire data ----
    logger.info("Step 1: Acquiring data...")

    from src.acquire import get_beige_data, get_fred_data

    beige_df = get_beige_data()
    logger.info(
        "Beige Book: %d rows, %d districts, %d report dates",
        len(beige_df),
        beige_df["district"].nunique(),
        beige_df["date"].nunique(),
    )

    fred_df = get_fred_data()
    logger.info("FRED indicators: %d rows", len(fred_df))

    # ---- Step 2: Prepare ----
    logger.info("Step 2: Preparing data...")

    from src.prepare import (
        prep_beige_data,
        align_time_periods,
        compute_national_aggregate,
    )

    beige_df = prep_beige_data(beige_df)

    # ---- Step 3: Sentiment scoring ----
    logger.info("Step 3: Scoring sentiment...")

    from src.sentiment import add_sentiment_scores

    beige_df = add_sentiment_scores(beige_df)
    logger.info(
        "Sentiment score range: [%.3f, %.3f]",
        beige_df["vader_compound"].min(),
        beige_df["vader_compound"].max(),
    )

    # ---- Step 4: Compute aggregates and align ----
    logger.info("Step 4: Computing national aggregates and aligning with FRED...")

    national_df = compute_national_aggregate(beige_df)
    merged_df = align_time_periods(national_df, fred_df)
    logger.info("Merged dataset: %d rows", len(merged_df))

    # ---- Step 5: Explore ----
    logger.info("Step 5: Generating visualizations...")

    from src.explore import (
        plot_sentiment_timeseries,
        plot_regional_comparison,
        plot_sentiment_vs_indicator,
    )

    plot_sentiment_timeseries(national_df)
    plot_regional_comparison(beige_df)

    indicator_labels = {
        "GDPC1": "Real GDP",
        "UNRATE": "Unemployment Rate (%)",
        "CPIAUCSL": "CPI",
        "SP500": "S&P 500",
    }
    for col, label in indicator_labels.items():
        if col in merged_df.columns:
            plot_sentiment_vs_indicator(merged_df, col, label)

    # ---- Step 6: Statistical tests ----
    logger.info("Step 6: Running statistical tests...")

    from src.hypothesis import compute_lagged_correlations, run_granger_tests

    print("\n" + "=" * 60)
    print("LAGGED CORRELATIONS")
    print("=" * 60)
    corr_df = compute_lagged_correlations(merged_df)
    if not corr_df.empty:
        print(corr_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("GRANGER CAUSALITY TESTS")
    print("=" * 60)
    run_granger_tests(merged_df)

    # ---- Step 7: Predictive models ----
    logger.info("Step 7: Running predictive models...")

    from src.model import run_all_regressions, out_of_sample_test

    print("\n" + "=" * 60)
    print("OLS REGRESSIONS")
    print("=" * 60)
    run_all_regressions(merged_df)

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE EVALUATION")
    print("=" * 60)
    for col in ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]:
        if col in merged_df.columns:
            out_of_sample_test(merged_df, col)

    # ---- Step 8: Sector-specific predictive analysis ----
    logger.info("Step 8: Sector-specific predictive analysis...")

    import pandas as pd
    from src.config import DATA_DIR
    from src.acquire import get_sector_fred_data
    from src.prepare import (
        compute_sector_national_aggregate,
        align_sector_with_indicators,
    )
    from src.hypothesis import (
        compute_sector_indicator_correlations,
        run_sector_granger_tests,
    )
    from src.model import run_sector_regressions, sector_out_of_sample_test
    from src.explore import plot_sector_vs_indicator, plot_sector_predictive_grid

    sector_csv = DATA_DIR / "sector_sentiment.csv"
    if sector_csv.exists():
        sector_df = pd.read_csv(sector_csv, parse_dates=["date"])
        sector_fred_df = get_sector_fred_data()

        sector_national = compute_sector_national_aggregate(sector_df)
        sector_merged = align_sector_with_indicators(sector_national, sector_fred_df)
        logger.info(
            "Sector-indicator merged: %d rows, %d sectors",
            len(sector_merged),
            sector_merged["sector"].nunique(),
        )

        print("\n" + "=" * 60)
        print("SECTOR-INDICATOR CORRELATIONS")
        print("=" * 60)
        compute_sector_indicator_correlations(sector_merged)

        print("\n" + "=" * 60)
        print("SECTOR GRANGER CAUSALITY")
        print("=" * 60)
        run_sector_granger_tests(sector_merged)

        print("\n" + "=" * 60)
        print("SECTOR OLS REGRESSIONS")
        print("=" * 60)
        run_sector_regressions(sector_merged)

        print("\n" + "=" * 60)
        print("SECTOR OUT-OF-SAMPLE EVALUATION")
        print("=" * 60)
        sector_out_of_sample_test(sector_merged)

        # Visualizations
        plot_sector_predictive_grid(sector_merged)
        for sector in sector_merged["sector"].unique():
            plot_sector_vs_indicator(sector_merged, sector)
    else:
        logger.warning("No sector_sentiment.csv found. Skipping sector analysis.")

    # ---- Step 9: Sector-gated sentiment (sentence-level) ----
    logger.info("Step 9: Sector-gated sentiment scoring...")

    from src.sectors import (
        build_sentence_sector_dataframe,
        aggregate_sentence_sector_scores,
    )

    sentence_sector_df = build_sentence_sector_dataframe(beige_df)
    gated_df = aggregate_sentence_sector_scores(sentence_sector_df)

    sentence_sector_df.to_csv(DATA_DIR / "sector_sentence_detail.csv", index=False)
    gated_df.to_csv(DATA_DIR / "sector_sentiment_gated.csv", index=False)
    logger.info(
        "Sentence-gated: %d sentences → %d aggregated rows",
        len(sentence_sector_df),
        len(gated_df),
    )

    # Compare gated vs paragraph-level predictive power
    if sector_csv.exists():
        gated_national = compute_sector_national_aggregate(gated_df)
        gated_merged = align_sector_with_indicators(gated_national, sector_fred_df)

        if not gated_merged.empty:
            print("\n" + "=" * 60)
            print("SENTENCE-GATED SECTOR CORRELATIONS")
            print("=" * 60)
            compute_sector_indicator_correlations(gated_merged)

    # ---- Step 10: Robustness checks ----
    logger.info("Step 10: Running robustness checks...")

    from src.robustness import run_all_robustness_checks, run_sector_fdr_correction

    run_all_robustness_checks(merged_df)

    # FDR correction on sector-district correlations (if regional data available)
    if sector_csv.exists():
        from src.hypothesis import compute_sector_district_correlations
        from src.acquire import get_regional_fred_data
        from src.prepare import align_regional_data

        regional_fred_df = get_regional_fred_data()
        regional_merged = align_regional_data(beige_df, regional_fred_df)
        if not regional_merged.empty:
            sector_district_corr = compute_sector_district_correlations(
                sector_df, regional_merged
            )
            if not sector_district_corr.empty:
                run_sector_fdr_correction(sector_district_corr)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
