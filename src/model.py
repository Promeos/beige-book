"""
Predictive models for testing Beige Book sentiment forecasting power.

OLS regression with lead-lag structure and out-of-sample evaluation.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error


def run_ols_regression(df, target_col, sentiment_col="sentiment_mean", controls=None):
    """
    Run OLS regression: indicator_{t+1} = α + β·sentiment_t + ε

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Aligned data where sentiment at T is matched with indicator at T+1.
    target_col : str
        Column name of the target indicator.
    sentiment_col : str
    controls : list of str
        Additional control variables (e.g., lagged indicator).

    Returns
    -------
    result : statsmodels.regression.linear_model.RegressionResultsWrapper
    """
    predictors = [sentiment_col]
    if controls:
        predictors.extend(controls)

    data = df[[target_col] + predictors].dropna()
    if len(data) < 20:
        print(f"Insufficient data for {target_col} regression ({len(data)} obs)")
        return None

    X = sm.add_constant(data[predictors])
    y = data[target_col]

    model = sm.OLS(y, X).fit()
    return model


def run_all_regressions(df, indicator_cols=None, sentiment_col="sentiment_mean"):
    """
    Run OLS regressions for all indicators, with and without controls.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    indicator_cols : list of str
    sentiment_col : str

    Returns
    -------
    results : dict
        results[indicator] = {"simple": model, "controlled": model}
    """
    if indicator_cols is None:
        indicator_cols = ["GDPC1", "UNRATE", "CPIAUCSL", "SP500"]

    results = {}
    for col in indicator_cols:
        if col not in df.columns:
            continue

        print(f"\n{'=' * 60}")
        print(f"Target: {col}")
        print(f"{'=' * 60}")

        # Simple regression: sentiment only
        print("\n--- Simple Model ---")
        simple = run_ols_regression(df, col, sentiment_col)
        if simple:
            print(simple.summary().tables[1])

        # Controlled regression: sentiment + lagged indicator
        lagged_col = f"{col}_lag1"
        df_ctrl = df.copy()
        df_ctrl[lagged_col] = df_ctrl[col].shift(1)

        print("\n--- Controlled Model (with lagged indicator) ---")
        controlled = run_ols_regression(
            df_ctrl, col, sentiment_col, controls=[lagged_col]
        )
        if controlled:
            print(controlled.summary().tables[1])

        results[col] = {"simple": simple, "controlled": controlled}

    return results


def out_of_sample_test(
    df, target_col, sentiment_col="sentiment_mean", train_end="2018-12-31"
):
    """
    Evaluate out-of-sample predictive accuracy.

    Compares a model using sentiment + lagged indicator vs. lagged indicator only.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    target_col : str
    sentiment_col : str
    train_end : str
        Last date in training set.

    Returns
    -------
    metrics : dict
        RMSE, MAE, and directional accuracy for both models.
    """
    df = df.copy()
    df[f"{target_col}_lag1"] = df[target_col].shift(1)
    lagged_col = f"{target_col}_lag1"

    data = df[["date", target_col, sentiment_col, lagged_col]].dropna()
    train = data[data["date"] <= train_end]
    test = data[data["date"] > train_end]

    if len(train) < 20 or len(test) < 5:
        print(f"Insufficient data for {target_col} out-of-sample test")
        return None

    # Model 1: Baseline (lagged indicator only)
    X_train_base = sm.add_constant(train[[lagged_col]])
    X_test_base = sm.add_constant(test[[lagged_col]])
    y_train = train[target_col]
    y_test = test[target_col]

    baseline = sm.OLS(y_train, X_train_base).fit()
    pred_base = baseline.predict(X_test_base)

    # Model 2: Sentiment + lagged indicator
    X_train_full = sm.add_constant(train[[sentiment_col, lagged_col]])
    X_test_full = sm.add_constant(test[[sentiment_col, lagged_col]])

    full_model = sm.OLS(y_train, X_train_full).fit()
    pred_full = full_model.predict(X_test_full)

    # Compute metrics
    metrics = {
        "baseline": {
            "rmse": np.sqrt(mean_squared_error(y_test, pred_base)),
            "mae": mean_absolute_error(y_test, pred_base),
            "directional_accuracy": _directional_accuracy(y_test, pred_base),
        },
        "sentiment_model": {
            "rmse": np.sqrt(mean_squared_error(y_test, pred_full)),
            "mae": mean_absolute_error(y_test, pred_full),
            "directional_accuracy": _directional_accuracy(y_test, pred_full),
        },
    }

    print(f"\nOut-of-sample results for {target_col}:")
    print(f"  Baseline RMSE: {metrics['baseline']['rmse']:.4f}")
    print(f"  Sentiment RMSE: {metrics['sentiment_model']['rmse']:.4f}")
    print(f"  Baseline Dir. Acc: {metrics['baseline']['directional_accuracy']:.1%}")
    print(
        f"  Sentiment Dir. Acc: {metrics['sentiment_model']['directional_accuracy']:.1%}"
    )

    improvement = metrics["baseline"]["rmse"] - metrics["sentiment_model"]["rmse"]
    if improvement > 0:
        print(f"  → Sentiment model improves RMSE by {improvement:.4f}")
    else:
        print("  → Sentiment model does NOT improve over baseline")

    return metrics


def _directional_accuracy(actual, predicted):
    """
    Fraction of times the model correctly predicts direction of change.

    Parameters
    ----------
    actual : array-like
    predicted : array-like

    Returns
    -------
    float
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    actual_dir = np.diff(actual) > 0
    pred_dir = np.diff(predicted) > 0
    if len(actual_dir) == 0:
        return 0.0
    return np.mean(actual_dir == pred_dir)


def run_sector_regressions(sector_merged_df):
    """
    Run OLS regressions for each sector: does sector sentiment predict
    its matched FRED indicator?

    Parameters
    ----------
    sector_merged_df : pandas.core.frame.DataFrame
        Columns: date, sector, sentiment_mean, indicator_value, indicator_id.

    Returns
    -------
    results : dict
        results[sector] = {"simple": model, "controlled": model_or_None}.
    """
    results = {}

    print("\nSector OLS Regressions")
    print("=" * 70)

    for sector in sorted(sector_merged_df["sector"].unique()):
        subset = sector_merged_df[sector_merged_df["sector"] == sector].copy()
        subset = subset[["sentiment_mean", "indicator_value"]].dropna()

        if len(subset) < 15:
            print(f"\n{sector}: insufficient data (n={len(subset)})")
            continue

        indicator_id = sector_merged_df[sector_merged_df["sector"] == sector][
            "indicator_id"
        ].iloc[0]

        print(f"\n{sector} → {indicator_id} (n={len(subset)})")
        print("-" * 50)

        # Simple model: indicator = α + β·sentiment
        y = subset["indicator_value"]
        X = sm.add_constant(subset[["sentiment_mean"]])
        simple = sm.OLS(y, X).fit()

        coef = simple.params.get("sentiment_mean", 0)
        p = simple.pvalues.get("sentiment_mean", 1)
        marker = "***" if p < 0.05 else ""
        print(
            f"  Simple:     β={coef:+.4f}, p={p:.4f}, R²={simple.rsquared:.3f} {marker}"
        )

        # Controlled model: indicator = α + β₁·sentiment + β₂·lag(indicator)
        subset = subset.copy()
        subset["indicator_lag"] = subset["indicator_value"].shift(1)
        clean = subset.dropna()

        controlled = None
        if len(clean) >= 15:
            y_c = clean["indicator_value"]
            X_c = sm.add_constant(clean[["sentiment_mean", "indicator_lag"]])
            controlled = sm.OLS(y_c, X_c).fit()

            coef_c = controlled.params.get("sentiment_mean", 0)
            p_c = controlled.pvalues.get("sentiment_mean", 1)
            marker_c = "***" if p_c < 0.05 else ""
            print(
                f"  Controlled: β={coef_c:+.4f}, p={p_c:.4f}, "
                f"R²={controlled.rsquared:.3f} {marker_c}"
            )

        results[sector] = {"simple": simple, "controlled": controlled}

    return results


def sector_out_of_sample_test(sector_merged_df, train_end="2018-12-31"):
    """
    Out-of-sample test for each sector: does adding sentiment improve
    prediction over a lagged-indicator baseline?

    Parameters
    ----------
    sector_merged_df : pandas.core.frame.DataFrame
        Columns: date, sector, sentiment_mean, indicator_value.
    train_end : str

    Returns
    -------
    results : dict
        results[sector] = {baseline: {rmse, mae}, sentiment_model: {rmse, mae}}.
    """
    results = {}

    print(
        "\nSector Out-of-Sample Tests (train ≤ {}, test > {})".format(
            train_end, train_end
        )
    )
    print("=" * 70)

    for sector in sorted(sector_merged_df["sector"].unique()):
        subset = sector_merged_df[sector_merged_df["sector"] == sector].copy()
        subset = subset.sort_values("date")
        subset["indicator_lag"] = subset["indicator_value"].shift(1)
        subset = subset[
            ["date", "sentiment_mean", "indicator_value", "indicator_lag"]
        ].dropna()

        train = subset[subset["date"] <= train_end]
        test = subset[subset["date"] > train_end]

        if len(train) < 15 or len(test) < 5:
            continue

        indicator_id = sector_merged_df[sector_merged_df["sector"] == sector][
            "indicator_id"
        ].iloc[0]

        y_train = train["indicator_value"]
        y_test = test["indicator_value"]

        # Baseline: lagged indicator only
        X_train_base = sm.add_constant(train[["indicator_lag"]])
        X_test_base = sm.add_constant(test[["indicator_lag"]])
        base_model = sm.OLS(y_train, X_train_base).fit()
        pred_base = base_model.predict(X_test_base)

        # Full: sentiment + lagged indicator
        X_train_full = sm.add_constant(train[["sentiment_mean", "indicator_lag"]])
        X_test_full = sm.add_constant(test[["sentiment_mean", "indicator_lag"]])
        full_model = sm.OLS(y_train, X_train_full).fit()
        pred_full = full_model.predict(X_test_full)

        base_rmse = np.sqrt(mean_squared_error(y_test, pred_base))
        full_rmse = np.sqrt(mean_squared_error(y_test, pred_full))
        improvement = base_rmse - full_rmse

        results[sector] = {
            "baseline": {
                "rmse": base_rmse,
                "mae": mean_absolute_error(y_test, pred_base),
            },
            "sentiment_model": {
                "rmse": full_rmse,
                "mae": mean_absolute_error(y_test, pred_full),
            },
        }

        marker = "✓" if improvement > 0 else "✗"
        print(
            f"  {sector:25s} → {indicator_id:15s}  "
            f"Base RMSE={base_rmse:.4f}  Sent RMSE={full_rmse:.4f}  "
            f"Δ={improvement:+.4f} {marker}"
        )

    return results
