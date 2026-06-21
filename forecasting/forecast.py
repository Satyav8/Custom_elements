"""
Demand forecasting for Part B.
Approach: Prophet per SKU vs. seasonal-naive baseline.
Holdout: last 4 weeks (strict temporal split, no leakage).
Run: python -m forecasting.forecast
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from prophet import Prophet

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sales_history.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
HOLDOUT_WEEKS = 4


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def seasonal_naive_forecast(train: pd.Series, n: int, season: int = 52) -> list:
    """Repeat the values from exactly one season ago."""
    preds = []
    for i in range(n):
        idx = len(train) - season + i
        preds.append(float(train.iloc[idx]) if idx >= 0 else float(train.mean()))
    return preds


def prophet_forecast(train_df: pd.DataFrame, n: int, has_promo: bool = True) -> list:
    df = train_df.rename(columns={"date": "ds", "units_sold": "y"})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    if has_promo and "promo_flag" in train_df.columns:
        m.add_regressor("promo_flag")
        future_regressors = pd.DataFrame({"ds": pd.date_range(df["ds"].max() + pd.Timedelta("7D"), periods=n, freq="W-MON"), "promo_flag": [0] * n})
    m.fit(df[["ds", "y"] + (["promo_flag"] if has_promo and "promo_flag" in train_df.columns else [])])

    future = m.make_future_dataframe(periods=n, freq="W")
    if has_promo and "promo_flag" in train_df.columns:
        future = future.merge(future_regressors, on="ds", how="left").fillna(0)

    forecast = m.predict(future)
    preds = forecast.tail(n)["yhat"].clip(lower=0).tolist()
    return preds


def run():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    skus = df["sku"].unique()
    all_results = []

    for sku in skus:
        sku_df = df[df["sku"] == sku].reset_index(drop=True)
        cutoff = sku_df["date"].max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)
        train = sku_df[sku_df["date"] <= cutoff]
        test = sku_df[sku_df["date"] > cutoff]

        if len(train) < 10 or len(test) == 0:
            continue

        y_true = test["units_sold"].tolist()
        n = len(test)

        # Baseline: seasonal naive
        baseline_preds = seasonal_naive_forecast(train["units_sold"], n)
        baseline_mae = mae(y_true, baseline_preds)
        baseline_mape = mape(y_true, baseline_preds)

        # Model: Prophet
        try:
            prophet_preds = prophet_forecast(train[["date", "units_sold", "promo_flag"]], n)
            model_mae = mae(y_true, prophet_preds)
            model_mape = mape(y_true, prophet_preds)
            beats_baseline = model_mae < baseline_mae
        except Exception as e:
            prophet_preds = baseline_preds
            model_mae = baseline_mae
            model_mape = baseline_mape
            beats_baseline = False
            print(f"  [{sku}] Prophet failed: {e}, falling back to baseline.")

        all_results.append({
            "sku": sku,
            "train_weeks": len(train),
            "test_weeks": n,
            "y_true": y_true,
            "baseline_preds": [round(p, 2) for p in baseline_preds],
            "model_preds": [round(p, 2) for p in prophet_preds],
            "baseline_mae": round(baseline_mae, 3),
            "baseline_mape": round(baseline_mape, 2),
            "model_mae": round(model_mae, 3),
            "model_mape": round(model_mape, 2),
            "beats_baseline": beats_baseline,
        })
        print(f"[{sku}] Baseline MAE={baseline_mae:.2f} | Prophet MAE={model_mae:.2f} | {'BEATS' if beats_baseline else 'worse'}")

    # Aggregate metrics
    overall_baseline_mae = np.mean([r["baseline_mae"] for r in all_results])
    overall_model_mae = np.mean([r["model_mae"] for r in all_results])
    n_beats = sum(r["beats_baseline"] for r in all_results)

    print(f"\n{'='*50}")
    print(f"Overall Baseline MAE : {overall_baseline_mae:.3f}")
    print(f"Overall Prophet MAE  : {overall_model_mae:.3f}")
    print(f"SKUs where Prophet beats baseline: {n_beats}/{len(all_results)}")
    print(f"{'='*50}")

    output = {
        "holdout_weeks": HOLDOUT_WEEKS,
        "model": "Prophet",
        "baseline": "Seasonal Naive (lag-52 weeks)",
        "overall": {
            "baseline_mae": round(overall_baseline_mae, 3),
            "model_mae": round(overall_model_mae, 3),
            "skus_beat": n_beats,
            "skus_total": len(all_results),
        },
        "per_sku": all_results,
    }

    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp, RESULTS_PATH)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
