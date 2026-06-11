"""Train, evaluate and predict with the hourly day-ahead model.

A HistGradientBoostingRegressor on the long (day x hour) table from
hourly_dataset.py. Evaluation: chronological holdout on the last months,
scored against hourly persistence ("tomorrow's hour h = today's hour h") -
a genuinely hard baseline, since the daily shape barely moves between days.

    python hourly_model.py                 # holdout eval + graph
    python hourly_model.py --months 6
    python hourly_model.py --zone SE3
    python hourly_model.py --predict [--retrain]
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from hourly_dataset import build_hourly_dataset, split_features_target, FEATURE_COLUMNS
from data_fetchers import fetch_prices_for_day, regional_hourly_forecast
from historical_data import WIND_STATIONS, load_smhi_hourly
from smhi_stations import download_station_latest_months, WIND_PARAM
from calendar_features import build_calendar_features

PLOT_PATH = "hourly_evaluation.png"


def model_path(zone: str) -> Path:
    """One artifact per zone - an SE4 model knows nothing about SE3 levels."""
    return Path(f"hourly_model_{zone}.joblib")


def build_model(random_state: int = 42) -> HistGradientBoostingRegressor:
    """
    HistGradientBoosting handles NaN natively and is quick on tabular data.
    Moderately regularised so the wild 2021-2022 stretch doesn't get
    memorised. `random_state` seeds the early-stopping validation split,
    so refits with different seeds give a feel for fit noise.
    """
    return HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=10,
        random_state=random_state,
    )


def _metrics(actual, predicted) -> dict:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
    }


def run_holdout(zone: str = "SE4", holdout_months: int = 4,
                df: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    """
    Train on everything but the last `holdout_months`, predict those.

    No printing or plotting here so the dashboard can reuse it; pass a
    pre-built `df` to skip the dataset build. Returns a dict with model, df,
    train_df, test_df, X_test, y_test and `results`, a per-(day, hour) frame
    of timestamp, hour, actual, predicted and baseline.
    """
    if df is None:
        df = build_hourly_dataset(zone=zone, verbose=verbose)

    cutoff = df["date"].max() - pd.DateOffset(months=holdout_months)
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff].copy()
    if test_df.empty:
        raise RuntimeError("Holdout window is empty - try fewer months.")

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    model = build_model()
    model.fit(X_train, y_train)

    # The prediction is for tomorrow's hour h, so the timestamp is date + 1 day + h.
    ts = (test_df["date"] + pd.Timedelta(days=1)
          + pd.to_timedelta(test_df["hour"], unit="h"))
    results = pd.DataFrame({
        "timestamp": ts.to_numpy(),
        "hour": test_df["hour"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": model.predict(X_test),
        # Persistence: tomorrow hour h = today hour h (a feature in X_test).
        "baseline": X_test["price_today_same_hour"].to_numpy(),
    }).sort_values("timestamp").reset_index(drop=True)

    return {"model": model, "df": df, "train_df": train_df, "test_df": test_df,
            "X_test": X_test, "y_test": y_test, "results": results}


def holdout_evaluation(zone: str = "SE4", holdout_months: int = 4) -> pd.DataFrame:
    """Run the holdout, print the report, and write the graph (CLI entry)."""
    h = run_holdout(zone=zone, holdout_months=holdout_months)
    train_df, test_df = h["train_df"], h["test_df"]
    print(
        f"\nChronological split:\n"
        f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} "
        f"({len(train_df):,} rows)\n"
        f"  Test : {test_df['date'].min().date()} to {test_df['date'].max().date()} "
        f"({len(test_df):,} rows)\n"
    )
    _report(h["results"])
    _plot(h["results"])
    return h["results"]


def _report(results: pd.DataFrame) -> None:
    model_m = _metrics(results["actual"], results["predicted"])
    base_m = _metrics(results["actual"], results["baseline"])
    mae_gain = (1 - model_m["mae"] / base_m["mae"]) * 100 if base_m["mae"] else float("nan")
    rmse_gain = (1 - model_m["rmse"] / base_m["rmse"]) * 100 if base_m["rmse"] else float("nan")

    print("=" * 60)
    print("  OUT-OF-SAMPLE RESULTS (hourly, held-out months)")
    print("=" * 60)
    print(f"  {'':14}{'MAE':>12}{'RMSE':>12}   (SEK/kWh)")
    print(f"  {'Model':14}{model_m['mae']:>12.4f}{model_m['rmse']:>12.4f}")
    print(f"  {'Baseline':14}{base_m['mae']:>12.4f}{base_m['rmse']:>12.4f}   (tomorrow h = today h)")
    print("-" * 60)
    print(f"  Improvement over baseline:  MAE {mae_gain:+.1f}%   RMSE {rmse_gain:+.1f}%")
    print("=" * 60)

    # Per-hour MAE, so we can see where the model helps most.
    by_hour = (results.assign(ae=(results["predicted"] - results["actual"]).abs())
               .groupby("hour")["ae"].mean())
    print("\n  MAE by local hour (SEK/kWh):")
    for block_start in range(0, 24, 6):
        cells = "  ".join(f"{h:02d}:{by_hour[h]:.3f}" for h in range(block_start, block_start + 6))
        print(f"    {cells}")


def _plot(results: pd.DataFrame) -> None:
    """Two-panel figure: a sample fortnight of the curve, and the mean daily shape."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping plot)")
        return

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8))

    # (a) First 14 days of the holdout, hourly.
    start = results["timestamp"].min()
    window = results[results["timestamp"] < start + pd.Timedelta(days=14)]
    ax_top.plot(window["timestamp"], window["actual"], label="Actual", linewidth=1.4)
    ax_top.plot(window["timestamp"], window["predicted"], label="Model", linewidth=1.2)
    ax_top.plot(window["timestamp"], window["baseline"], label="Baseline (today's hour)",
                linewidth=0.9, alpha=0.6, linestyle="--")
    ax_top.set_title("Hourly day-ahead price: actual vs. predicted (first 14 held-out days)")
    ax_top.set_ylabel("Price (SEK/kWh)")
    ax_top.legend(loc="upper right")

    # (b) Average price by hour of day over the whole holdout (the duck curve).
    shape = results.groupby("hour")[["actual", "predicted", "baseline"]].mean()
    ax_bot.plot(shape.index, shape["actual"], marker="o", label="Actual", linewidth=1.6)
    ax_bot.plot(shape.index, shape["predicted"], marker="o", label="Model", linewidth=1.4)
    ax_bot.set_title("Average daily price shape over the held-out period")
    ax_bot.set_xlabel("Local hour of day")
    ax_bot.set_ylabel("Mean price (SEK/kWh)")
    ax_bot.set_xticks(range(0, 24, 2))
    ax_bot.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\n  Plot saved to {PLOT_PATH}")


# --- live training / prediction ---

def _observed_today_hours(stations: dict, parameter: int, value_column: str,
                          day: date) -> dict:
    """
    Zone-mean observed value for each local hour of `day` so far, from the
    stations' latest-months feed: {hour: mean}. Cached CSVs older than ~3
    hours get re-downloaded (the feed itself lags about an hour). Hours no
    station has reported yet aren't in the dict.
    """
    per_hour: dict[int, list[float]] = {}
    for sid in stations:
        path = download_station_latest_months(parameter, str(sid), max_age_days=0.125)
        if path is None:
            continue
        try:
            obs = load_smhi_hourly(path, value_column, "value")
        except ValueError:
            continue  # one malformed station file shouldn't sink the prediction

        local = obs["timestamp_utc"].dt.tz_convert("Europe/Stockholm")
        mask = local.dt.date == day
        for hour, value in zip(local[mask].dt.hour, obs.loc[mask, "value"]):
            per_hour.setdefault(int(hour), []).append(float(value))
    return {hour: float(np.mean(vals)) for hour, vals in per_hour.items()}

def train(zone: str = "SE4", verbose: bool = True):
    """Fit the hourly model on all available data and save it to disk."""
    df = build_hourly_dataset(zone=zone, verbose=verbose)
    X, y = split_features_target(df)
    model = build_model()
    model.fit(X, y)
    path = model_path(zone)
    joblib.dump(model, path)
    if verbose:
        print(f"Hourly model saved to {path}")
    return model


def load_model(zone: str = "SE4"):
    path = model_path(zone)
    if not path.exists():
        raise FileNotFoundError(f"No hourly model at {path}. Run train() first.")
    return joblib.load(path)


def predict_tomorrow_hourly(model=None, zone: str = "SE4", verbose: bool = True) -> pd.DataFrame:
    """Tomorrow's 24 hourly prices from live data: hour, predicted (SEK/kWh)."""
    if model is None:
        model = load_model()

    today = date.today()
    tomorrow = today + timedelta(days=1)

    # Today's published hourly prices -> per-hour anchor + daily mean.
    tp = fetch_prices_for_day(today, zone=zone)
    local = tp["timestamp_utc"].dt.tz_convert("Europe/Stockholm")
    tp = tp.assign(hour=local.dt.hour.astype(int))
    price_today_by_hour = tp.groupby("hour")["price_sek_per_kwh"].mean().to_dict()
    price_today_daily_mean = float(tp["price_sek_per_kwh"].mean())

    # Zone-mean wind. Tomorrow's curve comes straight from the forecast.
    # Today's daily mean = observed hours so far, topped up with the forecast
    # for the rest of the day. The forecast alone starts at "now", and a mean
    # over 13:00-23:00 is not a daily mean - training used full days.
    wind = regional_hourly_forecast(WIND_STATIONS, "wind_speed", [today, tomorrow])
    wind_today = {**wind[today], **_observed_today_hours(
        WIND_STATIONS, WIND_PARAM, "Vindhastighet", today)}
    wind_tomorrow = wind[tomorrow]
    if not (wind_today and wind_tomorrow):
        raise RuntimeError(
            "SMHI returned no usable wind data for today and/or tomorrow; "
            "cannot build prediction features."
        )
    wind_today_mean = float(np.mean(list(wind_today.values())))

    cal = build_calendar_features(pd.Series([tomorrow])).iloc[0]
    doy = tomorrow.timetuple().tm_yday

    rows = []
    for hour in sorted(set(wind_tomorrow) & set(price_today_by_hour)):
        rows.append({
            "hour": hour,
            "price_today_same_hour": price_today_by_hour[hour],
            "price_today_daily_mean": price_today_daily_mean,
            "wind_today_mean": wind_today_mean,
            "wind_tomorrow_hour": wind_tomorrow[hour],
            "wind_delta_hour": wind_tomorrow[hour] - wind_today_mean,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_week_tomorrow": int(cal["day_of_week"]),
            "is_weekend_tomorrow": int(cal["is_weekend"]),
            "is_holiday_tomorrow": int(cal["is_holiday"]),
            "day_of_year_sin": np.sin(2 * np.pi * doy / 365),
            "day_of_year_cos": np.cos(2 * np.pi * doy / 365),
            "month": tomorrow.month,
        })

    if not rows:
        raise RuntimeError(
            "No hour is covered by today's prices and tomorrow's forecast at "
            "the same time; nothing to predict."
        )
    feat = pd.DataFrame(rows)
    feat["predicted"] = model.predict(feat[FEATURE_COLUMNS])
    out = feat[["hour", "predicted"]].copy()

    if verbose:
        _print_curve(tomorrow, out, price_today_daily_mean)
    return out


def _print_curve(day: date, out: pd.DataFrame, today_mean: float) -> None:
    peak = out.loc[out["predicted"].idxmax()]
    trough = out.loc[out["predicted"].idxmin()]
    print(f"\nPredicted hourly prices for {day} (SEK/kWh):")
    for _, r in out.iterrows():
        bar = "#" * min(48, int(max(0, r["predicted"]) / 0.05))  # cap spike days
        print(f"  {int(r['hour']):02d}:00  {r['predicted']:6.3f}  {bar}")
    print(f"\n  Daily mean    : {out['predicted'].mean():.3f}  (today {today_mean:.3f})")
    print(f"  Cheapest      : {int(trough['hour']):02d}:00 @ {trough['predicted']:.3f}")
    print(f"  Most expensive: {int(peak['hour']):02d}:00 @ {peak['predicted']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Hourly day-ahead price model")
    parser.add_argument("--predict", action="store_true",
                        help="Train (if needed) and predict tomorrow's 24 hourly prices.")
    parser.add_argument("--retrain", action="store_true", help="Force retrain before predicting.")
    parser.add_argument("--months", type=int, default=4, help="Months to hold out (default 4).")
    parser.add_argument("--zone", default="SE4", help="Price zone (default SE4).")
    args = parser.parse_args()

    if args.predict:
        needs_training = args.retrain or not model_path(args.zone).exists()
        model = train(zone=args.zone) if needs_training else load_model(args.zone)
        predict_tomorrow_hourly(model=model, zone=args.zone)
    else:
        holdout_evaluation(zone=args.zone, holdout_months=args.months)


if __name__ == "__main__":
    main()
