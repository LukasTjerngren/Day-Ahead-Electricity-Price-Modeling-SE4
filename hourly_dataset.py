"""Builds the modelling table: one row per (day, hour).

A row is today's price at hour h plus tomorrow's weather at hour h, and the
target is tomorrow's price at that hour. No leakage: today's prices were
published yesterday afternoon and the rest is forecast/calendar, so every
feature is available on day D before the auction closes. Exact list in
FEATURE_COLUMNS, target is price_tomorrow_hour (SEK/kWh).

Prices are grouped by local (date, hour): the 15-minute records (2025+)
average up to hourly, and DST 23/25-hour days work out without special-
casing - a missing or doubled local hour just drops or averages.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from price_cache import update_price_cache
from calendar_features import build_calendar_features
from smhi_stations import (
    download_station_archive,
    download_station_latest_months,
    curated_temperature_stations,
    curated_wind_stations,
    TEMPERATURE_PARAM,
    WIND_PARAM,
)
from historical_data import load_smhi_hourly

# elprisetjustnu.se API has data from 2021-11-01 onwards; earlier dates return 404.
TRAINING_START = date(2021, 11, 1)

# The temp_* columns stay in the table for the EDA but are not model inputs:
# in the seasonal backtest (analysis.seasonal_backtest), dropping them cost
# +0.005..0.017 MAE in the winter/spring folds and gained more than that in
# the summers, netting ~zero. Season + the day-before price already carry
# the seasonal demand effect. See README.
TEMP_COLUMNS = ["temp_today_mean", "temp_tomorrow_hour", "temp_delta_hour"]

FEATURE_COLUMNS = [
    "price_today_same_hour", "price_today_daily_mean",
    "wind_today_mean", "wind_tomorrow_hour", "wind_delta_hour",
    "hour_sin", "hour_cos",
    "day_of_week_tomorrow", "is_weekend_tomorrow", "is_holiday_tomorrow",
    "day_of_year_sin", "day_of_year_cos", "month",
]
TARGET_COLUMN = "price_tomorrow_hour"
BOOL_FEATURES = ["is_weekend_tomorrow", "is_holiday_tomorrow"]


def _hourly_prices(zone: str, price_end: date, verbose: bool):
    """
    Return two frames from the price cache:
      hourly : date, hour, price_hour       (mean price per local hour)
      daily  : date, price_daily_mean       (mean price per local day)
    """
    prices = update_price_cache(TRAINING_START, price_end, zone=zone, verbose=verbose)
    local = prices["timestamp_utc"].dt.tz_convert("Europe/Stockholm")
    prices = prices.assign(
        date=pd.to_datetime(local.dt.date),
        hour=local.dt.hour.astype(int),
    )
    hourly = (
        prices.groupby(["date", "hour"])["price_sek_per_kwh"]
        .mean().rename("price_hour").reset_index()
    )
    daily = (
        prices.groupby("date")["price_sek_per_kwh"]
        .mean().rename("price_daily_mean").reset_index()
    )
    return hourly, daily


def _regional_hourly(stations, parameter, value_column, out_column) -> pd.DataFrame:
    """
    Hourly mean of one parameter across stations: date, hour, <out_column>.

    Per station, the corrected-archive (which SMHI cuts off ~3 months before
    today) is topped up with latest-months; the archive wins where they
    overlap. Plain unweighted mean across stations - population-weighting
    was tried and didn't help (see README).
    """
    per_station = []
    for s in stations:
        paths = [download_station_archive(parameter, s["id"]),
                 download_station_latest_months(parameter, s["id"])]
        frames = [load_smhi_hourly(p, value_column, out_column)
                  for p in paths if p is not None]
        frames = [f for f in frames if not f.empty]
        if not frames:
            continue
        hourly = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="timestamp_utc", keep="first")  # archive wins
            .sort_values("timestamp_utc")
        )
        local = hourly["timestamp_utc"].dt.tz_convert("Europe/Stockholm")
        tidy = pd.DataFrame({
            "date": pd.to_datetime(local.dt.date),
            "hour": local.dt.hour.astype(int),
            out_column: hourly[out_column].to_numpy(),
        })
        # per-station hourly mean, which also collapses sub-hourly readings
        per_station.append(
            tidy.groupby(["date", "hour"])[out_column].mean().rename(s["id"])
        )

    if not per_station:
        raise RuntimeError(
            f"No data for any station (parameter {parameter}); "
            "check the network connection and the cache directory."
        )
    wide = pd.concat(per_station, axis=1)
    return wide.mean(axis=1).rename(out_column).reset_index()


def build_hourly_dataset(
    zone: str = "SE4",
    end: date | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the long (day x hour) modelling table. Columns: date, hour, all of
    FEATURE_COLUMNS, the temp_* EDA columns, and TARGET_COLUMN.
    """
    if verbose:
        print("Fetching hourly temperature from the SMHI stations ...")
    temp_h = _regional_hourly(curated_temperature_stations(), TEMPERATURE_PARAM,
                              "Lufttemperatur", "temp_hour")
    if verbose:
        print("Fetching hourly wind from the SMHI stations ...")
    wind_h = _regional_hourly(curated_wind_stations(), WIND_PARAM,
                              "Vindhastighet", "wind_hour")

    # Daily means come from the same hourly frames so they can't disagree.
    temp_daily = temp_h.groupby("date")["temp_hour"].mean().rename("temp_today_mean").reset_index()
    wind_daily = wind_h.groupby("date")["wind_hour"].mean().rename("wind_today_mean").reset_index()

    weather_end = min(temp_h["date"].max(), wind_h["date"].max()).date()
    if end is None:
        end = weather_end - timedelta(days=1)
    price_end = end + timedelta(days=1)

    if verbose:
        print(f"Ensuring price cache covers {TRAINING_START} to {price_end} ...")
    price_h, price_daily = _hourly_prices(zone, price_end, verbose)

    # --- "Today" grid: one row per (date, hour) ---
    today = (
        price_h.rename(columns={"price_hour": "price_today_same_hour"})
        .merge(price_daily.rename(columns={"price_daily_mean": "price_today_daily_mean"}), on="date")
        .merge(temp_daily, on="date")
        .merge(wind_daily, on="date")
    )

    # --- "Tomorrow" pieces, keyed back onto today's date (join_date = date - 1) ---
    def _shift_back(df, col, new):
        out = df.rename(columns={col: new}).copy()
        out["join_date"] = out["date"] - pd.Timedelta(days=1)
        return out[["join_date", "hour", new]]

    tom_price = _shift_back(price_h, "price_hour", "price_tomorrow_hour")
    tom_temp = _shift_back(temp_h, "temp_hour", "temp_tomorrow_hour")
    tom_wind = _shift_back(wind_h, "wind_hour", "wind_tomorrow_hour")

    df = (
        today
        .merge(tom_price, left_on=["date", "hour"], right_on=["join_date", "hour"]).drop(columns="join_date")
        .merge(tom_temp, left_on=["date", "hour"], right_on=["join_date", "hour"]).drop(columns="join_date")
        .merge(tom_wind, left_on=["date", "hour"], right_on=["join_date", "hour"]).drop(columns="join_date")
    )

    # --- Engineered features ---
    df["temp_delta_hour"] = df["temp_tomorrow_hour"] - df["temp_today_mean"]
    df["wind_delta_hour"] = df["wind_tomorrow_hour"] - df["wind_today_mean"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- Calendar / season for tomorrow's date ---
    df["tomorrow"] = df["date"] + pd.Timedelta(days=1)
    cal = build_calendar_features(df["tomorrow"].drop_duplicates())
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal[["date", "day_of_week", "is_weekend", "is_holiday"]].rename(columns={
        "day_of_week": "day_of_week_tomorrow",
        "is_weekend": "is_weekend_tomorrow",
        "is_holiday": "is_holiday_tomorrow",
    })
    df = df.merge(cal, left_on="tomorrow", right_on="date", suffixes=("", "_cal")).drop(columns="date_cal")

    doy = df["tomorrow"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365)
    df["month"] = df["tomorrow"].dt.month

    # the price cache returns everything it holds, which can run past a
    # caller-supplied `end` - trim back
    df = df[df["date"] <= pd.Timestamp(end)]
    df = df.drop(columns="tomorrow").sort_values(["date", "hour"]).reset_index(drop=True)

    if verbose:
        print(f"Hourly dataset ready: {len(df):,} (day x hour) rows, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")
    return df


def split_features_target(df: pd.DataFrame):
    """Split the hourly dataset into (X, y)."""
    X = df[FEATURE_COLUMNS].copy()
    for col in BOOL_FEATURES:
        X[col] = X[col].astype(int)
    y = df[TARGET_COLUMN]
    return X, y
