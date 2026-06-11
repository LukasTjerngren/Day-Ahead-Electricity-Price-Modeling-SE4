"""On-disk cache of fetched electricity prices, one CSV per zone.

A cold fetch of four-plus years takes a few minutes, so downloaded days are
kept in cache/prices_<zone>.csv and later runs only request what's missing.
The API starts at 2021-11-01; anything earlier 404s.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from data_fetchers import fetch_prices_for_day

CACHE_DIR = Path("cache")
REQUEST_DELAY_SECONDS = 0.15  # be polite to the API between requests


def cache_path(zone: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"prices_{zone}.csv"


def load_cached_prices(zone: str = "SE4") -> pd.DataFrame:
    """Whatever is already on disk; empty DataFrame if nothing is."""
    path = cache_path(zone)
    if not path.exists():
        return pd.DataFrame(columns=["timestamp_utc", "price_sek_per_kwh", "zone"])

    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def save_cached_prices(df: pd.DataFrame, zone: str = "SE4"):
    df.to_csv(cache_path(zone), index=False)


def update_price_cache(
    start: date,
    end: date,
    zone: str = "SE4",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Make sure the cache holds every day from start to end (inclusive),
    fetching only the missing ones, and return the full cached frame.
    """
    cached = load_cached_prices(zone)

    # Which local dates are already covered?
    if cached.empty:
        cached_dates: set[date] = set()
    else:
        cached_dates = set(
            cached["timestamp_utc"]
            .dt.tz_convert("Europe/Stockholm")
            .dt.date
        )

    all_dates = []
    current = start
    while current <= end:
        all_dates.append(current)
        current += timedelta(days=1)

    missing = [d for d in all_dates if d not in cached_dates]

    if not missing:
        if verbose:
            print(f"  Cache is up to date ({len(all_dates)} days, zone {zone}).")
        return cached

    if verbose:
        print(f"  Fetching {len(missing)} missing day(s) for zone {zone} ...")

    new_frames = []
    for i, day in enumerate(missing):
        try:
            df = fetch_prices_for_day(day, zone)
            new_frames.append(df)
        except requests.HTTPError as exc:
            if verbose:
                print(f"    Skipping {day}: {exc}")
        except Exception as exc:
            if verbose:
                print(f"    Error on {day}: {exc}")

        if verbose and (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{len(missing)} fetched")

        time.sleep(REQUEST_DELAY_SECONDS)

    if new_frames:
        combined = pd.concat([cached] + new_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp_utc").sort_values("timestamp_utc")
        save_cached_prices(combined, zone)
        if verbose:
            print(f"  Cache updated: {len(combined):,} rows total.")
        return combined

    return cached
