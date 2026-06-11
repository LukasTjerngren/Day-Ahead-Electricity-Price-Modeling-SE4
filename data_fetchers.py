"""Clients for the two live data sources: elprisetjustnu.se for hourly spot
prices, and the SMHI SNOW API for point weather forecasts."""

from datetime import date

import requests
import numpy as np
import pandas as pd

PRICE_API_BASE = "https://www.elprisetjustnu.se/api/v1/prices"
PRICE_ZONES = {"SE1", "SE2", "SE3", "SE4"}


def fetch_prices_for_day(target_date: date, zone: str = "SE4") -> pd.DataFrame:
    """Spot prices for one day: timestamp_utc, price_sek_per_kwh, zone."""
    if zone not in PRICE_ZONES:
        raise ValueError(f"zone must be one of {PRICE_ZONES}, got {zone!r}")

    url = (
        f"{PRICE_API_BASE}/{target_date.year}"
        f"/{target_date.month:02d}-{target_date.day:02d}_{zone}.json"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    records = response.json()

    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([r["time_start"] for r in records], utc=True),
        "price_sek_per_kwh": [r["SEK_per_kWh"] for r in records],
        "zone": zone,
    })
    return df.sort_values("timestamp_utc").reset_index(drop=True)


SMHI_FORECAST_BASE = (
    "https://opendata-download-metfcst.smhi.se"
    "/api/category/snow1g/version/1/geotype/point"
)


def fetch_smhi_forecast(lat: float, lon: float,
                        parameters: list[str] | None = None) -> pd.DataFrame:
    """
    Current SMHI SNOW forecast for a point. Covers 240+ hours, hourly for the
    first few days. Returns valid_time (UTC) plus one column per kept
    parameter (all of them if `parameters` is None).

    NB: the API has a ?parameters= query filter but it's broken from here:
    requests encodes the comma as %2C, SMHI doesn't decode it, and you
    silently get only the first parameter back. So fetch everything and
    subset locally.
    """
    url = f"{SMHI_FORECAST_BASE}/lon/{lon}/lat/{lat}/data.json"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    rows = []
    for step in response.json()["timeSeries"]:
        row = {"valid_time": pd.to_datetime(step["time"], utc=True)}
        row.update(step["data"])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("valid_time").reset_index(drop=True)
    if parameters is not None:
        df = df[["valid_time"] + [p for p in parameters if p in df.columns]]
    return df


def regional_hourly_forecast(stations: dict, value_param: str,
                             days: list[date]) -> dict:
    """
    Forecast for the given local days, averaged across stations per hour.

    `stations` is a {id: (name, lat, lon)} dict like TEMPERATURE_STATIONS and
    `value_param` e.g. "air_temperature" or "wind_speed". Each station is
    fetched once. Returns {day: {hour: mean}}. Hours the forecast doesn't
    cover (mostly: hours of today that already passed) aren't in the dict.
    """
    per_day: dict = {day: {} for day in days}
    for _name, lat, lon in stations.values():
        fc = fetch_smhi_forecast(lat, lon, parameters=[value_param])
        if value_param not in fc.columns:
            continue
        local = fc["valid_time"].dt.tz_convert("Europe/Stockholm")
        for day in days:
            mask = local.dt.date == day
            for hour, value in zip(local[mask].dt.hour, fc.loc[mask, value_param]):
                per_day[day].setdefault(int(hour), []).append(float(value))

    return {
        day: {hour: float(np.mean(vals)) for hour, vals in hours.items()}
        for day, hours in per_day.items()
    }
