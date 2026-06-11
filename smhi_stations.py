"""Download and cache the curated stations' history from the SMHI metobs API.

Two periods per station: corrected-archive is the quality-controlled history
but SMHI cuts it off ~3 months before today ("utom de senaste 3 mån"), and
latest-months covers the last ~4 months as mostly preliminary data. Stitched
together they reach the present. Nothing ships with the repo, it all lands
in cache/smhi_stations/.

API docs: https://opendata.smhi.se/apidocs/metobs/
"""

import time
from pathlib import Path

import requests

from historical_data import TEMPERATURE_STATIONS, WIND_STATIONS

METOBS_BASE = "https://opendata-download-metobs.smhi.se/api/version/latest"

TEMPERATURE_PARAM = 1   # Lufttemperatur, momentanvärde 1/tim
WIND_PARAM = 4          # Vindhastighet, medelvärde 10 min 1/tim

STATION_CACHE_DIR = Path("cache") / "smhi_stations"

# the archive cutoff only moves ~monthly; latest-months gets new data hourly
ARCHIVE_MAX_AGE_DAYS = 30
LATEST_MAX_AGE_DAYS = 1


def _to_station_list(station_dict: dict) -> list[dict]:
    """{id: (name, lat, lon)} -> [{id, name, lat, lon}]."""
    return [
        {"id": str(sid), "name": name, "lat": lat, "lon": lon}
        for sid, (name, lat, lon) in station_dict.items()
    ]


def curated_temperature_stations() -> list[dict]:
    return _to_station_list(TEMPERATURE_STATIONS)


def curated_wind_stations() -> list[dict]:
    return _to_station_list(WIND_STATIONS)


def _download_period(parameter: int, station_id: str, period: str,
                     dest: Path, max_age_days: float) -> Path | None:
    """
    Fetch one period CSV into `dest`, unless a cached copy younger than
    `max_age_days` is already there. If a re-download fails, the stale file
    is kept. None = nothing usable on disk and the download failed too.
    """
    STATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        age_days = (time.time() - dest.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return dest

    csv_url = (
        f"{METOBS_BASE}/parameter/{parameter}/station/{station_id}"
        f"/period/{period}/data.csv"
    )
    try:
        resp = requests.get(csv_url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException:
        return dest if dest.exists() else None

    dest.write_bytes(resp.content)
    return dest


def download_station_archive(parameter: int, station_id: str) -> Path | None:
    """The station's corrected-archive CSV (history up to ~3 months ago)."""
    dest = STATION_CACHE_DIR / f"{parameter}_{station_id}.csv"
    return _download_period(parameter, station_id, "corrected-archive",
                            dest, ARCHIVE_MAX_AGE_DAYS)


def download_station_latest_months(
    parameter: int, station_id: str,
    max_age_days: float = LATEST_MAX_AGE_DAYS,
) -> Path | None:
    """The station's latest-months CSV (the most recent ~4 months). Pass a
    smaller `max_age_days` when today's own hours matter (live prediction)."""
    dest = STATION_CACHE_DIR / f"{parameter}_{station_id}_latest.csv"
    return _download_period(parameter, station_id, "latest-months",
                            dest, max_age_days)
