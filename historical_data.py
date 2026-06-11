"""SE4 station registry + parser for the SMHI observation CSVs.

The CSVs are a bit awkward: a metadata block of varying length sits above
the data table (so the header row is found by content, not a fixed skip
count), each value column has its quality flag in the next column over, and
rows end with unnamed free-text columns. Hence the parsing below.

The live forecast is sampled at these same coordinates, so training and
prediction see the same locations.
"""

from pathlib import Path

import pandas as pd

# SMHI station number -> (name, lat, lon). Temperature stations near the big
# population centres (demand side), wind stations out on the coast by the
# wind farms (supply side).
TEMPERATURE_STATIONS = {
    52350: ("Malmö A", 55.5715, 13.0708),
    62410: ("Halmstad flygplats", 56.6833, 12.8167),
    64510: ("Växjö A", 56.8463, 14.8296),
    65090: ("Karlskrona-Söderstjerna", 56.1500, 15.5890),
    66420: ("Kalmar flygplats", 56.6784, 16.2922),
}

WIND_STATIONS = {
    52240: ("Falsterbo A", 55.3837, 12.8166),
    54290: ("Skillinge A", 55.4890, 14.3144),          # near Simrishamn
    62260: ("Hallands Väderö A", 56.4496, 12.5453),
    65090: ("Karlskrona-Söderstjerna", 56.1500, 15.5890),
    66420: ("Kalmar flygplats", 56.6784, 16.2922),
}

# G = controlled & approved, Y = rough-checked preliminary (most of the
# latest-months feed). Anything else, e.g. R for rejected, is dropped.
ACCEPTED_QUALITY = ("G", "Y")


def _find_data_header_row(csv_path: Path) -> int:
    """Line number of the 'Datum;Tid (UTC);...' row that starts the data."""
    with open(csv_path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if line.startswith("Datum;Tid"):
                return i
    raise ValueError(f"Could not find data header in {csv_path}")


def load_smhi_hourly(
    csv_path: str | Path,
    value_column: str,
    out_column: str,
    accepted_quality: tuple[str, ...] = ACCEPTED_QUALITY,
) -> pd.DataFrame:
    """
    One value series from an SMHI CSV, as [timestamp_utc, <out_column>].

    `value_column` is the Swedish column to extract ("Lufttemperatur",
    "Vindhastighet"), `out_column` what to call it in the result.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        sep=";",
        skiprows=_find_data_header_row(csv_path),
        encoding="utf-8-sig",
        low_memory=False,  # trailing free-text column otherwise warns on mixed types
    )

    columns = list(df.columns)
    if value_column not in columns:
        raise ValueError(f"{value_column!r} not found in {csv_path.name}; got {columns}")

    # The quality flag sits immediately right of the value column.
    quality_column = columns[columns.index(value_column) + 1]

    df = df[["Datum", "Tid (UTC)", value_column, quality_column]].copy()
    df.columns = ["date", "time", out_column, "quality"]

    df = df[df["quality"].isin(accepted_quality)]
    df = df.dropna(subset=["date", "time", out_column])

    df["timestamp_utc"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S", utc=True
    )
    df[out_column] = pd.to_numeric(df[out_column], errors="coerce")

    return (
        df[["timestamp_utc", out_column]]
        .dropna()
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
