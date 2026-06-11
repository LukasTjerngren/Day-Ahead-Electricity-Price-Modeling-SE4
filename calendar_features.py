"""Weekday and public-holiday features. Holidays and weekends shift
demand regardless of the weather: less industry running, lower load."""

from datetime import date, timedelta

import pandas as pd

# Fixed-date holidays; the names are just for reading.
FIXED_HOLIDAYS = {
    (1, 1): "Nyårsdagen",
    (1, 6): "Trettondedag jul",
    (5, 1): "Första maj",
    (6, 6): "Sveriges nationaldag",
    (12, 24): "Julafton",
    (12, 25): "Juldagen",
    (12, 26): "Annandag jul",
    (12, 31): "Nyårsafton",
}


def easter_sunday(year: int) -> date:
    """Easter Sunday via the anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _first_weekday_on_or_after(start: date, weekday: int) -> date:
    """First date >= start falling on the given weekday (0=Mon ... 6=Sun)."""
    return start + timedelta(days=(weekday - start.weekday()) % 7)


def swedish_holidays(year: int) -> set[date]:
    """All public holiday dates in the year, fixed and moveable."""
    days = {date(year, m, d) for (m, d) in FIXED_HOLIDAYS}

    e = easter_sunday(year)
    days |= {
        e - timedelta(days=2),    # Långfredag
        e,                        # Påskdagen
        e + timedelta(days=1),    # Annandag påsk
        e + timedelta(days=39),   # Kristi himmelsfärdsdag
        e + timedelta(days=49),   # Pingstdagen
    }

    midsummer_eve = _first_weekday_on_or_after(date(year, 6, 19), weekday=4)  # Friday
    days |= {midsummer_eve, midsummer_eve + timedelta(days=1)}
    days.add(_first_weekday_on_or_after(date(year, 10, 31), weekday=5))  # Alla helgons dag

    return days


def build_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """date, day_of_week (0=Mon), is_weekend, is_holiday for each input date."""
    days = pd.to_datetime(dates).dt.date
    holidays: set[date] = set()
    for year in {d.year for d in days}:
        holidays |= swedish_holidays(year)

    return pd.DataFrame({
        "date": list(days),
        "day_of_week": [d.weekday() for d in days],
        "is_weekend": [d.weekday() >= 5 for d in days],
        "is_holiday": [d in holidays for d in days],
    })
