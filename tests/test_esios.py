"""Valors d'ESIOS: períodes demanats, canvis d'hora, províncies, hores absents i previsions revisades."""
import datetime

import pandas as pd
import pytest

from epf.config.market import MARKET_TIMEZONE
from epf.data.esios import MONTHLY, PRICE, REVISED_FORECASTS, SERIES, hourly, load, periods

# 2024-10-27, demanda real (1293): la primera 02:00 porta `datetime_utc` 01:00Z,
# que és la de la segona.
AUTUMN = [
    {"datetime": "2024-10-27T00:00:00.000+02:00", "datetime_utc": "2024-10-26T22:00:00Z", "value": 21063.17},
    {"datetime": "2024-10-27T01:00:00.000+02:00", "datetime_utc": "2024-10-26T23:00:00Z", "value": 19838.75},
    {"datetime": "2024-10-27T02:00:00.000+02:00", "datetime_utc": "2024-10-27T01:00:00Z", "value": 18956.5},
    {"datetime": "2024-10-27T02:00:00.000+01:00", "datetime_utc": "2024-10-27T01:00:00Z", "value": 18397.92},
    {"datetime": "2024-10-27T03:00:00.000+01:00", "datetime_utc": "2024-10-27T02:00:00Z", "value": 18044.67},
]

# 2025-03-30, demanda real (1293): de la 01:00 es passa a la 03:00.
SPRING = [
    {"datetime": "2025-03-30T00:00:00.000+01:00", "value": 22541.08},
    {"datetime": "2025-03-30T01:00:00.000+01:00", "value": 21072.33},
    {"datetime": "2025-03-30T03:00:00.000+02:00", "value": 20130.58},
]

# 2024-03-09, potència nuclear disponible (474): València deixa de publicar-se a les 02:00.
NUCLEAR = [
    {"datetime": "2024-03-09T01:00:00.000+01:00", "value": 1063.9, "geo_name": "Valencia"},
    {"datetime": "2024-03-09T01:00:00.000+01:00", "value": 1005.8, "geo_name": "Cáceres"},
    {"datetime": "2024-03-09T01:00:00.000+01:00", "value": 3032.8, "geo_name": "Tarragona"},
    {"datetime": "2024-03-09T01:00:00.000+01:00", "value": 1003.4, "geo_name": "Guadalajara"},
    {"datetime": "2024-03-09T02:00:00.000+01:00", "value": 1005.8, "geo_name": "Cáceres"},
    {"datetime": "2024-03-09T02:00:00.000+01:00", "value": 3032.8, "geo_name": "Tarragona"},
    {"datetime": "2024-03-09T02:00:00.000+01:00", "value": 1003.4, "geo_name": "Guadalajara"},
]


def rest_of_day(day, known, value=0.0):
    """Completa amb `value` les hores locals del dia que no són a `known`."""
    start = pd.Timestamp(day).tz_localize(MARKET_TIMEZONE)
    slots = pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left")
    taken = set(pd.to_datetime([v["datetime"] for v in known], format="ISO8601", utc=True))
    return known + [{"datetime": slot.isoformat(), "value": value}
                    for slot in slots if slot.tz_convert("UTC") not in taken]


def test_autumn_keys_on_the_offset_and_merges_the_repeated_hour():
    day = hourly(rest_of_day(datetime.date(2024, 10, 27), AUTUMN))[0].iloc[0]
    assert list(day[:4]) == [21063.17, 19838.75, pytest.approx((18956.5 + 18397.92) / 2), 18044.67]


def test_spring_inserts_the_missing_hour():
    day = hourly(rest_of_day(datetime.date(2025, 3, 30), SPRING))[0].iloc[0]
    assert list(day[:4]) == [22541.08, 21072.33, pytest.approx((21072.33 + 20130.58) / 2), 20130.58]


def test_provinces_are_summed_and_an_absent_province_adds_nothing():
    day = hourly(NUCLEAR, anchor_days={datetime.date(2024, 3, 9)})[0].iloc[0]
    assert day["h01"] == pytest.approx(1063.9 + 1005.8 + 3032.8 + 1003.4)
    assert day["h02"] == pytest.approx(1005.8 + 3032.8 + 1003.4)


def test_an_absent_hour_is_zero_only_on_an_anchor_day():
    day = datetime.date(2024, 3, 9)
    assert hourly(NUCLEAR, anchor_days={day})[0].iloc[0]["h05"] == 0.0
    assert pd.isna(hourly(NUCLEAR, anchor_days=set())[0].iloc[0]["h05"])
    assert pd.isna(hourly(NUCLEAR)[0].iloc[0]["h05"])


def test_zeros_never_extend_beyond_the_published_days():
    anchors = {datetime.date(2024, 3, 8), datetime.date(2024, 3, 9), datetime.date(2024, 3, 10)}
    days = hourly(NUCLEAR, anchor_days=anchors)[0].index
    assert list(days) == [pd.Timestamp("2024-03-09")]


def test_the_published_count_is_of_local_hours_and_not_of_repaired_ones():
    autumn, autumn_hours = hourly(rest_of_day(datetime.date(2024, 10, 27), AUTUMN))
    spring, spring_hours = hourly(rest_of_day(datetime.date(2025, 3, 30), SPRING))
    assert autumn_hours.iloc[0] == 25 and len(autumn.columns) == 24
    assert spring_hours.iloc[0] == 23 and len(spring.columns) == 24


def test_revised_forecasts_are_refused_by_the_normal_loader():
    for indicator in REVISED_FORECASTS:
        with pytest.raises(ValueError):
            load(indicator, directory="/nonexistent")


def test_every_anchor_is_a_series_that_cannot_be_zero():
    for anchor in {a for a in SERIES.values() if a is not None}:
        assert SERIES[anchor] is None


def test_revised_forecasts_and_prices_are_not_admissible_series():
    assert not REVISED_FORECASTS & set(SERIES)
    assert PRICE not in SERIES


def test_monthly_periods_cover_every_day_once_to_the_end_of_this_year():
    today = datetime.date(2024, 3, 9)
    days = [(first, last) for _, first, last in periods(min(MONTHLY), today)]
    assert days[0][0] == datetime.date(2010, 1, 1)
    assert days[-1] == (datetime.date(2024, 12, 1), datetime.date(2024, 12, 31))
    assert all(last + datetime.timedelta(days=1) == following
               for (_, last), (following, _) in zip(days, days[1:]))
