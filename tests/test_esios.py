import datetime

import pandas as pd
import pytest

from epf.dataset.esios import (
    MARKET_TIMEZONE,
    hourly,
    repair_dst,
)

# 2024-10-27, demanda real (1293): la primera 02:00 porta `datetime_utc` 01:00Z, que és
# la de la segona.
AUTUMN = [
    {
        "datetime": "2024-10-27T00:00:00.000+02:00",
        "datetime_utc": "2024-10-26T22:00:00Z",
        "value": 21063.17,
    },
    {
        "datetime": "2024-10-27T01:00:00.000+02:00",
        "datetime_utc": "2024-10-26T23:00:00Z",
        "value": 19838.75,
    },
    {
        "datetime": "2024-10-27T02:00:00.000+02:00",
        "datetime_utc": "2024-10-27T01:00:00Z",
        "value": 18956.5,
    },
    {
        "datetime": "2024-10-27T02:00:00.000+01:00",
        "datetime_utc": "2024-10-27T01:00:00Z",
        "value": 18397.92,
    },
    {
        "datetime": "2024-10-27T03:00:00.000+01:00",
        "datetime_utc": "2024-10-27T02:00:00Z",
        "value": 18044.67,
    },
]

# 2025-03-30, demanda real (1293): de la 01:00 es passa a la 03:00.
SPRING = [
    {"datetime": "2025-03-30T00:00:00.000+01:00", "value": 22541.08},
    {"datetime": "2025-03-30T01:00:00.000+01:00", "value": 21072.33},
    {"datetime": "2025-03-30T03:00:00.000+02:00", "value": 20130.58},
]

# 2025-10-01, preu espanyol (600) demanat sense `time_trunc`, que ESIOS serveix per
# quarts d'hora.
QUARTERS = [
    {"datetime": "2025-10-01T00:00:00.000+02:00", "value": 105.1},
    {"datetime": "2025-10-01T00:15:00.000+02:00", "value": 104.24},
    {"datetime": "2025-10-01T00:30:00.000+02:00", "value": 102.28},
    {"datetime": "2025-10-01T00:45:00.000+02:00", "value": 102.0},
]

# 2024-03-09, potència nuclear disponible a València (474): l'últim valor és el de la
# 01:00, i a les 02:00 deixa de publicar-se, com en una parada de recàrrega.
NUCLEAR = [{"datetime": "2024-03-09T01:00:00.000+01:00", "value": 1063.9}]


def hours_of(*days):
    """Torna en UTC les hores locals de `days`, les que publica una àncora."""
    return {
        instant.tz_convert("UTC")
        for day in days
        for instant in pd.date_range(
            pd.Timestamp(day).tz_localize(MARKET_TIMEZONE), periods=24, freq="h"
        )
    }


def rest_of_day(day, known, value=0.0):
    """Completa amb `value` les hores locals del dia que no són a `known`."""
    start = pd.Timestamp(day).tz_localize(MARKET_TIMEZONE)
    slots = pd.date_range(
        start, start + pd.DateOffset(days=1), freq="h", inclusive="left"
    )
    taken = set(
        pd.to_datetime([v["datetime"] for v in known], format="ISO8601", utc=True)
    )
    return known + [
        {"datetime": slot.isoformat(), "value": value}
        for slot in slots
        if slot.tz_convert("UTC") not in taken
    ]


def test_autumn_keys_on_the_offset_and_merges_the_repeated_hour():
    day = hourly(rest_of_day(datetime.date(2024, 10, 27), AUTUMN)).iloc[0]
    assert list(day[:4]) == [
        21063.17,
        19838.75,
        pytest.approx((18956.5 + 18397.92) / 2),
        18044.67,
    ]


def test_spring_inserts_the_missing_hour():
    day = hourly(rest_of_day(datetime.date(2025, 3, 30), SPRING)).iloc[0]
    assert list(day[:4]) == [
        22541.08,
        21072.33,
        pytest.approx((21072.33 + 20130.58) / 2),
        20130.58,
    ]


def test_a_value_off_the_hour_is_refused_rather_than_dropped():
    with pytest.raises(ValueError, match="fora de l'hora"):
        hourly(QUARTERS)


def test_two_values_at_one_instant_are_refused_rather_than_summed():
    repeated = NUCLEAR + [
        {"datetime": "2024-03-09T01:00:00.000+01:00", "value": 1063.9}
    ]
    with pytest.raises(ValueError, match="més d'un valor"):
        hourly(repeated)


def test_an_absent_hour_is_zero_only_where_the_anchor_published_it():
    """Les hores de l'apagada, que l'àncora no va publicar, valien zero al programa."""
    anchor = hours_of("2024-03-09")
    assert hourly(NUCLEAR, anchor).iloc[0]["h05"] == 0.0
    morning = {
        instant for instant in anchor if instant.tz_convert(MARKET_TIMEZONE).hour < 4
    }
    assert pd.isna(hourly(NUCLEAR, morning).iloc[0]["h05"])
    assert pd.isna(hourly(NUCLEAR, set()).iloc[0]["h05"])
    assert pd.isna(hourly(NUCLEAR).iloc[0]["h05"])


def test_past_its_last_day_a_series_is_zero_where_the_anchor_published_and_not_before():
    """El carbó del programa va deixar de publicar-se mentre l'àncora seguia."""
    anchor = hours_of("2024-03-08", "2024-03-09", "2024-03-10")
    frame = hourly(NUCLEAR, anchor)
    assert list(frame.index) == [pd.Timestamp("2024-03-09"), pd.Timestamp("2024-03-10")]
    assert (frame.loc["2024-03-10"] == 0.0).all()


# 2025-03-30: 23 períodes, sense l'hora local 02. Primers cinc preus espanyols.
SPRING_PRICES = [3.6, 3.2, 0.65, 0.1, 0.09]

# 2024-10-27: 25 períodes; el tercer i el quart són l'hora local 02 repetida.
AUTUMN_PRICES = [87.71, 83.31, 82.23, 80.68, 81.3]


def test_a_normal_day_is_untouched():
    hours = list(range(24))
    assert repair_dst(hours) == hours


def test_spring_inserts_the_mean_of_the_neighbours():
    repaired = repair_dst(SPRING_PRICES + [0.0] * 18)
    assert len(repaired) == 24
    assert repaired[:4] == [3.6, 3.2, pytest.approx((3.2 + 0.65) / 2), 0.65]


def test_autumn_merges_the_two_auctions_of_the_repeated_hour():
    repaired = repair_dst(AUTUMN_PRICES + [0.0] * 20)
    assert len(repaired) == 24
    assert repaired[:4] == [87.71, 83.31, pytest.approx((82.23 + 80.68) / 2), 81.3]


def test_the_repair_only_moves_the_third_hour():
    day = [float(h) for h in range(23)]
    repaired = repair_dst(day)
    assert repaired[:2] == day[:2]
    assert repaired[3:] == day[2:]


def test_an_unknown_length_fails_loudly():
    with pytest.raises(ValueError):
        repair_dst([0.0] * 20)
