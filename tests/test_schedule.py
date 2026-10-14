"""El conjunt del programa: un grup és la suma dels seus indicadors, i cada sèrie es demana
com l'energia de l'hora.

Cap de les dues coses no peta quan falla. Un grup que perdés un indicador seguiria tenint
la forma d'un grup, i una sèrie demanada com la mitjana de l'hora cauria a la quarta part
a partir del 2025-10-01, un valor equivocat i una data alhora.
"""
import json

import pandas as pd

from epf.data import esios, schedule
from epf.market import MARKET_TIMEZONE

DAYS = pd.date_range("2024-03-04", periods=3, freq="D")


def write(directory, indicator, hours):
    """Un fitxer d'ESIOS d'un any amb un valor per hora, `hours` de `(instant, valor)`."""
    folder = directory / str(indicator)
    folder.mkdir(parents=True)
    values = [{"datetime": instant.isoformat(), "value": value, "geo_name": "Península"}
              for instant, value in hours]
    (folder / "2024.json").write_text(json.dumps({"indicator": {"values": values}}))


def every_hour(days=DAYS):
    return pd.date_range(days[0].tz_localize(MARKET_TIMEZONE), periods=24 * len(days),
                         freq="h")


def test_each_group_is_the_sum_of_its_indicators(tmp_path):
    hours = every_hour()
    write(tmp_path, schedule.ANCHOR, [(instant, 5000.0) for instant in hours])
    for indicator in (one for group in schedule.GROUPS.values() for one in group):
        # Cada indicador val el seu número, i el 8 deixa de publicar-se el segon dia, com
        # el subbituminós del 2025, de manera que hi ha de valer zero.
        published = hours[:24] if indicator == 8 else hours
        write(tmp_path, indicator, [(instant, float(indicator)) for instant in published])
    groups = schedule.load(tmp_path)
    assert (groups["nuclear"] == 4.0).all().all()
    assert (groups["hydro"] == 1.0 + 2.0 + 3.0).all().all()
    coal_gas = groups["coal_gas"]
    assert (coal_gas.loc[DAYS[0]] == 7.0 + 8.0 + 9.0 + 11.0 + 17.0).all()
    assert (coal_gas.loc[DAYS[1:]] == 7.0 + 9.0 + 11.0 + 17.0).all().all()


def test_every_series_is_asked_for_the_hours_sum(monkeypatch):
    asked = {}
    monkeypatch.setattr(esios, "session_for", lambda: None)
    monkeypatch.setattr(esios, "download", lambda indicator, folder, session, aggregation:
                        asked.__setitem__(indicator, aggregation))
    schedule.download()
    grouped = {one for group in schedule.GROUPS.values() for one in group}
    assert set(asked) == grouped | {schedule.ANCHOR}
    assert set(asked.values()) == {"sum"}
