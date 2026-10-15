"""El conjunt de dades: cada sèrie és la suma dels seus indicadors, cada indicador del
programa es demana com l'energia de l'hora, una previsió amb una altra geografia es refusa,
i el preu canvia de font al tall.

Cap d'aquestes coses no peta quan falla. Una sèrie que perdés un indicador seguiria tenint
la forma d'una sèrie, i un indicador del programa demanat com la mitjana de l'hora cauria a
la quarta part a partir del 2025-10-01, un valor equivocat i una data alhora.
"""
import pandas as pd
import pytest

from epf.dataset import build, esios, omie
from epf.parameters import HOUR_COLUMNS

DAYS = pd.date_range("2024-03-04", periods=3, freq="D", name="day")


def test_each_series_is_the_sum_of_its_indicators():
    # Cada indicador val el seu número d'ESIOS a totes les hores.
    frames = {name: pd.DataFrame(float(indicator.id), index=DAYS, columns=HOUR_COLUMNS)
              for name, indicator in build.INDICATORS.items()}
    found = build.series(frames)
    assert (found["nuclear_sched"] == 4.0).all().all()
    assert (found["coal_gas_sched"] == 7.0 + 8.0 + 9.0 + 11.0 + 17.0).all().all()
    assert (found["hydro_sched"] == 1.0 + 2.0 + 3.0).all().all()


def test_every_schedule_indicator_is_asked_for_the_hours_sum(monkeypatch):
    # Un indicador del programa es diu pel seu tipus al darrere, `_sched`.
    asked = {}
    monkeypatch.setattr(esios, "session_for", lambda: None)
    monkeypatch.setattr(omie, "download", lambda directory: None)
    monkeypatch.setattr(esios, "download", lambda indicator, folder, session:
                        asked.__setitem__(folder.name, indicator.aggregation))
    build.download()
    assert set(asked) == set(build.INDICATORS)
    assert {name for name, aggregation in asked.items() if aggregation == "sum"} == {
        name for name in asked if name.endswith("_sched")}


def test_a_forecast_served_for_another_geography_too_is_refused():
    # Una resposta d'ESIOS pot portar diverses geografies, com la del preu. Una previsió de
    # la Península que arribés amb Portugal al costat ja no seria la que diu la taula.
    values = [{"datetime": "2024-03-05T00:00:00.000+01:00", "value": value, "geo_name": name}
              for value, name in ((25000.0, "Península"), (5000.0, "Portugal"))]
    with pytest.raises(ValueError, match="geografies"):
        esios.admissible(values, build.INDICATORS["demand_fcst"])


def test_the_two_sources_meet_at_the_seam_without_overlapping():
    # El tall és el primer dia d'OMIE: ESIOS fins al dia abans, OMIE des d'aquell.
    seam = pd.Timestamp(omie.OMIE_FROM)
    assert omie.ESIOS_FIRST_DAY < omie.OMIE_FROM
    assert seam.date() == omie.OMIE_FROM
