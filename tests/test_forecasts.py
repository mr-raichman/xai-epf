"""El conjunt de les previsions: què el separa de la resta de sèries d'ESIOS."""
import json

import pytest

from epf.data import esios, forecasts


def test_the_forecasts_are_outside_the_generic_series_table():
    # Els dos llaços que recorren `SERIES` les buscarien a `downloads/esios/`, d'on
    # han marxat: l'auditoria les donaria per il·legibles, i el refresc a mà les
    # tornaria a baixar allà.
    assert not set(forecasts.INDICATORS.values()) & set(esios.SERIES)


def test_a_forecast_served_for_two_geographies_is_refused_rather_than_summed(tmp_path):
    # Una resposta d'ESIOS pot portar diverses geografies, com la del preu. Sumades,
    # Espanya i Portugal donarien una demanda ibèrica, plausible i falsa.
    folder = tmp_path / str(forecasts.INDICATORS["demand"])
    folder.mkdir()
    values = [{"datetime": "2024-03-05T00:00:00.000+01:00", "value": value, "geo_name": name}
              for value, name in ((25000.0, "España"), (5000.0, "Portugal"))]
    (folder / "2024.json").write_text(json.dumps({"indicator": {"values": values}}))
    with pytest.raises(ValueError, match="geografies"):
        forecasts.read("demand", tmp_path)
