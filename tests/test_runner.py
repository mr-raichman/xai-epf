"""Que el registre d'una execució es pugui escriure.

`described` escombra tot el que `parameters.py` té en majúscules, de manera que un
paràmetre nou hi entra sol i no se'n pot oblidar cap. El preu és que un tipus que
JSON no sap escriure hi entra igual de sol, i cau al final del camí endavant, amb
tota la feina feta i res desat.
"""
import json

import pandas as pd
import pytest

from epf import runner


def test_the_record_of_every_family_is_writable():
    forecasts = {"x": pd.DataFrame(0.0, index=pd.DatetimeIndex(["2024-01-01"]), columns=["h0"])}
    families = list(runner.families())
    for family in families:
        record = runner.described("abc1234", "test", family, families, True, forecasts, 1.0)
        json.dumps(record)


def test_the_record_says_which_families_the_invocation_chose():
    # `described` té un diccionari propi de tot el que s'escombra de `parameters.py`;
    # quan es deia igual que l'argument, el registre desava els noms dels paràmetres
    # en lloc de les famílies i seguia essent JSON vàlid.
    forecasts = {"x": pd.DataFrame(0.0, index=pd.DatetimeIndex(["2024-01-01"]), columns=["h0"])}
    record = runner.described("abc1234", "test", "naive", ["lear", "naive"], False,
                              forecasts, 1.0)
    assert record["chosen"] == ["lear", "naive"]
    assert record["family"] == "naive"


def test_a_value_json_cannot_write_is_refused_and_not_silently_dropped():
    with pytest.raises(TypeError):
        json.dumps(runner._plain({"bad": object()}))
