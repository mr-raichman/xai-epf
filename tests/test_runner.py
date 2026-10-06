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
    for family in runner.families():
        record = runner.described("abc1234", "test", family, True, forecasts, 1.0)
        json.dumps(record)


def test_a_value_json_cannot_write_is_refused_and_not_silently_dropped():
    with pytest.raises(TypeError):
        json.dumps(runner._plain({"bad": object()}))
