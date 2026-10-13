"""Que el registre d'una execució es pugui escriure.

`described` escombra tot el que `parameters.py` té en majúscules, de manera que un
paràmetre nou hi entra sol i no se'n pot oblidar cap. El preu és que un tipus que
JSON no sap escriure hi entra igual de sol, i cau al final del camí endavant, amb
tota la feina feta i res desat.
"""
import json

import pandas as pd
import pytest

from epf import paths, runner
from epf.models.ddnn import ddnn_hyperparameters
from epf.models.families import families, hyperparameters

@pytest.fixture(autouse=True)
def searched(tmp_path, monkeypatch):
    """Les cerques, al disc d'una prova i no al del projecte.

    `autouse` perquè el registre de qualsevol família passa per `hyperparameters`, i
    aquella llegeix la cerca de qui en tingui.
    """
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)


def test_the_record_of_every_family_is_writable():
    forecasts = {"x": pd.DataFrame(0.0, index=pd.DatetimeIndex(["2024-01-01"]), columns=["h0"])}
    table = families()
    for name, family in table.items():
        record = runner.described("abc1234", "test", "lago", name, list(table), family,
                                  forecasts, 1.0, 1)
        json.dumps(record)


def test_the_record_says_which_families_the_invocation_chose():
    # `described` té un diccionari propi de tot el que s'escombra de `parameters.py`;
    # quan es deia igual que l'argument, el registre desava els noms dels paràmetres
    # en lloc de les famílies i seguia essent JSON vàlid.
    forecasts = {"x": pd.DataFrame(0.0, index=pd.DatetimeIndex(["2024-01-01"]), columns=["h0"])}
    record = runner.described("abc1234", "test", "lago", "naive", ["lear", "naive"],
                              families()["naive"], forecasts, 1.0, 1)
    assert record["chosen"] == ["lear", "naive"]
    assert record["family"] == "naive"


def test_the_record_says_who_is_a_member_and_who_is_drawn():
    # La combinació i les figures surten del registre i no del registre de famílies,
    # de manera que una reconstrucció no ha de tornar a endevinar-ho. Amb LEAR no
    # coincideixen: els membres són les quatre finestres i la capçalera la seva mitjana.
    forecasts = {"x": pd.DataFrame(0.0, index=pd.DatetimeIndex(["2024-01-01"]), columns=["h0"])}
    record = runner.described("abc1234", "test", "lago", "lear", ["lear"], families()["lear"],
                              forecasts, 1.0, 1)
    assert record["headlines"] == ["lear"]
    assert len(record["members"]) == 4
    assert "lear" not in record["members"]


def test_every_family_with_numbers_records_some():
    # Mesurat el 2026-10-10: mentre el mòdul dels números es derivava del nom de la
    # família, les dues distribucionals en compartien carpeta, l'import fallava, i
    # `hyperparameters` tornava un diccionari buit. El `run.json` seguia essent vàlid i
    # no portava cap número dels que havien fet la predicció, que és justament el que
    # ha de portar quan el commit ja no els fixa.
    table = families()
    for name, family in table.items():
        found = hyperparameters(name)
        if family.hyperparameters is None:
            assert found == {}
        else:
            assert found, f"la família {name} no desa cap número"


def test_each_distributional_family_records_its_own_numbers():
    # Les dues comparteixen el mòdul de números i no els números. Si el registre els
    # demanés al mòdul, totes dues portarien els d'una, i el `run.json` seguiria essent
    # vàlid amb la distribució equivocada.
    for name in ("ddnn_normal", "ddnn_jsu"):
        found = hyperparameters(name)
        assert found["WINDOW"] == ddnn_hyperparameters.WINDOW
        assert found["blocks"] == ddnn_hyperparameters.COPIED[name]["blocks"]
        assert found["search"]["copied_from"] == ddnn_hyperparameters.COPIED_FROM[name]
        assert found["unsearched"] is True
        json.dumps(found)
