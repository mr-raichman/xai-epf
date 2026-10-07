"""Alineació del camí endavant: quins dies es prediuen i amb quina etiqueta.

Res d'aquí mesura cap error. El model és un que encerta sempre, de manera que el que
falla quan una prova falla és l'alineació i no el model.
"""
import datetime

import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.walk import forward

# Nom del pronòstic d'aquestes proves. `forward` torna una taula per nom, perquè un
# sol ajust pot donar-ne diversos.
ONE = "one"


def frame_of(days, start=datetime.date(2024, 1, 1), offset=0.0):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[d * 100 + hour + offset for hour in range(24)] for d in range(days)]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def perfect(X, y, day):
    """Un model que encerta, per comprovar l'alineació i no el model."""
    return {ONE: y.loc[day].to_numpy()}


def walked(*arguments, **keywords):
    """La taula de l'únic pronòstic, per llegir les proves d'una sola sèrie."""
    return forward(*arguments, **keywords)[ONE]


def test_every_day_of_the_period_is_predicted_and_kept_in_order():
    X = y = frame_of(30)
    out = walked(perfect, X, y, "2024-01-10", "2024-01-20")
    assert list(out.index) == list(pd.date_range("2024-01-10", "2024-01-20", freq="D"))
    assert list(out.columns) == HOUR_COLUMNS


def test_each_row_is_the_day_it_is_labelled_with():
    X = y = frame_of(30)
    out = walked(perfect, X, y, "2024-01-10", "2024-01-20")
    assert (out == y.loc[out.index]).to_numpy().all()


def test_taking_one_day_in_every_n_keeps_the_first():
    X = y = frame_of(30)
    out = walked(perfect, X, y, "2024-01-10", "2024-01-20", every=5)
    assert list(out.index) == [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-15"),
                               pd.Timestamp("2024-01-20")]


def test_an_empty_period_fails_loudly():
    X = y = frame_of(30)
    with pytest.raises(ValueError):
        walked(perfect, X, y, "2025-01-01", "2025-02-01")


def test_a_day_repeated_in_the_index_is_predicted_once():
    # La taula per hora té cada dia 24 vegades i només s'ha de predir una.
    y = frame_of(30)
    X = y.loc[y.index.repeat(24)]
    out = walked(lambda features, target, day: {ONE: target.loc[day].to_numpy()},
                 X, y, "2024-01-10", "2024-01-12")
    assert list(out.index) == list(pd.date_range("2024-01-10", "2024-01-12", freq="D"))


def test_a_forecast_absent_on_some_day_fails_loudly():
    # Un pronòstic hi és tots els dies o cap: si un dia en faltés, la taula que en
    # sortiria tindria menys files que dies i s'alinearia amb els dies equivocats.
    y = frame_of(30)

    def sometimes(X, target, day):
        names = {ONE: target.loc[day].to_numpy()}
        if day != pd.Timestamp("2024-01-11"):
            names["now_and_then"] = target.loc[day].to_numpy()
        return names

    with pytest.raises(ValueError, match="now_and_then"):
        forward(sometimes, y, y, "2024-01-10", "2024-01-12")
