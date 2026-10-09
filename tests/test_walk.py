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
    """La taula de l'únic pronòstic, muntada del que `forward` lliura dia a dia.

    `forward` no desa cap dia: els escriu qui li passa `on_day`, que a l'execució és
    el corredor i aquí és aquesta llista.
    """
    rows = []
    keywords["on_day"] = lambda day, frames: rows.append(frames[ONE])
    forward(*arguments, **keywords)
    return pd.concat(rows)


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


def test_resuming_keeps_the_sampled_days_exactly_where_they_were():
    # Reprendre no pot desplaçar la fase de `every`. Si el dia on es reprèn passés
    # com a `first`, un dia de cada cinc des del 16 donaria el 16, el 21 i el 26, i
    # cap d'ells és un dia que la passada sencera hagués predit.
    X = y = frame_of(30)
    whole = walked(perfect, X, y, "2024-01-10", "2024-01-26", every=5)
    rest = walked(perfect, X, y, "2024-01-10", "2024-01-26", every=5,
                  after=pd.Timestamp("2024-01-15"))
    assert list(whole.index) == [pd.Timestamp(day) for day in
                                 ("2024-01-10", "2024-01-15", "2024-01-20", "2024-01-25")]
    assert list(rest.index) == list(whole.index[2:])


def test_resuming_past_the_last_day_walks_nothing():
    # Una execució que va caure entre l'últim dia i les mètriques: els dies ja hi són
    # tots i el que queda per fer és tancar el directori.
    X = y = frame_of(30)
    assert forward(perfect, X, y, "2024-01-10", "2024-01-12",
                   after=pd.Timestamp("2024-01-12")).empty


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
