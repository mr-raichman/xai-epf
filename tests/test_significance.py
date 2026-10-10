"""Les dues proves de significació multivariants, sobre casos amb resposta coneguda.

Una prova de significació és exactament el que pot donar una xifra plausible i
equivocada: un p-valor surt sempre, i si el signe o els instruments estan al revés
surt igual de bonic i diu el contrari. Els casos d'aquí són els que l'article i la
seva implementació deixen comprovar sense cap dada real: un pronòstic clarament
millor, un clarament pitjor, i dos idèntics.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.evaluation import diebold_mariano, giacomini_white

DAYS = 200


def frame_of(values):
    index = pd.DatetimeIndex([datetime.date(2024, 1, 1) + datetime.timedelta(days=day)
                              for day in range(DAYS)], name="day")
    return pd.DataFrame(values, index=index, columns=HOUR_COLUMNS)


def prices(seed=0):
    rng = np.random.default_rng(seed)
    return frame_of(rng.normal(60, 20, size=(DAYS, len(HOUR_COLUMNS))))


def off_by(actual, size, seed=1):
    """Un pronòstic que erra `size` de mitjana, amb soroll propi."""
    rng = np.random.default_rng(seed)
    return actual + rng.normal(0, size, size=actual.shape)


def test_a_clearly_better_first_forecast_gives_a_p_value_near_one():
    # El p-valor és d'una banda i porta el signe de la diferència de pèrdues: val 1
    # quan el primer pronòstic és el millor. Si el signe estigués al revés, aquest cas
    # i el següent es canviarien de lloc sense que res més es notés.
    actual = prices()
    assert giacomini_white(actual, off_by(actual, 1.0), off_by(actual, 20.0)) > 0.99


def test_a_clearly_worse_first_forecast_gives_a_p_value_near_zero():
    actual = prices()
    assert giacomini_white(actual, off_by(actual, 20.0), off_by(actual, 1.0)) < 0.01


def test_two_identical_forecasts_cannot_be_separated():
    # Diferència de pèrdues idènticament zero. Hi passa sense NaN i sense rebutjar
    # res, que és el que ha de fer: no hi ha cap diferència a trobar.
    actual = prices()
    same = off_by(actual, 5.0)
    value = giacomini_white(actual, same, same)
    assert not np.isnan(value)
    assert value == pytest.approx(1.0)


def test_two_forecasts_of_the_same_quality_are_not_declared_different():
    # Mateixa mida d'error i sorolls independents: cap de les dues bandes no ha de
    # rebutjar al 5%, que és el que separa una prova d'un comparador de mitjanes.
    #
    # Es demanen totes dues bandes a propòsit. Un sol p-valor no distingeix «no hi ha
    # diferència» de «la diferència va en l'altre sentit»: val 1 sempre que el primer
    # pronòstic sigui millor de mitjana, per poc que sigui. Mesurat sobre aquests dos:
    # 1.0 en un sentit i 0.838 en l'altre.
    actual = prices()
    one, other = off_by(actual, 5.0, seed=1), off_by(actual, 5.0, seed=2)
    assert min(giacomini_white(actual, one, other),
               giacomini_white(actual, other, one)) > 0.05


def test_a_fifth_of_a_mae_apart_is_separated_where_the_same_quality_is_not():
    # El que la prova afegeix a comparar dues mitjanes és dir quan una diferència és
    # prou gran per al nombre de dies que hi ha. Amb 200 dies, 3,98 contra 4,81
    # EUR/MWh de MAE es rebutja i 3,98 contra 4,01 no.
    actual = prices()
    better, worse = off_by(actual, 5.0, seed=1), off_by(actual, 6.0, seed=2)
    assert giacomini_white(actual, worse, better) < 0.01


def test_the_order_of_the_pair_matters_and_is_not_symmetric():
    # És d'una banda: canviar l'ordre no dóna el mateix p-valor, i una taula que ho
    # dibuixés com si fos simètric estaria mentint.
    actual = prices()
    better, worse = off_by(actual, 1.0), off_by(actual, 20.0)
    assert giacomini_white(actual, better, worse) != giacomini_white(actual, worse, better)


def test_too_few_shared_days_fails_loudly():
    actual = prices()
    one = off_by(actual, 1.0)
    with pytest.raises(ValueError):
        giacomini_white(actual.iloc[:2], one.iloc[:2], one.iloc[:2])


def test_diebold_mariano_puts_a_clearly_better_first_forecast_near_one():
    actual = prices()
    assert diebold_mariano(actual, off_by(actual, 1.0), off_by(actual, 20.0)) > 0.99


def test_diebold_mariano_puts_a_clearly_worse_first_forecast_near_zero():
    actual = prices()
    assert diebold_mariano(actual, off_by(actual, 20.0), off_by(actual, 1.0)) < 0.01


def test_the_two_tests_read_the_same_way_round():
    # És el que fa que es puguin posar a la mateixa taula. Si una de les dues tingués
    # el signe al revés, totes dues seguirien donant p-valors plausibles i la taula
    # diria el contrari en la meitat de les caselles.
    actual = prices()
    better, worse = off_by(actual, 1.0), off_by(actual, 20.0)
    assert diebold_mariano(actual, better, worse) > 0.99
    assert giacomini_white(actual, better, worse) > 0.99
    assert diebold_mariano(actual, worse, better) < 0.01
    assert giacomini_white(actual, worse, better) < 0.01


def test_diebold_mariano_does_not_separate_two_forecasts_of_the_same_quality():
    actual = prices()
    one, other = off_by(actual, 5.0, seed=1), off_by(actual, 5.0, seed=2)
    assert min(diebold_mariano(actual, one, other),
               diebold_mariano(actual, other, one)) > 0.05


def test_diebold_mariano_refuses_a_forecast_against_itself():
    # La diferència de pèrdues és zero tots els dies i la z seria zero partit per
    # zero. Es refusa en lloc de tornar un NaN que una taula dibuixaria com si fos un
    # p-valor; la de Giacomini-White sí que hi passa i torna 1.
    actual = prices()
    same = off_by(actual, 5.0)
    with pytest.raises(ValueError, match="diagonal"):
        diebold_mariano(actual, same, same)
    assert giacomini_white(actual, same, same) == pytest.approx(1.0)


def test_diebold_mariano_refuses_too_few_days_and_a_missing_value():
    actual = prices()
    one, other = off_by(actual, 1.0), off_by(actual, 2.0)
    with pytest.raises(ValueError):
        diebold_mariano(actual.iloc[:1], one.iloc[:1], other.iloc[:1])
    broken = one.copy()
    broken.iloc[5, 3] = np.nan
    with pytest.raises(ValueError, match="absents"):
        diebold_mariano(actual, broken, other)


def test_a_missing_value_is_refused_rather_than_skipped():
    actual = prices()
    one, other = off_by(actual, 1.0), off_by(actual, 2.0)
    one.iloc[5, 3] = np.nan
    with pytest.raises(ValueError, match="absents"):
        giacomini_white(actual, one, other)
