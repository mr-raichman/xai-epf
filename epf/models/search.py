"""El que comparteixen les cerques de la part A: les files d'una prova i com es parteixen.

Una prova és un ajust, com al codi de Lago et al. (2021): amb els quatre anys anteriors
al primer dia de prova, aturat i puntuat amb la seva quarta part final en ordre de
calendari, partida per dies sencers a la taula per hora. La partició de la quarta part
és també la que fa servir la xarxa de l'article quan recalibra cada dia, aleshores
barrejant setmanes senceres. Cada família té la seva cerca a la seva carpeta, i
`epf search --family` tria quina.
"""
import numpy as np
import optuna
import pandas as pd

from epf.matrix import calibration
from epf.parameters import DAYS_PER_WEEK, FOUR_YEARS, TEST_PERIODS

# El primer dia de prova. Les cerques s'ajusten amb els quatre anys d'abans i no en
# toquen cap.
FIRST_TEST_DAY = pd.Timestamp(min(first for first, _ in TEST_PERIODS.values()))

# La part dels dies que s'aparta per aturar l'entrenament i puntuar la prova: el
# `percentage_val` del codi de Lago et al. (2021).
VALIDATION_SHARE = 0.25


def window(X, y):
    """Les files de totes les proves: les dels quatre anys anteriors al primer dia de
    prova, una per dia a la matriu diària i 24 a la taula per hora."""
    return calibration(X, y, FIRST_TEST_DAY, FOUR_YEARS)


def split(count, generator=None):
    """Els dies d'entrenament i els de validació d'una finestra de `count` dies, com els
    parteix el codi de Lago et al. (2021).

    La validació és la `VALIDATION_SHARE` final dels dies. Amb `generator`, abans s'hi
    barregen setmanes senceres, de set dies a comptar del primer, perquè dos dies veïns
    no quedin un a cada banda: és el que fa el codi en recalibrar, i la xarxa s'entrena
    també amb els dies més recents. Sense, l'ordre és el del calendari, que és el de les
    cerques. El tall és a un nombre de dies i no de setmanes, de manera que una setmana
    pot quedar partida entre les dues bandes, com al codi.
    """
    order = np.arange(count)
    if generator is not None:
        starts = generator.permutation(order[::DAYS_PER_WEEK])
        order = np.concatenate([order[start:start + DAYS_PER_WEEK] for start in starts])
    validation = int(VALIDATION_SHARE * count)
    return order[:count - validation], order[count - validation:]


def parts(X):
    """Les files d'entrenament i les de validació d'una prova sobre la taula per hora, com
    a màscares, partides per dies sencers en ordre de calendari: les 24 hores d'un dia
    van totes a la mateixa banda."""
    days = X.index.unique()
    train, validate = split(len(days))
    return X.index.isin(days[train]), X.index.isin(days[validate])


def completed(searching):
    """Les proves acabades d'un estudi."""
    return [one for one in searching.trials
            if one.state == optuna.trial.TrialState.COMPLETE]


def through_grid(searching, objective, points, trials=None):
    """Els punts d'una graella de `points` que l'estudi encara no ha provat, o menys si
    `trials` ho diu.

    Una graella ja sencera no en fa cap: el mostrejador d'Optuna en tornaria a provar un
    a l'atzar.
    """
    remaining = points - len(completed(searching))
    searching.optimize(objective, n_trials=remaining if trials is None
                       else min(trials, remaining))
