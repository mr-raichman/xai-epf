"""Camí endavant sobre un període, i l'error de les prediccions que en surten.

Cada dia es prediu amb un model reajustat només amb dies anteriors, que és com
s'hauria fet servir de debò. Res d'aquí tria cap paràmetre: escollir es fa sobre
el període de validació, no sobre el de prova.
"""
import numpy as np
import pandas as pd

from epf.market import HOURS_PER_DAY
from epf.data.calendar import HOUR_COLUMNS


def refuse_missing(frame, what):
    """Falla si la taula porta cap valor absent, anomenant on és el primer.

    Un NaN es veu a les mètriques d'un sol valor, que tornen NaN, i no es veu a les
    d'hora per hora, que el salten i informen l'hora com si fos neta. Es comprova
    als dos llocs on pot arribar: quan el camí endavant torna el que ha predit, que
    agafa un model que ha començat a fallar, i quan les mètriques reben el que
    llegeixen, que agafa una execució que ja és al disc.
    """
    missing = frame.isna()
    if not missing.to_numpy().any():
        return
    flat = missing.stack()
    row, column = next(iter(flat[flat].index))
    raise ValueError(f"{int(missing.to_numpy().sum())} valors absents a {what}, "
                     f"el primer a {row} hora {column}")


def walk_forward(predict, X, y, first, last, every=1, on_day=None):
    """Prediccions dels dies de `first` a `last`, una fila per dia.

    `predict` és una funció `(X, y, day)` que torna les 24 hores del dia; cada
    família s'hi adapta i cap coneix les altres. `every` en pren un dia de cada N i no canvia res del que veu el model: cada dia
    mostrejat es prediu igual que en una passada sencera.
    `on_day` es crida amb cada dia ja predit, perquè una passada de vuitanta minuts
    sense cap senyal no es distingeix d'un penjat.

    Els dies surten de l'índex sense repeticions: a la taula per hora cada dia hi és
    24 vegades i només s'ha de predir una.
    """
    period = (X.index >= pd.Timestamp(first)) & (X.index <= pd.Timestamp(last))
    days = X.index[period].unique()[::every]
    if days.empty:
        raise ValueError(f"cap dia de la matriu entre {first} i {last}")
    rows = []
    for day in days:
        rows.append(predict(X, y, day))
        if on_day is not None:
            on_day(day)
    predicted = pd.DataFrame(rows, index=days, columns=HOUR_COLUMNS)
    refuse_missing(predicted, "les prediccions del camí endavant")
    return predicted


def _days(predicted, actual):
    days = predicted.index.intersection(actual.index)
    if days.empty:
        raise ValueError("les prediccions i els preus reals no comparteixen cap dia")
    return days


def _shared(predicted, actual):
    days = _days(predicted, actual)
    return (predicted.loc[days] - actual.loc[days]).abs()


def mae(predicted, actual):
    """Error absolut mitjà en EUR/MWh, sobre els dies i hores que les dues taules tenen."""
    return float(_shared(predicted, actual).to_numpy().mean())


def mae_per_hour(predicted, actual):
    """Error absolut mitjà de cada hora del dia, en EUR/MWh."""
    return _shared(predicted, actual).mean()


def rmae(predicted, actual, naive):
    """MAE del model dividit pel de la predicció ingènua setmanal, sobre els dies que
    tots tres comparteixen.

    El rMAE de Lago et al. (2021): per sota d'1 el model guanya la predicció ingènua i
    per sobre la perd. El denominador és un error absolut mitjà d'un període sencer, de
    manera que mai no és a prop de zero, que és el que enfonsa les mesures que divideixen
    error per error.
    """
    days = _days(predicted, actual).intersection(naive.index)
    if days.empty:
        raise ValueError("el model, el preu real i la predicció ingènua no comparteixen cap dia")
    return mae(predicted.loc[days], actual) / mae(naive.loc[days], actual)


def rmae_per_hour(predicted, actual, naive):
    """El mateix hora per hora, amb un denominador per hora.

    Diverge del rMAE de l'article, que pren un sol denominador de tot el període: amb un
    per hora, el que es compara és com de difícil és cada hora i no com de cara és.
    """
    days = _days(predicted, actual).intersection(naive.index)
    if days.empty:
        raise ValueError("el model, el preu real i la predicció ingènua no comparteixen cap dia")
    floor = mae_per_hour(naive.loc[days], actual)
    if (floor <= 0).any():
        raise ValueError("alguna hora té la predicció ingènua perfecta i no s'hi pot dividir")
    return mae_per_hour(predicted.loc[days], actual) / floor


def error_dispersion(predicted, actual):
    """Cov-e de Maciejowska, Lipiecki i Uniejewski (2026): el logaritme del determinant
    de la matriu de segons moments dels errors d'un dia.

    Diu com de repartits estan els errors d'un mateix dia entre les 24 hores. Baixa quan
    els errors minven i quan es correlacionen entre ells, de manera que premia una corba
    equivocada en bloc per damunt d'una equivocada hora per hora: la primera encara té la
    forma del dia. Com més baix, millor, i barreja dispersió amb precisió.

    La matriu té rang com a molt el nombre de dies, així que amb menys de 24 és singular i
    el determinant és zero; aleshores no es calcula en lloc de tornar menys infinit.
    """
    days = _days(predicted, actual)
    errors = (predicted.loc[days] - actual.loc[days]).to_numpy()
    if len(errors) < HOURS_PER_DAY:
        raise ValueError(f"{len(errors)} dies per a una matriu de {HOURS_PER_DAY} hores, "
                         "que surt singular")
    sign, value = np.linalg.slogdet(errors.T @ errors / len(errors))
    if sign <= 0:
        raise ValueError("la matriu de segons moments dels errors no és definida positiva")
    return float(value)


def curve_correlations(predicted, actual):
    """Correlació de Spearman de cada dia entre la corba predita i la real.

    Spearman és Pearson sobre els rangs. Un dia en què les 24 hores tenen el mateix preu,
    o la mateixa predicció, no té rangs que variïn i surt absent en lloc de zero.
    """
    days = _days(predicted, actual)
    ranked = [frame.loc[days].rank(axis="columns").to_numpy() for frame in (predicted, actual)]
    centred = [values - values.mean(axis=1, keepdims=True) for values in ranked]
    spread = np.sqrt((centred[0] ** 2).sum(axis=1) * (centred[1] ** 2).sum(axis=1))
    # La divisió es fa només on hi ha dispersió: `np.where` calcularia les dues branques
    # i dividiria per zero igualment.
    usable = spread > 0
    correlation = np.full(len(days), np.nan)
    correlation[usable] = (centred[0] * centred[1]).sum(axis=1)[usable] / spread[usable]
    return pd.Series(correlation, index=days)


def extrema_price_deviations(predicted, actual):
    """MPD de cada dia: el tros de forquilla diària que es perd triant malament les hores.

    De Maciejowska, Lipiecki i Uniejewski (2026). Es mira a quina hora el pronòstic diu
    que hi haurà el mínim i el màxim del dia, i es compara el preu real d'aquelles hores
    amb el mínim i el màxim reals. Equival a la forquilla real menys la que el pronòstic
    hauria capturat, de manera que va en EUR/MWh i zero és encertar les dues hores.

    Amb dues hores empatades es pren la primera, que és el que fan `argmin` i `argmax`.
    """
    days = _days(predicted, actual)
    real = actual.loc[days].to_numpy()
    guess = predicted.loc[days].to_numpy()
    rows = np.arange(len(days))
    lost = (np.abs(real.min(axis=1) - real[rows, guess.argmin(axis=1)])
            + np.abs(real.max(axis=1) - real[rows, guess.argmax(axis=1)]))
    return pd.Series(lost, index=days)


def extrema_price_deviation(predicted, actual):
    """MPD: la mitjana d'aquelles pèrdues. Com més baix, millor."""
    return float(extrema_price_deviations(predicted, actual).mean())


def metrics(forecasts, actual, naive):
    """Taula llarga de totes les mètriques: pronòstic, mètrica, hora i valor.

    Llarga i no ampla perquè afegir-ne una no canviï la forma del fitxer: la ronda de
    mètriques encara ha de dir quines es reporten.
    """
    refuse_missing(actual, "el preu real")
    for name, frame in sorted(forecasts.items()):
        refuse_missing(frame, f"el pronòstic {name}")
    rows = []
    for name, frame in sorted(forecasts.items()):
        daily = curve_correlations(frame, actual)
        singles = (("mae", mae(frame, actual)),
                   ("rmae", rmae(frame, actual, naive)),
                   ("error_dispersion", error_dispersion(frame, actual)),
                   ("curve_correlation", float(daily.mean())),
                   ("curve_correlation_days", float(daily.notna().sum())),
                   ("extrema_price_deviation", extrema_price_deviation(frame, actual)))
        rows += [{"forecast": name, "metric": metric, "hour": "", "value": value}
                 for metric, value in singles]
        for metric, series in (("mae_by_hour", mae_per_hour(frame, actual)),
                               ("rmae_by_hour", rmae_per_hour(frame, actual, naive))):
            rows += [{"forecast": name, "metric": metric, "hour": hour, "value": value}
                     for hour, value in series.items()]
    return pd.DataFrame(rows)
