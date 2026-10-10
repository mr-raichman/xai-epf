"""L'error de les prediccions d'una execució, i la guarda que les ha de passar.

Totes les mesures comparen només els dies que les dues taules tenen. El camí endavant
que les produeix és a `walk.py`: aquí no hi ha cap bucle sobre el període, només el que
es diu d'un conjunt de prediccions ja fetes, vinguin de la memòria o del disc.

Res d'aquí tria cap paràmetre: escollir es fa sobre el període de validació, no sobre
el de prova.
"""
import numpy as np
import pandas as pd
from scipy import stats

from epf.market import HOURS_PER_DAY

# La prova de Giacomini-White de Lago et al. (2021) secció 5.5.2: la diferència de
# pèrdues es regressa sobre el conjunt d'informació del dia anterior, que és una
# constant i els seus propis retards. Un retard, que és el que fa servir la seva
# implementació, i per tant dos instruments i dos graus de llibertat.
GW_LAGS = 1


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



def giacomini_white(actual, first, second):
    """p-valor de la prova de Giacomini-White multivariant entre dos pronòstics.

    Afirmar que un model és millor que un altre demana una prova, i la simple de
    Diebold-Mariano no val per a la predicció diària, perquè les 24 hores es
    pronostiquen de cop des del mateix conjunt d'informació. Lago et al. (2021) secció
    5.5.2 prefereix la de Giacomini-White perquè és la generalització condicional:
    totes dues valen per a models imbricats, però només aquesta té en compte la
    incertesa d'haver estimat els paràmetres.

    La variant multivariant resumeix la comparació en un sol p-valor. Es fa la sèrie
    diària de diferència de pèrdues `d`, es regressa sobre una constant i el seu propi
    retard, i l'estadístic és `T` pel R² de regressar uns sobre els instruments
    multiplicats per `d`, contra una khi quadrat amb tants graus de llibertat com
    instruments.

    **És d'una banda, i això és tota la lectura.** L'estadístic porta el signe de la
    mitjana de `d`, de manera que el p-valor val 1 sempre que el primer pronòstic sigui
    millor de mitjana, per poc que sigui, i és petit quan el millor és el segon. Un sol
    valor no distingeix «no hi ha diferència» de «la diferència va a l'altre sentit»:
    qui vulgui saber si dos pronòstics es poden separar ha de demanar les dues bandes.
    L'article en diu dues proves d'una banda, i cada parella ordenada és una casella
    del tauler de la seva figura 5.

    La pèrdua és l'absoluta, `|e|`, que és la mateixa que mesura el MAE i la mateixa
    que la xarxa minimitza. L'article admet també la quadràtica i aquí no hi és.

    Hi ha una condició de la nota al peu 18 de l'article: la prova val mentre la
    finestra de calibratge no creixi amb la mostra, cosa que es compleix per a una
    finestra lliscant i no per a la de tot l'historial de LEAR. Les comparacions que
    impliquen aquella finestra queden fora de la condició, i qui les demana ho ha de
    saber: aquí no es refusa res, perquè la prova no sap de quina finestra ve cada
    pronòstic.
    """
    days = _days(first, actual).intersection(second.index)
    if len(days) < GW_LAGS + 2:
        raise ValueError(f"{len(days)} dies compartits no donen per a una prova amb "
                         f"{GW_LAGS} retard")
    for frame, what in ((first, "el primer pronòstic"), (second, "el segon"),
                        (actual, "el preu real")):
        refuse_missing(frame.loc[days], what)

    difference = (np.abs(actual.loc[days] - first.loc[days])
                  - np.abs(actual.loc[days] - second.loc[days])).to_numpy().mean(axis=1)
    instruments = np.vstack([np.ones(len(difference) - GW_LAGS),
                             difference[:-GW_LAGS]])
    regressors = instruments * difference[GW_LAGS:]

    count = regressors.shape[1]
    betas = np.linalg.lstsq(regressors.T, np.ones(count), rcond=None)[0]
    residuals = np.ones(count) - regressors.T @ betas
    statistic = count * (1 - np.mean(residuals ** 2)) * np.sign(difference.mean())
    return float(1 - stats.chi2.cdf(statistic, regressors.shape[0]))


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
