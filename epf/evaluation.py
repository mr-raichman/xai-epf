"""Camí endavant sobre un període, i l'error de les prediccions que en surten.

Cada dia es prediu amb un model reajustat només amb dies anteriors, que és com
s'hauria fet servir de debò. Res d'aquí tria cap paràmetre: escollir es fa sobre
el període de validació, no sobre el de prova.
"""
import pandas as pd

from epf.data.calendar import HOUR_COLUMNS


def walk_forward(predict, X, y, first, last, every=1, on_day=None):
    """Prediccions dels dies de `first` a `last`, una fila per dia.

    `predict` és una funció `(X, y, day)` que torna les 24 hores del dia; cada
    família s'hi adapta i cap coneix les altres. `every` en pren un dia de cada N,
    que serveix per a una passada de fum i no canvia res del que veu el model.
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
    return pd.DataFrame(rows, index=days, columns=HOUR_COLUMNS)


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


def mae_per_hour_over_price(predicted, actual):
    """MAE de cada hora dividit pel preu mitjà d'aquella hora, als dies compartits.

    El nivell del preu canvia molt al llarg del dia i l'error absolut el segueix, de
    manera que una hora cara sembla pitjor encara que no ho sigui. Divideix també la
    part de la variabilitat que sí que ha produït l'error, i per això és una lectura
    més i no la mètrica.
    """
    level = actual.loc[_days(predicted, actual)].mean()
    if (level <= 0).any():
        flat = ", ".join(level[level <= 0].index)
        raise ValueError(f"preu mitjà nul o negatiu a les hores {flat}: "
                         "dividir-hi l'error no vol dir res")
    return mae_per_hour(predicted, actual) / level


def metrics(forecasts, actual):
    """Taula llarga de totes les mètriques: pronòstic, mètrica, hora i valor.

    Llarga i no ampla perquè afegir-ne una no canviï la forma del fitxer: la ronda de
    mètriques encara ha de dir quines es reporten.
    """
    rows = []
    for name, frame in sorted(forecasts.items()):
        rows.append({"forecast": name, "metric": "mae", "hour": "", "value": mae(frame, actual)})
        for metric, series in (("mae_by_hour", mae_per_hour(frame, actual)),
                               ("mae_over_price", mae_per_hour_over_price(frame, actual))):
            rows += [{"forecast": name, "metric": metric, "hour": hour, "value": value}
                     for hour, value in series.items()]
    return pd.DataFrame(rows)
