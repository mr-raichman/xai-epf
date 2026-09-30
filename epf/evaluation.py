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
    """
    period = (X.index >= pd.Timestamp(first)) & (X.index <= pd.Timestamp(last))
    days = X.index[period][::every]
    if days.empty:
        raise ValueError(f"cap dia de la matriu entre {first} i {last}")
    rows = []
    for day in days:
        rows.append(predict(X, y, day))
        if on_day is not None:
            on_day(day)
    return pd.DataFrame(rows, index=days, columns=HOUR_COLUMNS)


def _shared(predicted, actual):
    days = predicted.index.intersection(actual.index)
    if days.empty:
        raise ValueError("les prediccions i els preus reals no comparteixen cap dia")
    return (predicted.loc[days] - actual.loc[days]).abs()


def mae(predicted, actual):
    """Error absolut mitjà en EUR/MWh, sobre els dies i hores que les dues taules tenen."""
    return float(_shared(predicted, actual).to_numpy().mean())


def mae_per_hour(predicted, actual):
    """Error absolut mitjà de cada hora del dia, en EUR/MWh."""
    return _shared(predicted, actual).mean()
