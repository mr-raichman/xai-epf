"""El camí endavant: predir cada dia del període amb el que se sabia el dia abans.

Cada dia es prediu amb un model reajustat només amb dies anteriors, que és com
s'hauria fet servir de debò. Res d'aquí tria cap paràmetre ni mesura cap error: triar
es fa sobre el període de validació i mesurar és de `evaluation.py`. El bucle no sap
quines famílies hi ha ni què és una finestra de calibratge: cada model lliga els seus
números abans d'arribar-hi.
"""
import collections

import pandas as pd

from epf.calendar import HOUR_COLUMNS
from epf.evaluation import refuse_missing


def forward(predict, X, y, first, last, every=1, on_day=None):
    """Prediccions dels dies de `first` a `last`, una taula per pronòstic.

    `predict` és una funció `(X, y, day)` que torna un diccionari de nom de pronòstic
    a les 24 hores del dia; cada família s'hi adapta i cap coneix les altres. Un sol
    ajust pot donar diversos pronòstics, que és el cas de les cinc finestres de LEAR i
    dels tres pronòstics ingenus: així el dia s'ajusta un sol cop i es camina un sol
    cop.
    `every` en pren un dia de cada N i no canvia res del que veu el model: cada dia
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
    rows = collections.defaultdict(list)
    for day in days:
        for name, prices in predict(X, y, day).items():
            rows[name].append(prices)
        if on_day is not None:
            on_day(day)
    predicted = {}
    for name, values in rows.items():
        if len(values) != len(days):
            raise ValueError(f"el pronòstic {name} només ha sortit {len(values)} de "
                             f"{len(days)} dies: un pronòstic hi és tots els dies o cap")
        frame = pd.DataFrame(values, index=days, columns=HOUR_COLUMNS)
        refuse_missing(frame, f"les prediccions de {name}")
        predicted[name] = frame
    return predicted
