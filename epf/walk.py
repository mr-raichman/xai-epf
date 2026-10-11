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


def forward(predict, X, y, first, last, every=1, after=None, on_day=None):
    """Camina els dies de `first` a `last` i lliura cada un a mesura que surt.

    `predict` és una funció `(X, y, day)` que torna un diccionari de nom de pronòstic
    a les 24 hores del dia; cada família s'hi adapta i cap coneix les altres. Un sol
    ajust pot donar diversos pronòstics, que és el cas de les quatre finestres de LEAR i
    dels tres pronòstics ingenus: així el dia s'ajusta un sol cop i es camina un sol
    cop.
    `every` en pren un dia de cada N i no canvia res del que veu el model: cada dia
    mostrejat es prediu igual que en una passada sencera.
    `after` reprèn una execució: els dies fins a aquell, inclòs, se salten perquè ja
    són al disc. El mostreig es compta sempre des de `first`, perquè donar-li un
    `first` posterior desplaçaria la fase de `every` i els dies predits ja no serien
    els mateixos.
    `on_day` rep el dia i una taula d'una fila per pronòstic, i és qui l'escriu: aquí
    no es desa cap dia, de manera que el que queda al disc és exactament el que s'ha
    caminat i una execució es pot reprendre del seu propi directori.

    Torna els dies que ha caminat, que no són tots els del període quan reprèn. Els
    dies surten de l'índex sense repeticions: a la taula per hora cada dia hi és 24
    vegades i només s'ha de predir una.
    """
    period = (X.index >= pd.Timestamp(first)) & (X.index <= pd.Timestamp(last))
    days = X.index[period].unique()[::every]
    if days.empty:
        raise ValueError(f"cap dia de la matriu entre {first} i {last}")
    if after is not None:
        days = days[days > pd.Timestamp(after)]

    counted = collections.Counter()
    for position, day in enumerate(days):
        frames = {}
        for name, prices in predict(X, y, day).items():
            frame = pd.DataFrame([prices], index=days[position:position + 1],
                                 columns=HOUR_COLUMNS)
            refuse_missing(frame, f"les prediccions de {name} del {day.date()}")
            frames[name] = frame
            counted[name] += 1
        if on_day is not None:
            on_day(day, frames)

    short = sorted(name for name, count in counted.items() if count != len(days))
    if short:
        raise ValueError(f"el pronòstic {short[0]} només ha sortit "
                         f"{counted[short[0]]} de {len(days)} dies: un pronòstic hi és "
                         "tots els dies o cap")
    return days
