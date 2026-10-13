"""Els números del gradient boosting, que arriben de dos llocs.

**Els que tria la cerca no són aquí**: `epf search --family lightgbm` els desa al disc, a
`paths.SEARCHES`, el punt de la graella que millor puntua i quants arbres hi va fer
l'ajust abans d'aturar-se, i `searched()` els llegeix. Mentre no s'ha cercat, `searched()`
torna `UNSEARCHED`, marcat.

Els que hi són són els que ningú no cerca: la finestra; i, del codi del grup de Trebbien
et al. (2023) d'on surt la graella, el sostre d'arbres i la paciència dels ajustos que
s'aturen, i les dues coses que hi fixa a cada punt.
"""
import json

from epf import paths
from epf.parameters import FOUR_YEARS, INPUTS

# Nom de la família i del seu únic pronòstic. El mòdul es diu `boosting` perquè és la
# tècnica; la família es diu per la implementació, que és el que es compara.
NAME = "lightgbm"

# La finestra de les xarxes neuronals, al camí endavant.
WINDOW = FOUR_YEARS

# Arbres com a molt d'un ajust que s'atura, i rondes sense millorar abans d'aturar-lo:
# els del codi del grup de Trebbien et al. (2023), que la graella de la cerca acompanya.
MOST_TREES = 1000
PATIENCE = 20

# El que aquell codi fixa a cada punt de la graella: la mostra de files es pren a cada
# arbre, i cap fulla no demana un pes mínim, de manera que la mida de les fulles la
# decideix només `min_child_samples`.
SUBSAMPLE_FREQ = 1
MIN_CHILD_WEIGHT = 0

# El punt mentre no s'ha cercat, inventat: els arbres, la taxa i les fulles d'abans de la
# graella, i els valors per defecte de LightGBM per a la resta, cap límit de profunditat,
# totes les files i 20 per fulla. El camí no canvia fins que la cerca el substitueix.
UNSEARCHED = {"learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "subsample": 1.0,
              "min_child_samples": 20, "trees": 500}


def path(stage):
    """On viu el punt que ha triat la cerca sobre un conjunt d'informació."""
    return paths.SEARCHES / stage / f"{NAME}.json"


def searched():
    """Els números de l'arbre i d'on surten.

    Els de la cerca sobre el conjunt que es camina, si n'hi ha, amb la seva procedència
    sota `search`; si no, `UNSEARCHED`, marcat. El conjunt que es camina és un de sol, i
    el desempaquetat peta el dia que n'hi hagi dos, que és quan aquesta funció haurà de
    saber quin.
    """
    (stage,) = INPUTS
    found = path(stage)
    if not found.exists():
        return UNSEARCHED | {"unsearched": True}
    record = json.loads(found.read_text())
    return record["hyperparameters"] | {
        "search": {name: value for name, value in record.items()
                   if name != "hyperparameters"}}


def save(stage, record):
    """Desa el que ha trobat la cerca sobre un conjunt, substituint el que hi hagi."""
    found = path(stage)
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text(json.dumps(record, indent=2) + "\n")
