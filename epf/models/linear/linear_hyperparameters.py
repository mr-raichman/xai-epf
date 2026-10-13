"""Els números de la xarxa elàstica.

La penalització i la barreja entre les dues les tria una cerca, `epf search --family
elastic_net`, que les desa al disc, a `paths.SEARCHES`, i `searched()` les llegeix;
`fit_rows` no en té cap valor per defecte. Mentre la cerca no s'ha fet, la part B i el
camí endavant en prenen aquests, que no ho són: la barreja és la millor d'Uniejewski et
al. (2016) i la penalització és inventada.
"""
import json

from epf import paths
from epf.parameters import FOUR_YEARS, INPUTS

NAME = "elastic_net"

# La finestra de les xarxes neuronals, al camí endavant.
WINDOW = FOUR_YEARS

# Sostre d'iteracions del descens de coordenades, el de LEAR, on a 2500 els ajusts
# quedaven a mitges sense queixar-se. Mesurat el 2026-10-11 sobre les files
# d'entrenament de la llavor 0 de la part B: amb penalitzacions de 1e-4 a 0,1 i barreges
# de 0,05 a 0,95, les 24 hores convergeixen. Sobre la graella de la cerca, mesurat el
# 2026-10-13 a la finestra de la part A, convergeixen tots els punts menys el més baix de
# la barreja 0,75, 8,2e-5, on 2 de les 24 hores arriben al sostre.
MAX_ITER = 10000

# Sense cercar.
MIXING = 0.75
PENALTY = 0.001


def path(stage):
    """On viuen el càstig i la barreja que ha triat la cerca sobre un conjunt
    d'informació."""
    return paths.SEARCHES / stage / f"{NAME}.json"


def searched():
    """El càstig i la barreja de la xarxa, i d'on surten.

    Els de la cerca sobre el conjunt que es camina, si n'hi ha, amb la seva procedència
    sota `search`; si no, `PENALTY` i `MIXING`, marcats. El conjunt que es camina és un de
    sol, i el desempaquetat peta el dia que n'hi hagi dos.
    """
    (stage,) = INPUTS
    found = path(stage)
    if not found.exists():
        return {"penalty": PENALTY, "mixing": MIXING, "unsearched": True}
    record = json.loads(found.read_text())
    return record["hyperparameters"] | {
        "search": {name: value for name, value in record.items()
                   if name != "hyperparameters"}}


def save(stage, record):
    """Desa el que ha trobat la cerca sobre un conjunt, substituint el que hi hagi."""
    found = path(stage)
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text(json.dumps(record, indent=2) + "\n")
