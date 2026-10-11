"""Els números de la xarxa elàstica.

La penalització i la barreja entre les dues les ha de triar una cerca, i `fit_rows` no
en té cap valor per defecte. Mentre la cerca no s'ha fet, la part B i el camí endavant
en prenen aquests, que no ho són: la barreja és la millor d'Uniejewski et al. (2016) i
la penalització és inventada.
"""
from epf.parameters import FOUR_YEARS

# La finestra de les xarxes neuronals, al camí endavant.
WINDOW = FOUR_YEARS

# Sostre d'iteracions del descens de coordenades, el de LEAR, on a 2500 els ajusts
# quedaven a mitges sense queixar-se. Mesurat el 2026-10-11 sobre les files
# d'entrenament de la llavor 0 de la part B: amb penalitzacions de 1e-4 a 0,1 i barreges
# de 0,05 a 0,95, les 24 hores convergeixen.
MAX_ITER = 10000

# Sense cercar.
MIXING = 0.75
PENALTY = 0.001


def searched():
    """El que ha triat la cerca: res encara, i el `run.json` ho diu."""
    return {"unsearched": True}
