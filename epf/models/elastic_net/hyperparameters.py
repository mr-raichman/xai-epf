"""Els números de la xarxa elàstica.

La penalització i la barreja entre les dues les tria una cerca, i `fit_rows` no en té
cap valor per defecte. Encara no s'han cercat, i els d'aquí no ho són: la barreja és la
millor d'Uniejewski et al. (2016) i la penalització és inventada.
"""

NAME = "elastic_net"

# Sostre d'iteracions del descens de coordenades: el `MaxIter` per defecte de `lasso.m`,
# l'eina amb què Uniejewski et al. (2016) ajusten les seves xarxes.
MAX_ITER = 100000

# Sense cercar.
MIXING = 0.75
PENALTY = 0.001
