"""Els números de la xarxa elàstica."""

# Nom del model i de la seva família.
NAME = "elastic_net"

# Sostre d'iteracions del descens de coordenades: el `MaxIter` per defecte de `lasso.m`,
# l'eina amb què Uniejewski et al. (2016) ajusten les seves xarxes.
MAX_ITER = 100000

# El punt mentre no s'ha cercat: la barreja, la millor d'Uniejewski et al. (2016), i el
# càstig, inventat.
MIXING = 0.75
PENALTY = 0.001
