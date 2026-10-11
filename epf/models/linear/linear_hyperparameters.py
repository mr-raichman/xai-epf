"""Els números de la xarxa elàstica que no surten de cap cerca.

La penalització i la barreja entre les dues la tria una cerca, i no hi tenen cap valor
per defecte: sense una cerca el model no s'ajusta.
"""
# Sostre d'iteracions del descens de coordenades, el de LEAR, on a 2500 els ajusts
# quedaven a mitges sense queixar-se. Mesurat el 2026-10-11 sobre les files
# d'entrenament de la llavor 0 de la part B: amb penalitzacions de 1e-4 a 0,1 i barreges
# de 0,05 a 0,95, les 24 hores convergeixen.
MAX_ITER = 10000
