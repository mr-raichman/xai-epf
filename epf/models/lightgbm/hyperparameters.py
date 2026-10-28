"""Els noms i els números de l'arbre."""

# Nom del model i de la seva família.
NAME = "lightgbm"

# La columna del comptador de la sonda de datació, els dies des del primer del període.
DAY_COUNTER = "day_counter"

# Arbres com a molt d'un ajust amb early stopping, tants que només la paciència l'atura:
# amb els 1000 del codi de Kruse et al. (2022), s'hi aturaven ajustos que encara
# milloraven.
MOST_TREES = 50000

# Rondes sense millorar abans d'aturar un ajust: les d'aquell codi, que la graella de la
# cerca acompanya.
PATIENCE = 20

# El que aquell codi fixa a cada punt de la graella: la mostra de files es pren a cada
# arbre, i cap fulla no demana un pes mínim, de manera que la mida de les fulles la
# decideix només `min_child_samples`.
SUBSAMPLE_FREQ = 1
MIN_CHILD_WEIGHT = 0
