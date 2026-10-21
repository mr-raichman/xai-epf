"""Els noms i els números de l'arbre."""

# Nom del model i de la seva família.
NAME = "lightgbm"

# La columna del comptador de la sonda de datació, els dies des del primer del període,
# i el seu grup a la importància per grups, que no és cap sèrie ni el calendari.
DAY_COUNTER = "day_counter"
DATE = "date"

# Arbres com a molt d'un ajust amb early stopping, i rondes sense millorar abans
# d'aturar-lo: els del codi de Kruse et al. (2022), que la graella de la cerca
# acompanya.
MOST_TREES = 1000
PATIENCE = 20

# El que aquell codi fixa a cada punt de la graella: la mostra de files es pren a cada
# arbre, i cap fulla no demana un pes mínim, de manera que la mida de les fulles la
# decideix només `min_child_samples`.
SUBSAMPLE_FREQ = 1
MIN_CHILD_WEIGHT = 0

# El punt mentre no s'ha cercat: els arbres i la taxa, inventats, i per a la resta els
# valors per defecte de LightGBM.
UNSEARCHED = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 1.0,
    "min_child_samples": 20,
    "trees": 500,
}
