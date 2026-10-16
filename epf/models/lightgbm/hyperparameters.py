"""Els números del gradient boosting.

Els que ningú no cerca són els del codi del grup de Trebbien et al. (2023) d'on surt la
graella: el sostre d'arbres i la paciència dels ajustos que s'aturen, i les dues coses
que hi fixa a cada punt. El punt de la graella encara no s'ha cercat, i el que hi ha és
`UNSEARCHED`, marcat com a tal al registre de cada passada.
"""

# Nom del model i de la seva família, el de la implementació, que és el que es compara.
NAME = "lightgbm"

# Arbres com a molt d'un ajust que s'atura, i rondes sense millorar abans d'aturar-lo:
# els del codi del grup de Trebbien et al. (2023), que la graella de la cerca acompanya.
MOST_TREES = 1000
PATIENCE = 20

# El que aquell codi fixa a cada punt de la graella: la mostra de files es pren a cada
# arbre, i cap fulla no demana un pes mínim, de manera que la mida de les fulles la
# decideix només `min_child_samples`.
SUBSAMPLE_FREQ = 1
MIN_CHILD_WEIGHT = 0

# El punt mentre no s'ha cercat, inventat: els arbres, la taxa i les fulles d'abans de
# la graella, i els valors per defecte de LightGBM per a la resta, cap límit de
# profunditat, totes les files i 20 per fulla.
UNSEARCHED = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 1.0,
    "min_child_samples": 20,
    "trees": 500,
}
