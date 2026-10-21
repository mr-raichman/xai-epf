"""Els números de les xarxes."""

from epf.parameters import THREADS

# Mida del batch, sostre d'èpoques i paciència de l'early stopping: els del codi de
# Marcjasz et al. (2023).
BATCH = 32
EPOCHS = 1500
PATIENCE = 50

# Els mínims de l'escala i del pes de la cua de les distribucions, i el factor del
# softplus que s'hi suma, els del codi de l'article: una escala de zero faria infinita
# la log-versemblança.
SCALE_FLOOR = 1e-3
STRETCH = 3
TAIL_FLOOR = 1

# Tots els números fixos que llegeixen els trials i els ajustos, uns per a les tres
# xarxes, de manera que un canvi dels mínims de les distribucions refusa també les
# cerques de la de punt, que no en té.
FIXED = {
    "batch": BATCH,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "scale_floor": SCALE_FLOOR,
    "stretch": STRETCH,
    "tail_floor": TAIL_FLOOR,
    "threads": THREADS,
}
