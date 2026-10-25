"""Els números de la xarxa neuronal."""

from epf.parameters import THREADS

# Mida del batch, sostre d'èpoques i paciència de l'early stopping: els del codi de
# Marcjasz et al. (2023).
BATCH = 32
EPOCHS = 1500
PATIENCE = 50

# Tots els números fixos que llegeixen els trials i els ajustos.
FIXED = {
    "batch": BATCH,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "threads": THREADS,
}
