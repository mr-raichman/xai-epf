"""Els números de la xarxa neuronal."""

from epf.parameters import THREADS

# Sostre d'èpoques, tantes que només la paciència atura un ajust: amb les 1500 del codi
# de Marcjasz et al. (2023), s'hi aturava un ajust a 1e-5 que encara millorava, i un a
# batch 256 en va fer 41604.
EPOCHS = 100000

# Paciència de l'early stopping, la d'aquell codi, comptada al seu batch: a qualsevol
# batch, la xarxa espera les mateixes actualitzacions dels pesos.
PATIENCE = 50
PATIENCE_BATCH = 32

# Tots els números fixos que llegeixen els trials i els ajustos.
FIXED = {
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "patience_batch": PATIENCE_BATCH,
    "threads": THREADS,
}
