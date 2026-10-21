"""Els números de les xarxes."""

# Mida del batch, sostre d'èpoques i paciència de l'early stopping: els del codi de
# Marcjasz et al. (2023).
BATCH = 32
EPOCHS = 1500
PATIENCE = 50

# Els números que cap cerca no toca, els mateixos per a les tres xarxes.
FIXED = {"batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE}
