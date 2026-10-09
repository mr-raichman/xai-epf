"""La xarxa de l'article sobre la matriu diària: una fila per dia i 24 sortides.

Dues capes amagades, Adam, aturada anticipada i **pèrdua absoluta**, que és el que
l'article minimitza i el que fa que l'error que es mesura i el que s'optimitza siguin
el mateix. Marc multivariant: un sol model dóna les 24 hores alhora.

Aquesta família no s'atribueix. Hi és per dir com de lluny són LEAR i el gradient
boosting, que són els que s'interpreten, del que es pot arribar a fer amb aquestes
dades: és la mesura de com de creïbles són les heurístiques que se'ls llegeixen, i
no un model del qual s'esperi cap explicació.

**Els seus números encara no s'han triat** i són a `dnn_hyperparameters.py` amb el
que això costa escrit al costat.
"""
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler

from epf.calendar import HOUR_COLUMNS
from epf.matrix import calibration
from epf.models.dnn import network
from epf.models.dnn.dnn_hyperparameters import (ACTIVATION, BATCH,
                                                BATCH_NORMALISATION, DROPOUT,
                                                EPOCHS, HIDDEN, INITIALISER, L1,
                                                LEARNING_RATE, PATIENCE,
                                                VALIDATION_WEEKS, WINDOW)
from epf.parameters import SEED

NAME = "dnn"


def fit(X, y, day, window=WINDOW, hidden=HIDDEN, activation=ACTIVATION,
        dropout=DROPOUT, batch_normalisation=BATCH_NORMALISATION, l1=L1,
        initialiser=INITIALISER, learning_rate=LEARNING_RATE, batch=BATCH,
        epochs=EPOCHS, patience=PATIENCE, validation_weeks=VALIDATION_WEEKS,
        seed=SEED):
    """La xarxa d'un dia i els escaladors que l'alimenten, ajustats amb dies anteriors.

    Els escaladors surten de la finestra de calibratge i mai del dia que es prediu.
    L'article tria entre sis preprocessats per a l'entrada i sis més per a la sortida;
    aquí n'hi ha un de fix, el de mitjana i desviació, i triar-lo és de la cerca.

    L'aturada anticipada és la de Keras i no el bucle a mà de la seva implementació,
    amb la mateixa paciència que ells fan servir en recalibrar i tornant els millors
    pesos, que és el que el bucle feia a mà.
    """
    features, target = calibration(X, y, day, window)
    inputs, outputs = StandardScaler().fit(features), StandardScaler().fit(target)
    scaled, scaled_target = inputs.transform(features), outputs.transform(target)

    keras.utils.set_random_seed(seed)
    train, validate = network.split(len(scaled), validation_weeks,
                                    np.random.default_rng(seed))
    model = network.build(scaled.shape[1], hidden, activation, dropout,
                          batch_normalisation, l1, initialiser,
                          head=keras.layers.Dense(len(HOUR_COLUMNS)),
                          loss="mae", learning_rate=learning_rate)
    model.fit(scaled[train], scaled_target[train], batch_size=batch, epochs=epochs,
              validation_data=(scaled[validate], scaled_target[validate]),
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=patience, restore_best_weights=True)],
              verbose=0)
    return model, inputs, outputs


def predict(model, X, day):
    """Els 24 preus de `day`, per nom de pronòstic."""
    model, inputs, outputs = model
    scaled = model.predict(inputs.transform(X.loc[[day]]), verbose=0)
    return {NAME: outputs.inverse_transform(scaled)[0]}
