"""La xarxa de l'article sobre la matriu diària: una fila per dia i 24 sortides.

Dues capes amagades, Adam, aturada anticipada i **pèrdua absoluta**, que és el que
l'article minimitza i el que fa que l'error que es mesura i el que s'optimitza siguin
el mateix. Marc multivariant: un sol model dóna les 24 hores alhora.

Aquesta família no s'atribueix. Hi és per dir com de lluny són LEAR i el gradient
boosting, que són els que s'interpreten, del que es pot arribar a fer amb aquestes
dades: és la mesura de com de creïbles són les heurístiques que se'ls llegeixen, i no
un model del qual s'esperi cap explicació.

**Els números que la defineixen els tria la cerca i no una persona**, i `fit` els pren
en un sol objecte perquè és així com es trien, es desen i se substitueixen: tots junts.
Els que la cerca no toca, la finestra, el pressupost i les setmanes de validació, són
arguments com a qualsevol altra família.
"""
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler

from epf.calendar import HOUR_COLUMNS
from epf.matrix import DAY_OF_WEEK_COLUMNS, calibration
from epf.models.dnn import network
from epf.models.dnn.dnn_hyperparameters import (BATCH, EPOCHS, MEMBERS, NAME,
                                                PATIENCE, VALIDATION_WEEKS, WINDOW,
                                                searched)
from epf.parameters import LAGS, SEED


def blocks(lags=LAGS):
    """Els blocs d'entrada que té l'etapa, que és el que la cerca engega i apaga.

    Viuen aquí i no a la cerca perquè són del model: el conjunt que la cerca desa diu
    quins blocs ha deixat engegats i és `fit` qui n'ha de fer columnes. Lago et al.
    (2021) i Marcjasz et al. (2023) en tenen onze i catorze perquè els seus conjunts
    porten previsions, emissions i combustibles; l'etapa del preu té els quatre
    retards i el dia de la setmana.
    """
    found = {f"lag{lag}": [f"lag{lag}_{hour}" for hour in HOUR_COLUMNS]
             for lag in lags}
    found["day_of_week"] = DAY_OF_WEEK_COLUMNS
    return found


def columns_of(names):
    """Les columnes de la matriu dels blocs anomenats, en l'ordre de la matriu."""
    found = blocks()
    return [column for name in found if name in names for column in found[name]]


def fit(X, y, day, tuned=None, window=WINDOW, members=MEMBERS, batch=BATCH,
        epochs=EPOCHS, patience=PATIENCE, validation_weeks=VALIDATION_WEEKS,
        seed=SEED):
    """Les xarxes d'un dia i els escaladors que les alimenten, de dies anteriors.

    Els escaladors surten de la finestra de calibratge i mai del dia que es prediu.
    L'article tria entre sis preprocessats per a l'entrada i sis més per a la sortida;
    aquí n'hi ha un de fix, el de mitjana i desviació, perquè què fa cada un dels sis
    no està escrit enlloc que s'hagi llegit.

    `members` xarxes, cada una amb la seva llavor: a l'article cada membre surt d'una
    cerca pròpia, i aquí la cerca encara no s'ha repetit, de manera que el que varia
    entre membres és només la inicialització. Amb un membre no es mitjana res.

    L'aturada anticipada és la de Keras i no el bucle a mà de la implementació de
    l'article, amb la mateixa paciència que ells fan servir en recalibrar i tornant
    els millors pesos, que és el que el bucle feia a mà.
    """
    tuned = searched() if tuned is None else tuned
    columns = columns_of(tuned["blocks"])
    features, target = calibration(X[columns], y, day, window)
    inputs, outputs = StandardScaler().fit(features), StandardScaler().fit(target)
    scaled, scaled_target = inputs.transform(features), outputs.transform(target)

    networks = []
    for member in range(members):
        keras.utils.set_random_seed(seed + member)
        train, validate = network.split(len(scaled), validation_weeks,
                                        np.random.default_rng(seed + member))
        model = network.build(
            scaled.shape[1], tuned["hidden"], tuned["activation"], tuned["dropout"],
            tuned["batch_normalisation"], tuned["l1"], tuned["initialiser"],
            # Una capa nova per membre i no la mateixa: a Keras una capa reutilitzada
            # comparteix pesos, de manera que passar-ne la instància faria que la
            # segona xarxa heretés la sortida de la primera i la mitjana de quatre
            # membres no fos de quatre xarxes independents.
            head=lambda tensor: keras.layers.Dense(len(HOUR_COLUMNS))(tensor),
            loss="mae", learning_rate=tuned["learning_rate"])
        model.fit(scaled[train], scaled_target[train], batch_size=batch,
                  epochs=epochs,
                  validation_data=(scaled[validate], scaled_target[validate]),
                  callbacks=[keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=patience,
                      restore_best_weights=True)],
                  verbose=0)
        networks.append(model)
    return networks, inputs, outputs, columns


def predict(model, X, day):
    """Els 24 preus de `day`, la mitjana de les xarxes del model, per nom."""
    networks, inputs, outputs, columns = model
    row = inputs.transform(X.loc[[day], columns])
    return {NAME: np.mean([outputs.inverse_transform(one.predict(row, verbose=0))[0]
                           for one in networks], axis=0)}
