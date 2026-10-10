"""Les xarxes distribucionals de Marcjasz et al. (2023), sobre la matriu diària.

La mateixa xarxa de Lago et al. (2021) amb la sortida redefinida: on aquella dóna 24
preus, aquestes donen els paràmetres d'una distribució per hora i minimitzen una
log-versemblança en lloc de l'error absolut. L'article ho diu així: les dues "differ
only in the output construction in the network and hyperparameters corresponding to the
missing distribution layer", de manera que aquí no es torna a escriure cap capa
amagada, només el cap i la pèrdua, sobre `models/dnn/network.py`.

Una família per distribució, que és el que fa l'article, i la diferència entre les dues
és la seva pròpia evidència que els moments alts compten.

**El pronòstic que surt al CSV és la mediana de la distribució**, que és el que
minimitza l'error absolut i el que l'article fa servir per al MAE. Els paràmetres van a
part, un fitxer per família, perquè una distribució no són 24 preus i tot el que
llegeix una execució espera 24 preus.
"""
import functools

import keras
import numpy as np
from sklearn.preprocessing import StandardScaler

from epf.matrix import calibration
from epf.models.ddnn.ddnn_hyperparameters import (BATCH, EPOCHS, MEMBERS, PATIENCE,
                                                  VALIDATION_WEEKS, WINDOW, searched)
from epf.models.dnn import network
from epf.models.dnn.dnn import columns_of
from epf.parameters import SEED


def fit(distribution, X, y, day, tuned=None, window=WINDOW, members=MEMBERS,
        batch=BATCH, epochs=EPOCHS, patience=PATIENCE,
        validation_weeks=VALIDATION_WEEKS, seed=SEED):
    """Les xarxes d'un dia, ajustades només amb dies anteriors.

    L'objectiu s'escala com a la xarxa de punt, i els paràmetres que en surten tornen
    a les unitats del preu desfent aquell escalat: la localització és un preu i
    l'escala una distància entre preus, de manera que l'una es desfà amb el centre i
    l'altra només amb la dispersió.
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
            head=lambda tensor: keras.layers.Dense(distribution.units)(tensor),
            loss=distribution.loss, learning_rate=tuned["learning_rate"])
        model.fit(scaled[train], scaled_target[train], batch_size=batch,
                  epochs=epochs,
                  validation_data=(scaled[validate], scaled_target[validate]),
                  callbacks=[keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=patience,
                      restore_best_weights=True)],
                  verbose=0)
        networks.append(model)
    return networks, inputs, outputs, columns


def parameters(distribution, model, X, day):
    """Els paràmetres de la distribució d'un dia, `(24, paràmetres)`, en EUR/MWh.

    La mitjana dels membres es fa sobre els paràmetres i no sobre les distribucions:
    amb un sol membre no canvia res, i amb més d'un és la mitjana que l'article anomena
    de barreja vertical sense la barreja, cosa que caldrà decidir si els membres pugen.
    """
    networks, inputs, outputs = model[0], model[1], model[2]
    row = inputs.transform(X.loc[[day], model[3]])
    found = np.mean([distribution.parameters(one.predict(row, verbose=0))
                     for one in networks], axis=0)[0]

    # Desfer l'escalat de l'objectiu: el centre només mou la localització i la
    # dispersió mou tota distància, de manera que l'escala i les localitzacions no es
    # desfan igual. L'asimetria i el pes de la cua no tenen unitats i es queden.
    centre, spread = outputs.mean_, np.sqrt(outputs.var_)
    undone = found.copy()
    undone[:, 0] = found[:, 0] * spread + centre
    undone[:, 1] = found[:, 1] * spread
    return undone


def predict(distribution, model, X, day, name):
    """El preu puntual del dia: la mediana de la distribució, per nom de pronòstic."""
    return {name: distribution.point(parameters(distribution, model, X, day))}


def bound(name, distribution):
    """Les tres funcions d'una família i la seva distribució, amb aquesta ja lligada.

    `families.py` les crida sense saber de quina distribució són, de manera que es
    lliga aquí. És el primer argument de totes tres a propòsit, que és el que deixa
    fer-ho amb `functools.partial` i no amb res més enginyós.
    """
    return (functools.partial(fit, distribution),
            functools.partial(predict, distribution, name=name),
            distribution._replace(
                of_day=functools.partial(parameters, distribution)))
