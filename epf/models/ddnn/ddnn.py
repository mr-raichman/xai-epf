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
import collections
import functools

import keras
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

from epf.matrix import calibration
from epf.models.ddnn.ddnn_hyperparameters import (BATCH, EPOCHS, MEMBERS, PATIENCE,
                                                  VALIDATION_WEEKS, WINDOW, searched)
from epf.models.ddnn.distributions import parameters as raw_parameters
from epf.models.dnn import network
from epf.models.dnn.dnn import columns_of
from epf.parameters import SEED

# Una xarxa ajustada i el que cal per llegir-la: els dos escaladors, el de l'objectiu
# el que torna els paràmetres a EUR/MWh, i les columnes que llegeix, en el seu ordre.
Fitted = collections.namedtuple("Fitted", "network inputs outputs columns")

# L'atribució de cada paràmetre: `values` és `(files, columnes, 24, paràmetres)` i
# `base` `(24, paràmetres)`, en les unitats en què es desa cada paràmetre, EUR/MWh la
# localització i l'escala i sense unitats l'asimetria i el pes de la cua.
Explanation = collections.namedtuple("Explanation", "names columns values base")


def fit_rows(distribution, train, validate, tuned, batch=BATCH, epochs=EPOCHS,
             patience=PATIENCE, seed=SEED):
    """Una xarxa ajustada amb les files de `train` i aturada quan deixa de millorar a
    les de `validate`, totes dues parelles `(X, y)`.

    Els escaladors s'ajusten amb les dues, que són totes les files que la xarxa mira i
    cap de les que es prediuen. L'objectiu s'escala com a la xarxa de punt, i els
    paràmetres en tornen a les unitats del preu desfent aquell escalat.
    """
    (X, y), (X_validate, y_validate) = train, validate
    inputs = StandardScaler().fit(pd.concat([X, X_validate]))
    outputs = StandardScaler().fit(pd.concat([y, y_validate]))
    keras.utils.set_random_seed(seed)
    model = network.build(
        X.shape[1], tuned["hidden"], tuned["activation"], tuned["dropout"],
        tuned["batch_normalisation"], tuned["l1"], tuned["initialiser"],
        head=lambda tensor: keras.layers.Dense(distribution.units)(tensor),
        loss=distribution.loss, learning_rate=tuned["learning_rate"])
    model.fit(inputs.transform(X), outputs.transform(y), batch_size=batch,
              epochs=epochs,
              validation_data=(inputs.transform(X_validate), outputs.transform(y_validate)),
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=patience, restore_best_weights=True)],
              verbose=0)
    return Fitted(model, inputs, outputs, list(X.columns))


def _undone(found, outputs):
    """Paràmetres a les unitats del preu, desfent l'escalat de l'objectiu.

    El centre només mou la localització i la dispersió mou tota distància, de manera
    que l'escala i les localitzacions no es desfan igual. L'asimetria i el pes de la
    cua no tenen unitats i es queden.
    """
    centre, spread = outputs.mean_, np.sqrt(outputs.var_)
    undone = found.copy()
    undone[..., 0] = found[..., 0] * spread + centre
    undone[..., 1] = found[..., 1] * spread
    return undone


def parameters_rows(distribution, fitted, X):
    """Els paràmetres de cada fila en EUR/MWh, `(files, 24, paràmetres)`."""
    raw = fitted.network.predict(fitted.inputs.transform(X[fitted.columns]), verbose=0)
    return _undone(raw_parameters(distribution, raw), fitted.outputs)


def predict_rows(distribution, fitted, X):
    """El preu puntual de cada fila i hora, la mediana de la seva distribució."""
    return distribution.point(parameters_rows(distribution, fitted, X))


def attribute_rows(distribution, fitted, X, background):
    """L'atribució de cada paràmetre de cada fila, amb DeepExplainer.

    S'explica cada paràmetre tal com el dona la xarxa, i res que se'n construeixi:
    DeepExplainer no hi té cap regla per a `Log`, `Sinh` ni `Neg`, i no suma bé a
    través de cap combinació de paràmetres. La base és la sortida mitjana sobre
    `background`, que per a la part B són les files d'entrenament de la llavor. La
    localització i l'escala es tornen a EUR/MWh amb l'escalat de l'objectiu, que és
    lineal en cada una i per tant manté la suma.
    """
    explained = keras.Model(fitted.network.input,
                            keras.layers.Lambda(distribution.flat)(fitted.network.output))
    explainer = shap.DeepExplainer(
        explained, fitted.inputs.transform(background[fitted.columns]))
    values = np.asarray(explainer.shap_values(
        fitted.inputs.transform(X[fitted.columns]), check_additivity=False))
    count = len(distribution.names)
    values = values.reshape(len(X), len(fitted.columns), count, -1).transpose(0, 1, 3, 2)
    base = np.asarray(explainer.expected_value).reshape(count, -1).T

    values[..., :2] *= np.sqrt(fitted.outputs.var_)[:, np.newaxis]
    base = _undone(base, fitted.outputs)
    return Explanation(distribution.names, fitted.columns, values, base)


def fit(distribution, X, y, day, tuned=None, window=WINDOW, members=MEMBERS,
        batch=BATCH, epochs=EPOCHS, patience=PATIENCE,
        validation_weeks=VALIDATION_WEEKS, seed=SEED):
    """Les xarxes d'un dia, ajustades només amb dies anteriors.

    Cada membre aparta les seves setmanes de validació barrejant tota la finestra, que
    és el que fa l'article per predir, i ajusta amb la resta.
    """
    tuned = searched() if tuned is None else tuned
    features, target = calibration(X[columns_of(tuned["blocks"])], y, day, window)
    fitted = []
    for member in range(members):
        train, validate = network.split(len(features), validation_weeks,
                                        np.random.default_rng(seed + member))
        fitted.append(fit_rows(
            distribution, (features.iloc[train], target.iloc[train]),
            (features.iloc[validate], target.iloc[validate]), tuned, batch, epochs,
            patience, seed + member))
    return tuple(fitted)


def parameters(distribution, model, X, day):
    """Els paràmetres de la distribució d'un dia, `(24, paràmetres)`, en EUR/MWh.

    La mitjana dels membres es fa sobre els paràmetres i no sobre les distribucions:
    amb un sol membre no canvia res, i amb més d'un és la mitjana que l'article anomena
    de barreja vertical sense la barreja, cosa que caldrà decidir si els membres pugen.
    """
    return np.mean([parameters_rows(distribution, fitted, X.loc[[day]])
                    for fitted in model], axis=0)[0]


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
