"""El que comparteixen totes les xarxes de Marcjasz et al. (2023), les distribucionals i
la de punt, sobre la matriu diària: les entrades, les capes, la sortida i l'ajust.

Com les construeix el codi de l'article: les entrades normalitzades per lots com a
primera capa, sense cap escalador i amb els preus sense escalar; un dropout opcional
després de l'entrada; dues capes amagades, cada una amb la seva activació i una L1 als
pesos i una altra a la sortida; i una capa lineal per paràmetre de la sortida, cada una
amb la seva L1. Les distribucionals minimitzen una log-versemblança, i la de punt
l'error absolut.

Cada xarxa llegeix totes les columnes de la matriu, com les altres famílies, i el dia de
la setmana en una sola columna, com el seu codi el dóna a la xarxa.
"""

import collections

import keras
import numpy as np
from keras import ops

from epf.contract.matrix import DAY_OF_WEEK_COLUMNS
from epf.models.nn.hyperparameters import BATCH, EPOCHS, PATIENCE
from epf.parameters import HOUR_COLUMNS, HOURS_PER_DAY, SEED

# La columna del dia de la setmana que llegeix la xarxa, de dilluns 0 a diumenge 6, en
# lloc dels set indicadors de la matriu: és el dia que es prediu, on el codi de
# l'article en posa un altre.
DAY = "day_of_week"

# La sortida d'una xarxa, una distribució o els 24 preus de la de punt. `loss` és la
# pèrdua que minimitza, `flat` porta la sortida crua a les unitats en què es desen els
# paràmetres, `point` el preu puntual que se'n deriva, `quantiles` els quantils predits
# i `names` com es diuen els seus paràmetres, en l'ordre en què surten de la xarxa.
#
# `flat` és d'operacions de Keras i torna els paràmetres en el mateix ordre pla que la
# sortida crua: és alhora el que llegeix `parameters` i l'última capa del que
# DeepExplainer explica.
Distribution = collections.namedtuple("Distribution", "names loss flat point quantiles")


def heads(raw, count):
    """Les `count` columnes de paràmetres que la xarxa ha tret, cada una (files, 24)."""
    return [
        raw[..., number * HOURS_PER_DAY : (number + 1) * HOURS_PER_DAY]
        for number in range(count)
    ]


def parameters(distribution, raw):
    """Els paràmetres en les unitats en què es desen i es llegeixen, `(files, 24,
    P)`."""
    flat = ops.convert_to_numpy(distribution.flat(raw))
    return np.stack(heads(flat, len(distribution.names)), axis=-1)


def inputs(X):
    """Les entrades de la xarxa: totes les columnes de `X`, amb el dia de la setmana al
    davant en una sola columna, `DAY`, en lloc dels set indicadors."""
    found = X.drop(columns=DAY_OF_WEEK_COLUMNS).astype(float)
    found.insert(0, DAY, X[DAY_OF_WEEK_COLUMNS].to_numpy().argmax(axis=1).astype(float))
    return found


def build(distribution, columns, tuned):
    """La xarxa del codi de l'article per a una sortida i uns números, compilada."""
    entry = keras.Input(shape=(columns,))
    tensor = keras.layers.BatchNormalization()(entry)
    if tuned["dropout"] > 0:
        tensor = keras.layers.Dropout(tuned["dropout"])(tensor)
    for neurons, activation, weights, outputs in zip(
        tuned["hidden"],
        tuned["activation"],
        tuned["l1"],
        tuned["activity_l1"],
        strict=True,
    ):
        tensor = keras.layers.Dense(
            neurons,
            activation=activation,
            kernel_regularizer=keras.regularizers.L1(weights),
            activity_regularizer=keras.regularizers.L1(outputs),
        )(tensor)
    layers = [
        keras.layers.Dense(
            len(HOUR_COLUMNS), kernel_regularizer=keras.regularizers.L1(penalty)
        )(tensor)
        for penalty in tuned["parameter_l1"]
    ]
    if len(layers) != len(distribution.names):
        raise ValueError(
            f"{len(layers)} L1 de paràmetre per a una distribució de "
            f"{len(distribution.names)} paràmetres"
        )
    model = keras.Model(entry, keras.layers.Concatenate()(layers))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tuned["learning_rate"]),
        loss=distribution.loss,
    )
    return model


def fit_rows(
    distribution,
    train,
    validate,
    tuned,
    batch=BATCH,
    epochs=EPOCHS,
    patience=PATIENCE,
    seed=SEED,
):
    """Una xarxa ajustada amb les files de `train` i aturada quan deixa de millorar a
    les de `validate`, totes dues parelles `(X, y)`."""
    (X, y), (X_validate, y_validate) = train, validate
    values = inputs(X).to_numpy()
    # L'estat que Keras guarda entre ajustos, buidat: sense això un ajust amb dropout
    # depèn del que el mateix procés va ajustar abans.
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build(distribution, values.shape[1], tuned)
    model.fit(
        values,
        y.to_numpy(dtype=float),
        batch_size=batch,
        epochs=epochs,
        validation_data=(
            inputs(X_validate).to_numpy(),
            y_validate.to_numpy(dtype=float),
        ),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    return model


def parameters_rows(distribution, fitted, X):
    """Els paràmetres de cada fila, `(files, 24, paràmetres)`."""
    raw = fitted.predict(inputs(X).to_numpy(), verbose=0)
    return parameters(distribution, raw)


def predict_rows(distribution, fitted, X):
    """El preu puntual de cada fila i hora, el de la seva sortida."""
    return distribution.point(parameters_rows(distribution, fitted, X))
