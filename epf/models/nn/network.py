"""La xarxa de Marcjasz et al. (2023), com la construeix el codi de l'article.

Les distribucionals i la de punt la fan servir sobre la matriu diària, cada una amb la
seva sortida.
"""

import collections

import keras
import numpy as np
from keras import ops

from epf.contract.matrix import DAY_OF_WEEK_COLUMNS
from epf.models.nn.hyperparameters import BATCH, EPOCHS, PATIENCE
from epf.parameters import HOUR_COLUMNS, HOURS_PER_DAY, SEED

# El dia de la setmana del dia que es prediu, de dilluns 0 a diumenge 6, en una sola
# columna com al codi de l'article, que per error hi posa una funció del número de fila.
DAY = "day_of_week"

# La sortida d'una xarxa, una distribució o els 24 preus de la de punt. `names` són els
# seus paràmetres, en l'ordre en què surten de la xarxa; `loss`, la pèrdua que
# minimitza; `flat`, la sortida crua portada a les unitats en què es desen els
# paràmetres, en el mateix ordre pla; `point`, el preu puntual que se'n deriva; i
# `quantiles`, els quantils predits. `flat` és d'operacions de Keras perquè també és
# l'última capa del que explica DeepExplainer.
Distribution = collections.namedtuple("Distribution", "names loss flat point quantiles")


def heads(raw, count):
    """Torna els `count` paràmetres d'una sortida de la xarxa, cada un `(files, 24)`."""
    return [
        raw[..., number * HOURS_PER_DAY : (number + 1) * HOURS_PER_DAY]
        for number in range(count)
    ]


def parameters(distribution, raw):
    """Torna els paràmetres de la sortida crua, en les unitats en què es desen.

    Són `(files, 24, paràmetres)`.
    """
    flat = ops.convert_to_numpy(distribution.flat(raw))
    return np.stack(heads(flat, len(distribution.names)), axis=-1)


def inputs(X):
    """Torna les entrades de la xarxa, `X` amb `DAY` en lloc de les set dummies."""
    found = X.drop(columns=DAY_OF_WEEK_COLUMNS).astype(float)
    found.insert(0, DAY, X[DAY_OF_WEEK_COLUMNS].to_numpy().argmax(axis=1).astype(float))
    return found


def build(distribution, columns, tuned):
    """Torna la xarxa compilada d'una sortida, `distribution`, i uns números, `tuned`.

    Falla si `tuned` no porta una L1 de paràmetre per cada paràmetre de la sortida.
    """
    entry = keras.Input(shape=(columns,))
    # L'únic escalador, com al codi de l'article; els preus que prediu no s'escalen.
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
    """Ajusta una xarxa amb `train`, amb early stopping sobre `validate`.

    Tots dos són parelles `(X, y)`. Torna la xarxa amb els pesos de la millor època.
    Buida l'estat de Keras i fixa amb `seed` totes les seeds globals.
    """
    (X, y), (X_validate, y_validate) = train, validate
    values = inputs(X).to_numpy()
    # Sense això, un ajust amb dropout depèn del que el mateix procés va ajustar abans.
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
    """Torna els paràmetres de cada fila de `X`, `(files, 24, paràmetres)`."""
    raw = fitted.predict(inputs(X).to_numpy(), verbose=0)
    return parameters(distribution, raw)


def predict_rows(distribution, fitted, X):
    """Torna el preu puntual de cada fila i hora, el de la seva sortida."""
    return distribution.point(parameters_rows(distribution, fitted, X))
