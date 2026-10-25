"""La xarxa neuronal, la xarxa de punt de Marcjasz et al. (2023).

Es construeix com al codi de l'article, sobre la matriu diària.
"""

import keras

from epf.contract.matrix import DAY_OF_WEEK, DAY_OF_WEEK_COLUMNS
from epf.models.dnn.hyperparameters import BATCH, EPOCHS, PATIENCE
from epf.parameters import HOURS_PER_DAY, seeded


def inputs(X):
    """Torna les entrades de la xarxa, `X` amb `DAY_OF_WEEK` per les set dummies."""
    found = X.drop(columns=DAY_OF_WEEK_COLUMNS).astype(float)
    # El dia de la setmana del dia que es prediu, de dilluns 0 a diumenge 6, en una sola
    # columna com al codi de l'article, que per error hi posa una funció del número de
    # fila.
    found.insert(
        0, DAY_OF_WEEK, X[DAY_OF_WEEK_COLUMNS].to_numpy().argmax(axis=1).astype(float)
    )
    return found


def build(columns, tuned):
    """Torna la xarxa compilada d'uns números, `tuned`, sobre `columns` entrades."""
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
    model = keras.Model(entry, keras.layers.Dense(HOURS_PER_DAY)(tensor))
    # L'error absolut, la pèrdua `'mae'` del codi de l'article.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tuned["learning_rate"]),
        loss=keras.losses.mean_absolute_error,
    )
    return model


def fit_rows(
    train,
    validate,
    tuned,
    seed,
    batch=BATCH,
    epochs=EPOCHS,
    patience=PATIENCE,
):
    """Ajusta una xarxa amb `train`, amb early stopping sobre `validate`.

    Tots dos són parelles `(X, y)`. Torna la xarxa amb els pesos de la millor època.
    Buida l'estat de Keras i fixa totes les seeds globals amb la de l'ajust de la
    partició `seed`.
    """
    (X, y), (X_validate, y_validate) = train, validate
    values = inputs(X).to_numpy()
    # Sense això, un ajust amb dropout depèn del que el mateix procés va ajustar abans.
    keras.backend.clear_session()
    keras.utils.set_random_seed(seeded("fit", seed))
    model = build(values.shape[1], tuned)
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


def predict_rows(fitted, X):
    """Torna el preu de cada fila de `X` i cada hora, `(files, 24)`."""
    return fitted.predict(inputs(X).to_numpy(), verbose=0)
