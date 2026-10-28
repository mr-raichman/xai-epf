"""La xarxa neuronal, la xarxa de punt de Marcjasz et al. (2023).

Es construeix com al codi de l'article, sobre la matriu diària.
"""

import keras

from epf.contract.matrix import DAY_OF_WEEK, DAY_OF_WEEK_COLUMNS
from epf.models.dnn.hyperparameters import EPOCHS, PATIENCE, PATIENCE_BATCH
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
    # L'únic escalador, com al codi de l'article; el que prediu no s'escala.
    tensor = keras.layers.BatchNormalization()(entry)
    for neurons, activation in zip(tuned["hidden"], tuned["activation"], strict=True):
        tensor = keras.layers.Dense(neurons, activation=activation)(tensor)
    model = keras.Model(entry, keras.layers.Dense(HOURS_PER_DAY)(tensor))
    # L'error absolut, la pèrdua `'mae'` del codi de l'article.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tuned["learning_rate"]),
        loss=keras.losses.mean_absolute_error,
    )
    return model


def patience_at(batch):
    """Torna la paciència de l'early stopping en èpoques a un batch de `batch` dies.

    Són les èpoques que fan tantes actualitzacions dels pesos com `PATIENCE` èpoques a
    `PATIENCE_BATCH`.
    """
    return PATIENCE * batch // PATIENCE_BATCH


def stopped(epochs, batch):
    """Torna per què s'atura un ajust que es queda l'època `epochs`, i quantes en fa.

    Per `"ceiling"`, `EPOCHS`, si amb la paciència del seu batch, `batch`, el passaria,
    i si no per `"patience"`.
    """
    if epochs + patience_at(batch) > EPOCHS:
        return "ceiling", EPOCHS
    return "patience", epochs + patience_at(batch)


def fit_rows(train, validate, tuned, seed, epochs=EPOCHS, patience=None):
    """Ajusta una xarxa amb `train`, amb early stopping sobre `validate`.

    Tots dos són parelles `(X, y)`. El batch és el de `tuned`, i `patience`, en èpoques,
    la d'aquest batch si és `None`. Torna la xarxa amb els pesos de la millor època, i
    aquesta època, comptada des d'1. Buida l'estat de Keras i fixa totes les seeds
    globals amb la de l'ajust de la partició `seed`.
    """
    (X, y), (X_validate, y_validate) = train, validate
    values = inputs(X).to_numpy()
    # Allibera l'estat global de Keras, que segons la seva documentació creix en memòria
    # amb cada model que el procés construeix.
    keras.backend.clear_session()
    keras.utils.set_random_seed(seeded("fit", seed))
    model = build(values.shape[1], tuned)
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_at(tuned["batch"]) if patience is None else patience,
        restore_best_weights=True,
    )
    model.fit(
        values,
        y.to_numpy(dtype=float),
        batch_size=tuned["batch"],
        epochs=epochs,
        validation_data=(
            inputs(X_validate).to_numpy(),
            y_validate.to_numpy(dtype=float),
        ),
        callbacks=[stop],
        verbose=0,
    )
    return model, stop.best_epoch + 1


def predict_rows(fitted, X):
    """Torna la predicció de cada fila de `X` i cada hora, `(files, 24)`."""
    return fitted.predict(inputs(X).to_numpy(), verbose=0)
