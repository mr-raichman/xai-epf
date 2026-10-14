"""Les xarxes distribucionals de Marcjasz et al. (2023), sobre la matriu diària.

Com les construeix el codi de l'article, que és el que va produir les seves xifres: les
entrades normalitzades per lots com a primera capa, sense cap escalador i amb els preus
sense escalar; un dropout opcional després de l'entrada; dues capes amagades, cada una
amb la seva activació i una L1 als pesos i una altra a la sortida; i una capa lineal per
paràmetre de la distribució, cada una amb la seva L1. Minimitzen una log-versemblança.

Una família per distribució, que és el que fa l'article, i la diferència entre les dues
és la seva pròpia evidència que els moments alts compten.

**La predicció puntual és la mediana de la distribució**, que és el que minimitza
l'error absolut i el que l'article fa servir per al MAE. Els paràmetres es desen a part,
perquè una distribució no són 24 preus.
"""
import collections

import keras
import numpy as np
import shap

from epf.calendar import HOUR_COLUMNS
from epf.models.ddnn.blocks import blocks, inputs
from epf.models.ddnn.ddnn_hyperparameters import BATCH, EPOCHS, PATIENCE
from epf.models.ddnn.distributions import parameters as raw_parameters
from epf.parameters import SEED

# Una xarxa ajustada i els blocs que llegeix, que són els que la tornen a llegir.
Fitted = collections.namedtuple("Fitted", "network names")

# L'atribució de cada paràmetre: `values` és `(files, columnes, 24, paràmetres)`, `base`
# `(24, paràmetres)` i `inputs` les entrades explicades, una columna per a cada una de
# `columns`, en les unitats en què es desa cada paràmetre: EUR/MWh la localització i
# l'escala, i sense unitats l'asimetria i el pes de la cua.
Explanation = collections.namedtuple("Explanation", "names columns inputs values base")


def build(distribution, columns, tuned):
    """La xarxa del codi de l'article per a una distribució i uns números, compilada."""
    entry = keras.Input(shape=(columns,))
    tensor = keras.layers.BatchNormalization()(entry)
    if tuned["dropout"] > 0:
        tensor = keras.layers.Dropout(tuned["dropout"])(tensor)
    for neurons, activation, weights, outputs in zip(
            tuned["hidden"], tuned["activation"], tuned["l1"], tuned["activity_l1"],
            strict=True):
        tensor = keras.layers.Dense(
            neurons, activation=activation,
            kernel_regularizer=keras.regularizers.L1(weights),
            activity_regularizer=keras.regularizers.L1(outputs))(tensor)
    heads = [keras.layers.Dense(len(HOUR_COLUMNS),
                                kernel_regularizer=keras.regularizers.L1(penalty))(tensor)
             for penalty in tuned["parameter_l1"]]
    if len(heads) != len(distribution.names):
        raise ValueError(f"{len(heads)} L1 de paràmetre per a una distribució de "
                         f"{len(distribution.names)} paràmetres")
    model = keras.Model(entry, keras.layers.Concatenate()(heads))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=tuned["learning_rate"]),
                  loss=distribution.loss)
    return model


def fit_rows(distribution, train, validate, tuned, batch=BATCH, epochs=EPOCHS,
             patience=PATIENCE, seed=SEED):
    """Una xarxa ajustada amb les files de `train` i aturada quan deixa de millorar a
    les de `validate`, totes dues parelles `(X, y)`, sobre tots els blocs de `X`."""
    (X, y), (X_validate, y_validate) = train, validate
    names = list(blocks(X.columns))
    values = inputs(X, names).to_numpy()
    keras.utils.set_random_seed(seed)
    model = build(distribution, values.shape[1], tuned)
    model.fit(values, y.to_numpy(dtype=float), batch_size=batch,
              epochs=epochs,
              validation_data=(inputs(X_validate, names).to_numpy(),
                               y_validate.to_numpy(dtype=float)),
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=patience, restore_best_weights=True)],
              verbose=0)
    return Fitted(model, names)


def parameters_rows(distribution, fitted, X):
    """Els paràmetres de cada fila, `(files, 24, paràmetres)`."""
    raw = fitted.network.predict(inputs(X, fitted.names).to_numpy(), verbose=0)
    return raw_parameters(distribution, raw)


def predict_rows(distribution, fitted, X):
    """El preu puntual de cada fila i hora, la mediana de la seva distribució."""
    return distribution.point(parameters_rows(distribution, fitted, X))


def attribute_rows(distribution, fitted, X, background):
    """L'atribució de cada paràmetre de cada fila, amb DeepExplainer.

    S'explica cada paràmetre tal com el dóna la xarxa, i res que se'n construeixi:
    DeepExplainer no hi té cap regla per a `Log`, `Sinh` ni `Neg`, i no suma bé a
    través de cap combinació de paràmetres. La base és la sortida mitjana sobre
    `background`, que a la passada són les files d'entrenament de la llavor. Com que
    la xarxa no escala el preu, la localització i l'escala surten ja en EUR/MWh.
    """
    explained = keras.Model(fitted.network.input,
                            keras.layers.Lambda(distribution.flat)(fitted.network.output))
    # shap avisa que retalla el fons a 100 files, però ho fa l'emmascarador de la seva
    # classe base, que aquest explicador no fa servir: el càlcul les pren totes.
    explainer = shap.DeepExplainer(
        explained, inputs(background, fitted.names).to_numpy())
    shown = inputs(X, fitted.names)
    # Keras avisa que l'estructura de les entrades no és l'esperada perquè DeepExplainer
    # li passa el tensor dins d'una llista; el model el pren igual, i la suma de cada
    # paràmetre, que es comprova, ho confirma.
    values = np.asarray(explainer.shap_values(shown.to_numpy(), check_additivity=False))
    count = len(distribution.names)
    values = values.reshape(len(X), shown.shape[1], count, -1).transpose(0, 1, 3, 2)
    base = np.asarray(explainer.expected_value).reshape(count, -1).T
    return Explanation(distribution.names, list(shown.columns), shown.to_numpy(), values,
                       base)
