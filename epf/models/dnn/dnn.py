"""La xarxa de l'article sobre la matriu diària: una fila per dia i 24 sortides.

Com la construeix el codi de Lago et al. (2021), que és el que va produir les seves
xifres, i allà on el codi i el text discrepen mana el codi: una activació i una L1 per
a tota la xarxa, la sortida inclosa; dues capes amagades, la segona només si en té
prou neurones; les entrades i la sortida escalades amb un dels sis escalats del codi;
Adam; la seva aturada; i **pèrdua absoluta**, que és el que l'article minimitza i el que
fa que l'error que es mesura i el que s'optimitza siguin el mateix. Marc multivariant:
un sol model dóna les 24 hores alhora.

Aquesta família no s'atribueix. Hi és per dir com de lluny són els models que
s'expliquen del que es pot arribar a fer amb aquestes dades: és la mesura de com de
creïbles són les heurístiques que se'ls llegeixen, i no un model del qual s'esperi cap
explicació.

**Els números que la defineixen els tria la cerca i no una persona**, i `fit` els pren
en un objecte per membre perquè és així com es trien, es desen i se substitueixen: tots
els d'un membre junts, de la seva cerca.

Tot és sobre files donades: `fit_rows` ajusta una xarxa amb les files d'entrenament i
les de validació que se li donin, i `fit`, el camí endavant, les treu de la finestra
amb `models.search.split`, barrejant setmanes senceres. Les entrades, per blocs i amb el
dia en una sola columna, són les de `models/blocks.py`, que la xarxa distribucional
llegeix igual.
"""
import collections

import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from epf.calendar import HOUR_COLUMNS
from epf.matrix import calibration
from epf.models.blocks import inputs
from epf.models.dnn.dnn_hyperparameters import (BATCH, CLIP, EPOCHS, NAME, PATIENCE,
                                                SMALLEST_LAYER, WINDOW, searched)
from epf.models.scaling import Median, Scaler
from epf.models.search import split

# Una xarxa ajustada i el que cal per llegir-la: els escaladors de les entrades i de la
# sortida, el segon el que torna a EUR/MWh.
Fitted = collections.namedtuple("Fitted", "network inputs outputs")

# Un membre del camí endavant: la seva xarxa ajustada i els blocs que llegeix, que són
# els de la seva cerca i no els d'un altre membre.
Member = collections.namedtuple("Member", "fitted blocks")

# Les activacions que el codi de Lago et al. (2021) posa com una capa a part, darrere
# d'una `Dense` lineal, amb el pendent negatiu del seu `LeakyReLU`.
SEPARATE = {"LeakyReLU": lambda: keras.layers.LeakyReLU(negative_slope=0.001),
            "PReLU": keras.layers.PReLU}

# L'activació amb la qual el codi de l'article apaga neurones amb `AlphaDropout` i no
# amb `Dropout`, el que manté la mitjana i la variància que `selu` necessita, i amb la
# qual imposa `SELU_INITIALISER` a totes les capes, la sortida inclosa, sigui quina sigui
# la inicialització que s'hagi cercat.
SELU = "selu"
SELU_INITIALISER = "lecun_normal"


class _Unscaled:
    """L'escalat `No`: les files tal com són."""

    def __init__(self, values):
        pass

    def forward(self, values):
        return values

    def back(self, values):
        return values


class _Fitted:
    """Un escalador de scikit-learn ajustat, amb els noms dels de `models.scaling`."""

    def __init__(self, scaler, values):
        self.scaler = scaler.fit(values)

    def forward(self, values):
        return self.scaler.transform(values)

    def back(self, values):
        return self.scaler.inverse_transform(values)


# Els sis escalats entre els quals tria la cerca, pels noms que els dóna el codi de Lago
# et al. (2021), cada un ajustat amb les files que se li donen i aplicat columna a
# columna: cap, a [0, 1], a [-1, 1], estandarditzat, centrat per la mediana i dividit per
# la MAD, i l'asinh d'aquest últim, que és el de LEAR.
SCALINGS = {
    "No": _Unscaled,
    "Norm": lambda values: _Fitted(MinMaxScaler((0, 1)), values),
    "Norm1": lambda values: _Fitted(MinMaxScaler((-1, 1)), values),
    "Std": lambda values: _Fitted(StandardScaler(), values),
    "Median": Median,
    "Invariant": Scaler,
}


def build(columns, tuned):
    """La xarxa del codi de Lago et al. (2021) amb uns números, compilada.

    Cada capa amagada és `Dense` amb l'activació a dins, o lineal i l'activació darrere si
    és de `SEPARATE`; després la normalització per lots si hi és, i el dropout al final.
    **La normalització va després de l'activació i no abans**, que és el que diu la seva
    implementació i el contrari del que se'n diria de memòria. Una sola activació i una
    sola L1 per a tota la xarxa, la sortida inclosa, i una capa de menys de
    `SMALLEST_LAYER` neurones no hi és.
    """
    activation = tuned["activation"]
    initialiser = SELU_INITIALISER if activation == SELU else tuned["initialiser"]
    entry = keras.Input(shape=(columns,))
    tensor = entry
    for neurons in tuned["hidden"]:
        if neurons < SMALLEST_LAYER:
            continue
        tensor = keras.layers.Dense(
            neurons, activation=None if activation in SEPARATE else activation,
            kernel_initializer=initialiser,
            kernel_regularizer=keras.regularizers.L1(tuned["l1"]))(tensor)
        if activation in SEPARATE:
            tensor = SEPARATE[activation]()(tensor)
        if tuned["batch_normalisation"]:
            tensor = keras.layers.BatchNormalization()(tensor)
        if tuned["dropout"] > 0:
            switch = keras.layers.AlphaDropout if activation == SELU else keras.layers.Dropout
            tensor = switch(tuned["dropout"])(tensor)
    output = keras.layers.Dense(len(HOUR_COLUMNS), kernel_initializer=initialiser,
                                kernel_regularizer=keras.regularizers.L1(tuned["l1"]))(tensor)
    model = keras.Model(entry, output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=tuned["learning_rate"],
                                                  clipvalue=CLIP),
                  loss="mae")
    return model


def improved(best, loss, error):
    """El millor parell de pèrdua i error després d'una època, o None si no ha millorat.

    És la regla del codi de Lago et al. (2021): l'època compta com a millora si baixa la
    pèrdua o si baixa l'error en EUR/MWh. Quan baixa la pèrdua, el millor error passa a
    ser el d'aquella època encara que sigui més alt, que és el que fa el codi.
    """
    if loss < best[0]:
        return loss, error
    if error < best[1]:
        return best[0], error
    return None


class Stopping(keras.callbacks.Callback):
    """L'aturada del codi de Lago et al. (2021), on la de Keras mira només la pèrdua.

    Cada època mira dues coses a les files de validació: la pèrdua de la xarxa, a
    l'espai escalat i amb la L1, i l'error absolut mitjà en EUR/MWh, desfent l'escalat
    de la sortida. Amb un escalat que no és lineal totes dues poden discrepar sobre quina
    època és la millor. Quan cap de les dues no millora en `patience` èpoques s'atura, i
    la xarxa es queda els pesos de l'última època que va millorar.
    """

    def __init__(self, rows, prices, outputs, patience):
        super().__init__()
        self.rows, self.prices, self.outputs = rows, prices, outputs
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.best = (np.inf, np.inf)
        self.waited = 0
        self.weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        predicted = self.outputs.back(self.model.predict(self.rows, verbose=0))
        found = improved(self.best, logs["val_loss"],
                         np.mean(np.abs(predicted - self.prices)))
        if found is None:
            self.waited += 1
            self.model.stop_training = self.waited >= self.patience
        else:
            self.best, self.waited, self.weights = found, 0, self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)


def prices(fitted, values):
    """Els preus en EUR/MWh que dóna una xarxa ajustada per a les entrades `values`."""
    return fitted.outputs.back(fitted.network.predict(fitted.inputs.forward(values),
                                                      verbose=0))


def fit_rows(train, validate, tuned, seed, batch=BATCH, epochs=EPOCHS,
             patience=PATIENCE):
    """Una xarxa ajustada amb les files de `train` i aturada amb les de `validate`,
    totes dues parelles `(entrades, preus)` en arrays.

    Els escaladors s'ajusten només amb les d'entrenament, com al codi de l'article, i
    les de validació s'hi escalen i prou: l'aturada mira files que l'ajust no ha vist de
    cap manera. `seed` fixa tot l'atzar de l'ajust.
    """
    (X, y), (X_validate, y_validate) = train, validate
    scaled_inputs = SCALINGS[tuned["scale_inputs"]](X)
    outputs = SCALINGS[tuned["scale_outputs"]](y)
    keras.utils.set_random_seed(seed)
    model = build(X.shape[1], tuned)
    rows = scaled_inputs.forward(X_validate)
    model.fit(scaled_inputs.forward(X), outputs.forward(y), batch_size=batch,
              epochs=epochs, validation_data=(rows, outputs.forward(y_validate)),
              callbacks=[Stopping(rows, y_validate, outputs, patience)], verbose=0)
    return Fitted(model, scaled_inputs, outputs)


def fit(X, y, day, members=None, window=WINDOW, batch=BATCH, epochs=EPOCHS,
        patience=PATIENCE):
    """Les xarxes d'un dia, una per membre, ajustades només amb dies anteriors.

    `members` són els números de cada membre, que per defecte són els de `searched()`.
    Cada un porta els de la seva pròpia cerca, com a l'article: els seus blocs, la seva
    xarxa i la seva llavor, que fixa també quines setmanes s'aparten, cosa que el codi fa
    amb la mateixa llavor cercada.
    """
    members = searched()["members"] if members is None else members
    features, target = calibration(X, y, day, window)
    found = target.to_numpy(dtype=float)
    fitted = []
    for tuned in members:
        values = inputs(features, tuned["blocks"]).to_numpy()
        train, validate = split(len(values), np.random.default_rng(tuned["seed"]))
        fitted.append(Member(fit_rows((values[train], found[train]),
                                      (values[validate], found[validate]), tuned,
                                      tuned["seed"], batch, epochs, patience),
                             tuned["blocks"]))
    return fitted


def predict(model, X, day):
    """Els 24 preus de `day`, la mitjana de les xarxes dels membres, per nom."""
    return {NAME: np.mean([prices(member.fitted,
                                  inputs(X.loc[[day]], member.blocks).to_numpy())[0]
                           for member in model], axis=0)}
