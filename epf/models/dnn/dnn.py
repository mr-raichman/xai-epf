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
en un sol objecte perquè és així com es trien, es desen i se substitueixen: tots junts.

Tot és sobre files donades: `fit_rows` ajusta una xarxa amb les files d'entrenament i
les de validació que se li donin, i `fit`, el camí endavant, les treu de la finestra
amb `split`.
"""
import collections

import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from epf.calendar import HOUR_COLUMNS
from epf.matrix import DAY_OF_WEEK_COLUMNS, calibration
from epf.models.dnn import network
from epf.models.dnn.dnn_hyperparameters import (BATCH, CLIP, EPOCHS, MEMBERS, NAME,
                                                PATIENCE, SMALLEST_LAYER,
                                                VALIDATION_SHARE, WINDOW, searched)
from epf.models.scaling import Median, Scaler
from epf.parameters import DAYS_PER_WEEK

# Una xarxa ajustada i el que cal per llegir-la: els escaladors de les entrades i de la
# sortida, el segon el que torna a EUR/MWh.
Fitted = collections.namedtuple("Fitted", "network inputs outputs")

# El bloc del dia de la setmana, que la matriu escriu en set indicadors i aquesta xarxa
# llegeix en una sola columna.
DAY = "day_of_week"

# La inicialització que el codi de Lago et al. (2021) imposa a totes les capes, la
# sortida inclosa, quan l'activació és `selu`, sigui quina sigui la que s'hagi cercat.
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


def blocks(columns):
    """Les columnes de la matriu per bloc: cada sèrie a cada retard, amb les seves 24
    hores, i el dia de la setmana.

    Són el que la cerca de Lago et al. (2021) engega i apaga, onze al seu conjunt: el
    dia, els quatre retards del preu, i les dues previsions al dia que es lliura, al
    d'abans i a fa una setmana. El bloc és el nom de la columna sense el que va després
    de l'últim guió baix, que és l'hora o el número del dia, de manera que surten de la
    matriu i no d'una llista que se'n pugui separar.
    """
    found = {}
    for column in columns:
        found.setdefault(column.rsplit("_", 1)[0], []).append(column)
    return found


def columns_of(columns, names):
    """Les columnes dels blocs anomenats, en l'ordre de la matriu."""
    found = blocks(columns)
    return [column for name in found if name in names for column in found[name]]


def inputs(X, names):
    """Les entrades dels blocs anomenats, amb el dia de la setmana al davant en una sola
    columna, de dilluns 0 a diumenge 6, com el codi de l'article el dóna a la xarxa.

    Els set indicadors de la matriu no hi entren: escalats per la mediana, cada un es
    dividiria per una MAD de zero.
    """
    kept = [column for column in columns_of(X.columns, names)
            if column not in DAY_OF_WEEK_COLUMNS]
    values = X[kept].to_numpy(dtype=float)
    if DAY in names:
        day = X[DAY_OF_WEEK_COLUMNS].to_numpy().argmax(axis=1)
        values = np.column_stack([day, values])
    return values


def split(count, generator=None):
    """Les files d'entrenament i les de validació d'una finestra de `count` dies, com
    les parteix el codi de Lago et al. (2021).

    La validació és la `VALIDATION_SHARE` final de les files. Amb `generator`, abans s'hi
    barregen setmanes senceres, de set files a comptar de la primera, perquè dos dies
    veïns no quedin un a cada banda: és el que fa el codi en recalibrar, i la xarxa
    s'entrena també amb els dies més recents. Sense, l'ordre és el del calendari, que és
    el de la cerca. El tall és a un nombre de files i no de setmanes, de manera que una
    setmana pot quedar partida entre les dues bandes, com al codi.
    """
    order = np.arange(count)
    if generator is not None:
        starts = generator.permutation(order[::DAYS_PER_WEEK])
        order = np.concatenate([order[start:start + DAYS_PER_WEEK] for start in starts])
    validation = int(VALIDATION_SHARE * count)
    return order[:count - validation], order[count - validation:]


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
    selu = tuned["activation"] == network.SELU
    initialiser = SELU_INITIALISER if selu else tuned["initialiser"]
    hidden = [neurons for neurons in tuned["hidden"] if neurons >= SMALLEST_LAYER]
    model = network.build(
        X.shape[1], hidden, [tuned["activation"]] * len(hidden), tuned["dropout"],
        tuned["batch_normalisation"], [tuned["l1"]] * len(hidden), initialiser,
        head=lambda tensor: keras.layers.Dense(
            len(HOUR_COLUMNS), kernel_initializer=initialiser,
            kernel_regularizer=keras.regularizers.L1(tuned["l1"]))(tensor),
        loss="mae", learning_rate=tuned["learning_rate"], clip=CLIP)
    rows = scaled_inputs.forward(X_validate)
    model.fit(scaled_inputs.forward(X), outputs.forward(y), batch_size=batch,
              epochs=epochs, validation_data=(rows, outputs.forward(y_validate)),
              callbacks=[Stopping(rows, y_validate, outputs, patience)], verbose=0)
    return Fitted(model, scaled_inputs, outputs)


def fit(X, y, day, tuned=None, window=WINDOW, members=MEMBERS, batch=BATCH,
        epochs=EPOCHS, patience=PATIENCE):
    """Les xarxes d'un dia, ajustades només amb dies anteriors, i els blocs que llegeixen.

    `members` xarxes. A l'article cada membre surt d'una cerca pròpia, amb la seva
    llavor; aquí la cerca encara no s'ha repetit, de manera que el que varia entre
    membres és la llavor i prou, la cercada més el número del membre. La llavor fixa
    també quines setmanes s'aparten, cosa que el codi fa amb la mateixa llavor cercada.
    """
    tuned = searched() if tuned is None else tuned
    features, target = calibration(X, y, day, window)
    values, prices = inputs(features, tuned["blocks"]), target.to_numpy(dtype=float)
    fitted = []
    for member in range(members):
        seed = tuned["seed"] + member
        train, validate = split(len(values), np.random.default_rng(seed))
        fitted.append(fit_rows((values[train], prices[train]),
                               (values[validate], prices[validate]), tuned, seed,
                               batch, epochs, patience))
    return fitted, tuned["blocks"]


def predict(model, X, day):
    """Els 24 preus de `day`, la mitjana de les xarxes del model, per nom."""
    fitted, names = model
    row = inputs(X.loc[[day]], names)
    return {NAME: np.mean([one.outputs.back(
        one.network.predict(one.inputs.forward(row), verbose=0))[0] for one in fitted],
        axis=0)}
