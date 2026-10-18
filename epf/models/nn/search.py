"""La cerca de les xarxes, les distribucionals i la de punt: l'espai del codi de
Marcjasz et al. (2023), `nn.py`, sense els seus selectors de blocs d'entrada, i com s'hi
prova cada configuració. Cada xarxa llegeix totes les columnes, com les altres
famílies."""

from epf.models.nn import network
from epf.models.nn.ddnn.distributions import DISTRIBUTIONS
from epf.models.nn.dnn.train import POINT, POINT_FAMILY

# La matriu de la cerca, la diària, com la de la passada.
MATRIX = "daily"

# Un espai continu, que es mostreja a l'atzar, i no cap graella.
GRID = None

# L'espai del codi de Marcjasz et al. (2023): dues capes amagades, les neurones de cada
# una, les sis activacions pels seus noms, el dropout de després de l'entrada, el
# coeficient de cada L1, tots en el mateix interval logarítmic, i la taxa
# d'aprenentatge, també logarítmica.
LAYERS = 2
NEURONS = (16, 1024)
ACTIVATIONS = ("sigmoid", "relu", "elu", "tanh", "softplus", "softmax")
DROPOUT = (0.0, 1.0)
L1 = (1e-5, 10.0)
LEARNING_RATE = (1e-5, 0.1)


def penalty(optuna_trial, name):
    """Una L1 com la demana el codi: primer si n'hi ha, i el coeficient només si n'hi
    ha. Sense, és zero, que és una L1 que no fa res."""
    if not optuna_trial.suggest_categorical(f"regularize_{name}", (False, True)):
        return 0.0
    return optuna_trial.suggest_float(f"{name}_l1", *L1, log=True)


def space(optuna_trial, distribution, penalised=True):
    """Els hiperparàmetres d'un intent, en la forma en què els llegeix `network.build`.

    Sense `penalised`, la capa de sortida no té cap L1 i no se'n cerca cap, que és la
    xarxa de punt de l'article.

    L'L1 de cada paràmetre es demana pel nom que té a `distribution`, i en el seu ordre:
    el codi de l'article posa el pes de la cua de Johnson SU abans de l'asimetria.
    """
    layers = range(1, LAYERS + 1)
    return {
        "hidden": [
            optuna_trial.suggest_int(f"neurons_{layer}", *NEURONS) for layer in layers
        ],
        "activation": [
            optuna_trial.suggest_categorical(f"activation_{layer}", ACTIVATIONS)
            for layer in layers
        ],
        "dropout": (
            optuna_trial.suggest_float("dropout_rate", *DROPOUT)
            if optuna_trial.suggest_categorical("dropout", (False, True))
            else 0.0
        ),
        "l1": [penalty(optuna_trial, f"h{layer}_kernel") for layer in layers],
        "activity_l1": [
            penalty(optuna_trial, f"h{layer}_activation") for layer in layers
        ],
        "parameter_l1": (
            [penalty(optuna_trial, name) for name in distribution.names]
            if penalised
            else [0.0] * len(distribution.names)
        ),
        "learning_rate": optuna_trial.suggest_float(
            "learning_rate", *LEARNING_RATE, log=True
        ),
    }


def trial(family, optuna_trial, train, validate, seed):
    """Una configuració de l'espai per a la sortida de la família, ajustada amb les
    files d'entrenament, `(X, y)`, aturada sobre tota la validació i puntuada per la
    pèrdua de Keras sobre ella, amb les L1 a dins, que és el que torna el codi de
    l'article. La xarxa de punt no cerca cap L1 a la sortida."""
    X_validate, y_validate = validate
    point = family == POINT_FAMILY
    distribution = POINT if point else DISTRIBUTIONS[family]
    tuned = space(optuna_trial, distribution, penalised=not point)
    fitted = network.fit_rows(distribution, train, validate, tuned)
    return tuned, float(
        fitted.evaluate(
            network.inputs(X_validate).to_numpy(),
            y_validate.to_numpy(dtype=float),
            verbose=0,
        )
    )
