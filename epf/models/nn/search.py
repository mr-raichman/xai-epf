"""La cerca de les xarxes, sobre l'espai del codi de Marcjasz et al. (2023)."""

from epf.models.nn import network
from epf.models.nn.ddnn.distributions import DISTRIBUTIONS
from epf.models.nn.dnn.train import POINT, POINT_FAMILY

# La matriu de la cerca, la diària, com la de la passada.
MATRIX = "daily"

# Sense graella: l'espai es mostreja a l'atzar.
GRID = None

# L'espai de `nn.py`, sense els seus selectors de blocs d'entrada: cada xarxa llegeix
# totes les columnes.
LAYERS = 2
NEURONS = (16, 1024)
ACTIVATIONS = ("sigmoid", "relu", "elu", "tanh", "softplus", "softmax")
DROPOUT = (0.0, 1.0)
L1 = (1e-5, 10.0)
LEARNING_RATE = (1e-5, 0.1)


def penalty(optuna_trial, name):
    """Torna el coeficient d'una L1 de l'intent, o zero si no en vol cap.

    Com al codi de l'article, l'intent tria primer si n'hi ha, i el coeficient només si
    n'hi ha.
    """
    if not optuna_trial.suggest_categorical(f"regularize_{name}", (False, True)):
        return 0.0
    return optuna_trial.suggest_float(f"{name}_l1", *L1, log=True)


def space(optuna_trial, distribution, penalised=True):
    """Torna els números d'un intent, en la forma en què els llegeix `network.build`.

    Sense `penalised`, com a la xarxa de punt de l'article, la sortida no té cap L1 i no
    se'n cerca cap. L'L1 de cada paràmetre es demana pel nom que té a `distribution`, i
    en el seu ordre, i no en el del codi de l'article, que posa el pes de la cua de
    Johnson SU abans de l'asimetria.
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
    """Ajusta una configuració de l'espai i la puntua per la pèrdua de la validació.

    S'ajusta amb la sortida de la família i les files d'entrenament, `train`, amb early
    stopping sobre tota la validació, `validate`. La pèrdua és la de Keras, amb les L1 a
    dins, com la que puntua el codi de l'article. Torna la configuració i la puntuació.
    """
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
