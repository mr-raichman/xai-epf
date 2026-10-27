"""La cerca de la xarxa neuronal, sobre l'espai del codi de Marcjasz et al. (2023)."""

from epf.models.dnn import hyperparameters, network

# La matriu de la cerca, la diària, com la de la passada.
MATRIX = "daily"

# Sense graella: l'espai es mostreja a l'atzar.
GRID = None

# L'espai de `Python/nn.py`, el fitxer de la cerca del codi de Marcjasz et al. (2023),
# sense els seus selectors de blocs d'entrada, perquè la xarxa llegeix totes les
# columnes, ni `softmax`, a través de la qual l'explicació de DeepExplainer no torna la
# predicció.
LAYERS = 2
NEURONS = (16, 1024)
ACTIVATIONS = ("sigmoid", "relu", "elu", "tanh", "softplus")
DROPOUT = (0.0, 1.0)
L1 = (1e-5, 10.0)
LEARNING_RATE = (1e-5, 0.1)

# Com es cerca, que el study desa i compara amb el d'ara: l'espai, i els números
# fixos de la xarxa.
FIXED = hyperparameters.FIXED
SPACE = {
    "layers": LAYERS,
    "neurons": NEURONS,
    "activations": ACTIVATIONS,
    "dropout": DROPOUT,
    "l1": L1,
    "learning_rate": LEARNING_RATE,
}


def penalty(optuna_trial, name):
    """Torna el coeficient d'una L1 del trial, o zero si no en vol cap.

    Com al codi de l'article, el trial tria primer si n'hi ha, i el coeficient només si
    n'hi ha.
    """
    if not optuna_trial.suggest_categorical(f"regularize_{name}", (False, True)):
        return 0.0
    return optuna_trial.suggest_float(f"{name}_l1", *L1, log=True)


def drawn(family, optuna_trial):
    """Torna els números d'un trial, en la forma en què els llegeix `network.build`.

    Com a la xarxa de punt de Marcjasz et al. (2023), la sortida no té cap L1 i no se'n
    cerca cap.
    """
    layers = range(1, LAYERS + 1)
    # El sampler els treu tots d'un sol flux, i en un altre ordre en treuria uns altres.
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
        "learning_rate": optuna_trial.suggest_float(
            "learning_rate", *LEARNING_RATE, log=True
        ),
    }


def trial(family, drawn, train, validate, seed, deleted):
    """Ajusta els números d'un trial, `drawn`, i els puntua per la pèrdua de validació.

    S'ajusta amb les files d'entrenament, `train`, amb early stopping sobre tota la
    validació, `validate`. La pèrdua és la de Keras, amb les L1 a dins, com la que
    puntua el codi de l'article. Torna els números i la puntuació. `deleted` no hi fa
    res, perquè la validació no es parteix.
    """
    X_validate, y_validate = validate
    fitted = network.fit_rows(train, validate, drawn, seed)
    return drawn, float(
        fitted.evaluate(
            network.inputs(X_validate).to_numpy(),
            y_validate.to_numpy(dtype=float),
            verbose=0,
        )
    )
