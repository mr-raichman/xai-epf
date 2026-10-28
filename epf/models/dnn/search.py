"""La cerca de la xarxa neuronal, sobre l'espai del codi de Marcjasz et al. (2023)."""

from epf.models.dnn import hyperparameters, network

# La matriu de la cerca, la diària, com la de la passada.
MATRIX = "daily"

# Sense graella: l'espai el mostreja TPE.
GRID = None

# Trials de la cerca a cada seed, 100 pel còmput, on Marcjasz et al. (2023) en fan 2048,
# demanats en onades de 5, la meitat dels 10 que TPE tria a l'atzar abans de fer servir
# el seu model, perquè aquest s'actualitzi sovint. Una onada es demana sencera quan
# l'anterior ha tornat sencera, i TPE tria a partir dels trials tornats, de manera que
# una cerca fa els mateixos trials siguin quants siguin els processos.
TRIALS = 100
TRIALS_PER_WAVE = 5

# L'espai de `Python/nn.py`, el fitxer de la cerca del codi de Marcjasz et al. (2023),
# sense els seus selectors de blocs d'entrada, perquè la xarxa llegeix totes les
# columnes, ni el dropout ni les quatre L1, apagats al millor trial de cada un dels seus
# estudis de punt, i les L1 a més dins de la pèrdua que atura i puntua un ajust.
LAYERS = 2
# Des de 64 neurones: no en té menys cap capa del 5% millor dels trials d'aquells
# estudis, i Lago et al. (2021) treuen una capa de menys de 50.
NEURONS = (64, 1024)
# Sense sigmoid ni tanh, mai en aquell 5% millor, ni softmax, a través de la qual
# l'explicació de DeepExplainer no torna la predicció.
ACTIVATIONS = ("relu", "elu", "softplus")
# Fins a 1e-2, una dècada per sobre d'aquell 5% millor, com 1e-5 n'és per sota.
LEARNING_RATE = (1e-5, 1e-2)
# El batch en dies, que aquell codi fixa a 32, cercat: cada seed tria el seu.
BATCHES = (32, 64, 128, 256)

# Com es cerca, que el study desa i compara amb el d'ara: l'espai, el sampler i les
# onades, que decideixen quines configuracions té un study, i els números fixos de la
# xarxa.
FIXED = hyperparameters.FIXED
SPACE = {
    "layers": LAYERS,
    "neurons": NEURONS,
    "activations": ACTIVATIONS,
    "learning_rate": LEARNING_RATE,
    "batches": BATCHES,
    "sampler": "TPE",
    "trials_per_wave": TRIALS_PER_WAVE,
}


def drawn(family, optuna_trial):
    """Torna els números d'un trial, com els llegeixen `network.build` i `fit_rows`."""
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
        "learning_rate": optuna_trial.suggest_float(
            "learning_rate", *LEARNING_RATE, log=True
        ),
        "batch": optuna_trial.suggest_categorical("batch", BATCHES),
    }


def trial(family, drawn, train, validate, seed):
    """Ajusta els números d'un trial, `drawn`, i els puntua per la pèrdua de validació.

    S'ajusta amb les files d'entrenament, `train`, amb early stopping sobre tota la
    validació, `validate`. La pèrdua és la de validació de Keras, com la que puntua el
    codi de l'article, i sense cap càstig és el MAE. Torna els números, la puntuació i
    l'època amb què es queda.
    """
    X_validate, y_validate = validate
    fitted, epochs = network.fit_rows(train, validate, drawn, seed)
    score = fitted.evaluate(
        network.inputs(X_validate).to_numpy(),
        y_validate.to_numpy(dtype=float),
        verbose=0,
    )
    return drawn, float(score), {"epochs": epochs}
