"""L'espai de la cerca de les xarxes distribucionals, el del codi de Marcjasz et al.
(2023), `nn.py`, sense els seus selectors de blocs d'entrada: cada xarxa llegeix totes
les columnes, com les altres famílies."""

# L'espai del codi de Marcjasz et al. (2023): dues capes amagades, les neurones de cada
# una, les sis activacions pels seus noms, el dropout de després de l'entrada, el
# coeficient de cada L1, tots en el mateix interval logarítmic, i la taxa d'aprenentatge,
# també logarítmica.
LAYERS = 2
NEURONS = (16, 1024)
ACTIVATIONS = ("sigmoid", "relu", "elu", "tanh", "softplus", "softmax")
DROPOUT = (0.0, 1.0)
L1 = (1e-5, 10.0)
LEARNING_RATE = (1e-5, 0.1)


def penalty(trial, name):
    """Una L1 com la demana el codi: primer si n'hi ha, i el coeficient només si n'hi ha.
    Sense, és zero, que és una L1 que no fa res."""
    if not trial.suggest_categorical(f"regularize_{name}", (False, True)):
        return 0.0
    return trial.suggest_float(f"{name}_l1", *L1, log=True)


def space(trial, distribution):
    """Els hiperparàmetres d'una prova, en la forma en què els llegeix `ddnn.build`.

    L'L1 de cada paràmetre es demana pel nom que té a `distribution`, i en el seu ordre:
    el codi de l'article posa el pes de la cua de Johnson SU abans de l'asimetria.
    """
    layers = range(1, LAYERS + 1)
    return {
        "hidden": [trial.suggest_int(f"neurons_{layer}", *NEURONS) for layer in layers],
        "activation": [trial.suggest_categorical(f"activation_{layer}", ACTIVATIONS)
                       for layer in layers],
        "dropout": (trial.suggest_float("dropout_rate", *DROPOUT)
                    if trial.suggest_categorical("dropout", (False, True)) else 0.0),
        "l1": [penalty(trial, f"h{layer}_kernel") for layer in layers],
        "activity_l1": [penalty(trial, f"h{layer}_activation") for layer in layers],
        "parameter_l1": [penalty(trial, name) for name in distribution.names],
        "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE, log=True),
    }
