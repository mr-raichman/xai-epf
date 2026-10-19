"""Els números de les xarxes."""

# Mida del batch, sostre d'èpoques i paciència de l'early stopping: els del codi de
# Marcjasz et al. (2023).
BATCH = 32
EPOCHS = 1500
PATIENCE = 50

# Els números de cada família mentre no s'ha cercat: de les quatre cerques de
# `COPIED_TRIALS` trials que l'article va publicar per al mercat alemany, per a la xarxa
# de punt i per a cada distribució, la de millor puntuació pròpia, el fitxer
# `COPIED_FROM` del seu repositori. Sense els blocs d'entrada que aquella cerca va
# triar: la xarxa llegeix totes les columnes. L'L1 dels paràmetres va en l'ordre de
# `distributions`, i no en el del codi de l'article, que posa el pes de la cua de
# Johnson SU abans de l'asimetria; la xarxa de punt no en té cap a la sortida, i el seu
# zero és una L1 que no fa res.
COPIED = {
    "dnn": {
        "hidden": [979, 767],
        "activation": ["softplus", "relu"],
        "dropout": 0.0,
        "l1": [0.0, 0.0],
        "activity_l1": [0.0, 0.0],
        "parameter_l1": [0.0],
        "learning_rate": 0.0005252223827375965,
    },
    "ddnn_normal": {
        "hidden": [110, 823],
        "activation": ["softplus", "relu"],
        "dropout": 0.0,
        "l1": [0.0, 0.0],
        "activity_l1": [1.0076687725977794e-05, 0.0],
        "parameter_l1": [0.0, 0.0],
        "learning_rate": 0.00032041091802416773,
    },
    "ddnn_jsu": {
        "hidden": [243, 895],
        "activation": ["softplus", "elu"],
        "dropout": 0.0,
        "l1": [0.0, 0.0],
        "activity_l1": [0.0, 0.0],
        "parameter_l1": [0.0, 0.0, 0.0, 0.0],
        "learning_rate": 0.000281493495779889,
    },
}
COPIED_FROM = {
    "dnn": "FINAL_DE_selection_prob_point3",
    "ddnn_normal": "FINAL_DE_selection_prob_normal4",
    "ddnn_jsu": "FINAL_DE_selection_prob_jsu3",
}
COPIED_TRIALS = 2048

# Els números que cap cerca no toca, els mateixos per a les tres xarxes.
FIXED = {"batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE}


def copied(name):
    """Torna els números d'una família mentre no s'ha cercat, i d'on es van copiar."""
    return COPIED[name] | {
        "copied_from": COPIED_FROM[name],
        "copied_trials": COPIED_TRIALS,
    }
