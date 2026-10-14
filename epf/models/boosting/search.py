"""L'espai de la cerca de LightGBM: la graella del grup de Trebbien et al. (2023).

L'article no publica la seva cerca, i el seu codi, que la porta, no és enlloc. La graella
és la del codi de dos dels seus autors per a un altre problema, amb LightGBM.
"""

# La graella del codi de Kruse, Schäfer i Witthaut per a l'activació de la reserva
# secundària, `scripts/6_model_fit.py` de `johkruse/XAI-for-aFRR-activation`: 288 punts.
GRID = {
    "num_leaves": (10, 30, 200, 900),
    "max_depth": (10, 5),
    "subsample": (1.0, 0.9, 0.5),
    "learning_rate": (0.001, 0.05, 0.01, 0.1),
    "min_child_samples": (50, 100, 500),
}


def space(trial):
    """El punt de la graella d'una prova."""
    return {name: trial.suggest_categorical(name, values) for name, values in GRID.items()}
