"""La cerca de l'arbre, sobre la graella del grup de Trebbien et al. (2023).

L'article no publica la seva cerca, i el seu codi, que la porta, no és enlloc. La
graella és la del codi de dos dels seus autors per a un altre problema, amb LightGBM.
"""

from epf.contract.metrics import mae
from epf.contract.split import validation_parts
from epf.models.lightgbm.hyperparameters import MOST_TREES, NAME
from epf.models.lightgbm.train import FAMILIES, fit_rows, predict_rows
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# La matriu de la cerca, la per hora, com la de la passada.
MATRIX = "hourly"

# La graella del codi de Kruse, Schäfer i Witthaut (2022) per a l'activació de la
# reserva secundària, `scripts/6_model_fit.py` de `johkruse/XAI-for-aFRR-activation`.
GRID = {
    "num_leaves": (10, 30, 200, 900),
    "max_depth": (10, 5),
    "subsample": (1.0, 0.9, 0.5),
    "learning_rate": (0.001, 0.05, 0.01, 0.1),
    "min_child_samples": (50, 100, 500),
}

# Com es cerca, que el study desa i compara amb el d'ara: la graella, i els números que
# cap trial no cerca, els que la passada desa al seu registre.
SPACE = GRID
FIXED = FAMILIES[NAME]["fixed"]


def drawn(family, optuna_trial):
    """Torna el punt de la graella que proposa un trial."""
    return {
        name: optuna_trial.suggest_categorical(name, values)
        for name, values in GRID.items()
    }


def trial(family, drawn, train, validate, seed):
    """Ajusta el punt d'un trial, `drawn`, i el puntua pel MAE de tota la validació.

    S'ajusta amb les files d'entrenament, `train`, i l'early stopping l'atura quan
    qualsevol de les quatre parts de la validació, `validate`, deixa de millorar, com a
    la passada. Torna el punt i la puntuació.
    """
    (X, y), (X_validate, y_validate) = train, validate
    weeks = validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK)
    parts = [X_validate.index.isin(days) for days in weeks]
    model = fit_rows(
        X,
        y,
        drawn,
        MOST_TREES,
        validation=[(X_validate[part], y_validate[part]) for part in parts],
    )
    return drawn, mae(predict_rows(model, X_validate), y_validate)
