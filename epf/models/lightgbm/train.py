"""L'arbre, LightGBM sobre la matriu per hora, i la seva passada.

Un sol model per a les 24 hores, que rep l'hora com a columna.
"""

import numpy as np
from lightgbm import LGBMRegressor, early_stopping

from epf import paths
from epf.contract import matrix
from epf.contract.explanation import PRICE_UNITS, TOLERANCE, check, group
from epf.contract.metrics import scored
from epf.contract.record import Fitted, save_model
from epf.contract.split import split, validation_parts
from epf.models.lightgbm import attribution
from epf.models.lightgbm.hyperparameters import (
    MIN_CHILD_WEIGHT,
    MOST_TREES,
    NAME,
    PATIENCE,
    SUBSAMPLE_FREQ,
    UNSEARCHED,
)
from epf.parameters import (
    BEST,
    BLACKOUT_WEEK,
    DEVELOPMENT,
    INTERACTION_ROWS,
    PROJECT_SEED,
    THREADS,
)

# La família, com la planifica una passada: el sostre d'arbres i la paciència, que no es
# cerquen, el punt de lloc mentre no hi ha cerca, sense els seus arbres, perquè els
# arbres els decideix l'early stopping, i les `BEST` millors de cada seed de la seva
# pròpia cerca.
FAMILIES = {
    NAME: {
        "fixed": {"most_trees": MOST_TREES, "patience": PATIENCE},
        "placeholder": {
            name: value for name, value in UNSEARCHED.items() if name != "trees"
        },
        "kept": BEST,
        "searched_as": NAME,
    }
}


def lightgbm(point, trees, project_seed=PROJECT_SEED):
    """Torna un regressor de LightGBM sense ajustar d'un punt de la graella, `point`.

    Fa `trees` arbres com a molt, i `project_seed` fixa quines files pren cada arbre
    quan el punt en pren menys de totes.
    """
    return LGBMRegressor(
        n_estimators=trees,
        learning_rate=point["learning_rate"],
        num_leaves=point["num_leaves"],
        max_depth=point["max_depth"],
        subsample=point["subsample"],
        subsample_freq=SUBSAMPLE_FREQ,
        min_child_samples=point["min_child_samples"],
        min_child_weight=MIN_CHILD_WEIGHT,
        random_state=project_seed,
        n_jobs=THREADS,
        verbose=-1,
        # Sense totes dues, dos ajustos de la mateixa finestra no donen els mateixos
        # arbres, i la predicció no ho destapa perquè segueix un sol camí per arbre
        # mentre SHAP recorre tots els camins pesats per la cobertura.
        deterministic=True,
        force_row_wise=True,
    )


def fit_rows(X, y, point, trees, validation=(), patience=PATIENCE):
    """Ajusta i torna un model d'un punt de la graella amb les files donades.

    Sense `validation` fa `trees` arbres, ni un més ni un menys. Amb `validation`,
    parelles `(X, y)`, `trees` és el sostre, i l'early stopping atura l'entrenament quan
    qualsevol d'elles porta `patience` arbres sense millorar, i el model es queda amb el
    millor arbre d'aquella: amb quatre parelles és l'early stopping de Trebbien et al.
    (2023), que parteix la validació en quatre. Cada crida fa un regressor nou, de
    manera que cap ajust no hereta res de l'anterior.
    """
    model = lightgbm(point, trees)
    if not validation:
        return model.fit(X, y)
    parts, targets = zip(*validation, strict=True)
    return model.fit(
        X,
        y,
        eval_X=parts,
        eval_y=targets,
        callbacks=[early_stopping(patience, verbose=False)],
    )


def predict_rows(model, X):
    return model.predict(X)


def fitted(X, y, groups, parts, explained, seed, numbers, best):
    """Ajusta l'arbre d'una configuració i n'explica les files dels dies explicats.

    L'early stopping l'atura quan qualsevol de les quatre parts de la validació deixa de
    millorar. A la millor configuració de la seed, `best`, explica també les
    interaccions d'una mostra de `INTERACTION_ROWS` d'aquelles files. `groups` és el
    grup de cada columna de `X`, i `numbers` el punt de la configuració.
    """
    training, test, shown = (
        X.index.isin(days) for days in (parts.training, parts.test, explained)
    )
    weeks = validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK)
    validation = [(X[part], y[part]) for part in (X.index.isin(days) for days in weeks)]
    model = fit_rows(
        X[training], y[training], numbers, MOST_TREES, validation=validation
    )
    values, base = attribution.attribute_rows(model, X[shown])
    # La base més les contribucions torna la predicció.
    gap = np.abs(base + values.sum(axis=1) - predict_rows(model, X[shown])).max()
    arrays = {
        "columns": np.array(X.columns, dtype=str),
        "groups": groups,
        "inputs": X[shown].to_numpy(),
        "units": np.array([PRICE_UNITS]),
        "values": values,
        "base": base,
        "trees": model.best_iteration_,
    }
    if best:
        count = int(shown.sum())
        sample = np.sort(
            np.random.default_rng(seed).choice(
                count, min(INTERACTION_ROWS, count), replace=False
            )
        )
        interactions = attribution.interaction_rows(model, X[shown].iloc[sample])
        # Cada fila d'interaccions torna la contribució de la seva columna.
        gap = max(gap, np.abs(interactions.sum(axis=2) - values[sample]).max())
        arrays |= {"interaction_rows": sample, "interactions": interactions}
    return Fitted(
        predict_rows(model, X[test]),
        y[test].to_numpy(),
        arrays,
        gap,
        TOLERANCE,
        stored=lambda found: attribution.sums(found, best),
    )


def train(family, folder, configurations, explained_days, saved=lambda: None):
    """Ajusta, comprova, puntua i desa cada configuració del pla que encara no hi és.

    `configurations` és el pla de la família per seed, i `explained_days` els dies de
    prova que s'expliquen de cada seed, tots amb `None`. Torna les parelles `[seed,
    configuració]` que ha desat ara, i crida `saved` en desar cada una.
    """
    X, y = matrix.matrices(matrix.series()).hourly
    groups = np.array([group(column) for column in X.columns])
    done = []
    for seed, planned in configurations.items():
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        explained = (
            parts.test if explained_days is None else parts.test[:explained_days]
        )
        for configuration in planned:
            rank = configuration["rank"]
            path = folder / paths.SEED.format(seed=seed, rank=rank)
            if path.exists():
                continue
            found = fitted(
                X,
                y,
                groups,
                parts,
                explained,
                seed,
                configuration["numbers"],
                best=rank == 0,
            )
            check(found, rank)
            scores = scored(found.predicted, found.actual)
            save_model(path, parts, explained, found, scores, rank)
            done.append([seed, rank])
            saved()
    return done
