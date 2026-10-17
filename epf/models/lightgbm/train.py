"""Gradient boosting sobre la matriu per hora: un sol model per a totes les hores, i la
seva passada, de la matriu als seus fitxers.

LightGBM no fa sortida múltiple i TreeSHAP tampoc no l'admetria, així que la
família comparteix un model entre les 24 hores i rep l'hora com a columna. A
canvi, les interaccions de SHAP amb l'hora diuen com canvia un efecte al llarg
del dia.

Per a cada llavor i configuració: l'arbre ajustat amb les files d'entrenament i aturat
sobre les de validació, el preu de les setmanes de prova, la seva puntuació, l'atribució
de les files que s'expliquen i, a la millor configuració, les seves interaccions, la
comprovació que sumen la predicció que expliquen, i el fitxer.
"""

import time

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
from epf.parameters import BEST, BLACKOUT_WEEK, DEVELOPMENT, INTERACTION_ROWS, SEED

# La família, com la planifica una passada: el sostre d'arbres i la paciència, que no es
# cerquen, el punt de lloc mentre no hi ha cerca, sense els seus arbres, perquè els
# arbres els decideix l'aturada, i les deu millors de cada llavor de la seva pròpia
# cerca.
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


def lightgbm(point, trees, seed=SEED):
    """Un regressor de LightGBM sense ajustar, i repetible, d'un punt de la graella i
    amb `trees` arbres com a molt.

    La llavor fixa quines files pren cada arbre quan el punt en pren menys de totes.

    `deterministic` amb `force_row_wise` perquè dos ajustos de la mateixa finestra
    donin els mateixos arbres: sense això no els donen, i la predicció no ho
    destapa perquè segueix un sol camí per arbre mentre SHAP recorre tots els camins
    pesats per la cobertura. Mesurat el 2026-10-07 sobre la finestra de 1085 dies, tres
    ajustos: 1,21 s cada un i models diferents sense les dues banderes, 1,88 s i
    idèntics amb elles. No són números que es toquin per guanyar error, i per això no
    viuen al fitxer d'hiperparàmetres.
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
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )


def fit_rows(X, y, point, trees, validation=(), patience=PATIENCE):
    """Un model ajustat amb les files donades, d'un punt de la graella.

    Sense `validation` fa `trees` arbres, ni un més ni un menys. Amb `validation`,
    parelles `(X, y)`, `trees` és el sostre, i l'entrenament s'atura quan qualsevol
    d'elles porta `patience` arbres sense millorar, i el model es queda amb el millor
    arbre d'aquella: amb quatre parelles és l'aturada de Trebbien et al. (2023), que
    parteix la validació en quatre. Cada crida fa un regressor nou, de manera que cap
    ajust no hereta res de l'anterior.
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
    """El preu de cada fila."""
    return model.predict(X)


def fitted(X, y, groups, parts, explained, seed, numbers, best):
    """L'arbre d'una configuració, aturat quan qualsevol de les quatre parts de la
    validació deixa de millorar, l'atribució de les files explicades i, a la millor
    configuració de la llavor, `best`, les interaccions sobre una mostra d'aquelles
    files. `groups` és el grup de cada columna de `X`."""
    training, test, shown = (
        X.index.isin(days) for days in (parts.training, parts.test, explained)
    )
    weeks = validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK)
    validation = [(X[part], y[part]) for part in (X.index.isin(days) for days in weeks)]
    model = fit_rows(
        X[training], y[training], numbers, MOST_TREES, validation=validation
    )
    values, base = attribution.attribute_rows(model, X[shown])
    # La base més els valors torna la predicció.
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
        # Cada fila d'interaccions torna el valor de la seva columna.
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


def train(family, folder, configurations, explained_days):
    """La passada de la família: cada configuració de cada llavor ajustada, comprovada,
    puntuada i desada. Torna les parelles `[llavor, configuració]` que ha desat ara."""
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
            start = time.time()
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
            check(found, family, seed, rank)
            scores = scored(found.predicted, found.actual)
            save_model(path, parts, explained, found, scores, rank)
            done.append([seed, rank])
            print(
                f"{family} llavor {seed} configuració {rank}: R² {scores['r2']:.3f}, "
                f"MAE {scores['mae']:.2f} EUR/MWh, residu de l'explicació "
                f"{found.gap:.1e}, {time.time() - start:.1f} s",
                flush=True,
            )
    return done
