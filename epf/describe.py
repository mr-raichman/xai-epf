"""La part B: el protocol de Trebbien et al. (2023) sobre el període de desenvolupament.

Les setmanes es barregen per llavor i es reparteixen en entrenament, validació i
prova, i la robustesa és la dispersió entre llavors. A diferència del seu article, el
preu del dia abans és una entrada, i per això l'entrenament perd el dia de cada costat
d'una setmana retinguda.

`epf describe` ajusta cada família explicada sobre cada conjunt i cada llavor, la puntua
sobre les setmanes de prova al costat del sòl de l'ordre de mèrit i n'explica les files
de prova. Tot es desa per llavor, de manera que les figures surten del que hi ha al disc
i no d'un ajust nou. Mentre la cerca no s'ha fet, cada família hi entra amb uns números
que no s'han cercat, i el registre de cada una ho diu.
"""
import collections
import json
import shutil
import time

import numpy as np
import pandas as pd

from epf import matrix, parameters
from epf.attribution.stored import TOLERANCE, TRANSFORMED_TOLERANCE
from epf.data import forecasts, price
from epf.models.boosting import boosting, boosting_hyperparameters
from epf.models.ddnn import ddnn, ddnn_hyperparameters
from epf.models.ddnn.distributions import FAMILIES as DISTRIBUTIONS
from epf.models.linear import linear, linear_hyperparameters
from epf.parameters import (DAYS_PER_WEEK, DESCRIPTIONS, DEVELOPMENT, GAP,
                            INTERACTION_ROWS, MEASUREMENT, SHARES, TEST)
from epf.results import run_files

# Els dies de cada part d'una partició, en l'ordre en què es reparteixen les setmanes.
Split = collections.namedtuple("Split", "training validation test")

# Parts en què es parteix la validació perquè l'entrenament s'aturi quan qualsevol
# d'elles deixa de millorar, les de Trebbien et al. (2023).
VALIDATION_PARTS = 4


def _deal(seed, period, shares):
    """Els dies del període i, per a cada part, les setmanes que li toquen, en l'ordre
    en què la llavor les ha repartit.

    El període es talla en setmanes de set dies des del primer, que es barregen amb la
    llavor i es reparteixen en l'ordre de `Split` segons els percentatges de `shares`.
    """
    days = pd.date_range(*period, freq="D")
    weeks, remainder = divmod(len(days), DAYS_PER_WEEK)
    if remainder:
        raise ValueError(f"{len(days)} dies no són setmanes senceres")
    if sum(shares) != 100 or any(weeks * share % 100 for share in shares):
        raise ValueError(f"{shares} no reparteix {weeks} setmanes senceres")
    order = np.random.default_rng(seed).permutation(weeks)
    ends = np.cumsum([weeks * share // 100 for share in shares])
    return days, [order[start:end] for start, end in zip([0, *ends[:-1]], ends)]


def _days_of(days, weeks):
    """Els dies de les setmanes donades, en ordre de data."""
    return days[np.sort(np.concatenate([np.arange(week * DAYS_PER_WEEK,
                                                  (week + 1) * DAYS_PER_WEEK)
                                        for week in weeks]))]


def split(seed, period, shares=SHARES, gap=GAP):
    """Els dies d'entrenament, de validació i de prova d'una llavor.

    Després de repartir les setmanes surt de l'entrenament tot dia que tingui un dia
    retingut a `gap` dies o menys, a qualsevol costat: aquest dia no és de cap part.
    """
    days, dealt = _deal(seed, period, shares)
    part_of_week = np.empty(len(days) // DAYS_PER_WEEK, dtype=int)
    for part, weeks in enumerate(dealt):
        part_of_week[weeks] = part
    part_of_day = np.repeat(part_of_week, DAYS_PER_WEEK)

    training = part_of_day == 0
    beside = np.zeros(len(days), dtype=bool)
    for offset in range(1, gap + 1):
        beside[offset:] |= ~training[:-offset]
        beside[:-offset] |= ~training[offset:]
    return Split(days[training & ~beside], days[part_of_day == 1], days[part_of_day == 2])


def validation_parts(seed, period, count=VALIDATION_PARTS, shares=SHARES):
    """Les setmanes de validació d'una llavor en `count` parts iguals, en l'ordre en
    què la llavor les ha repartit, de manera que cada part és de setmanes a l'atzar."""
    days, dealt = _deal(seed, period, shares)
    weeks = dealt[Split._fields.index("validation")]
    if len(weeks) % count:
        raise ValueError(f"{len(weeks)} setmanes de validació no fan {count} parts iguals")
    return [_days_of(days, chunk) for chunk in np.split(weeks, count)]


# ------------------------------------------------------------------------- el pas

# Nivell del camí de les execucions on viu la part B, al lloc on un camí endavant té el
# seu conjunt d'informació: els conjunts de la part B en són carpetes per sota.
STAGE = "describe"

# Els conjunts, que són carpetes del pas, i el sòl de l'ordre de mèrit, que hi és com
# una família més perquè es puntua igual. `forecasts` es diu com el conjunt de dades de
# les previsions i és una altra cosa.
FORECASTS, PREVIOUS_DAY, DATE_PROBE = "forecasts", "previous_day", "date_probe"
FLOOR = "floor"

# Grau del polinomi del sòl, el cúbic de Trebbien et al. (2023).
FLOOR_DEGREE = 3

RECORD = "record.json"
SEED_FILE = "seed_{seed}.npz"

# El que surt de descriure una família en una llavor: la predicció i el preu de les
# files de prova en el mateix ordre, el que es desa de l'explicació, i el pitjor residu
# de l'explicació contra la predicció que explica, amb el marge que li toca.
Described = collections.namedtuple("Described", "predicted actual arrays gap tolerance")


def _parts_of(X, parts, explained):
    """Les files de cada part i les que s'expliquen, com a màscares."""
    return [X.index.isin(days) for days in (*parts, explained)]


def _tree(X, y, parts, explained, seed):
    """L'arbre, aturat quan qualsevol de les quatre parts de la validació deixa de
    millorar, i les interaccions sobre una mostra de les files que s'expliquen."""
    train, _, test, shown = _parts_of(X, parts, explained)
    validation = [(X[part], y[part]) for part in
                  (X.index.isin(days) for days in validation_parts(seed, DEVELOPMENT))]
    model = boosting.fit_rows(X[train], y[train], validation=validation)
    values, base = boosting.attribute_rows(model, X[shown])
    count = int(shown.sum())
    sample = np.sort(np.random.default_rng(seed).choice(
        count, min(INTERACTION_ROWS, count), replace=False))
    interactions = boosting.interaction_rows(model, X[shown].iloc[sample])
    # Dues sumes: la base més els valors torna la predicció, i cada fila d'interaccions
    # torna el valor de la seva columna.
    gap = max(np.abs(base + values.sum(axis=1) - boosting.predict_rows(model, X[shown])).max(),
              np.abs(interactions.sum(axis=2) - values[sample]).max())
    return Described(boosting.predict_rows(model, X[test]), y[test].to_numpy(),
                     {"columns": np.array(X.columns), "values": values, "base": base,
                      "interaction_rows": sample, "interactions": interactions,
                      "trees": model.best_iteration_}, gap, TOLERANCE)


def _net(X, y, parts, explained, seed):
    """La xarxa elàstica, amb els números sense cercar del seu fitxer."""
    train, _, test, shown = _parts_of(X, parts, explained)
    model = linear.fit_rows(X[train], y[train], linear_hyperparameters.PENALTY,
                            linear_hyperparameters.MIXING)
    explanation = linear.attribute_rows(model, X[shown])
    gap = np.abs(explanation.base + explanation.values.sum(axis=1)
                 - explanation.transformed).max()
    return Described(linear.predict_rows(model, X[test]).to_numpy(), y[test].to_numpy(),
                     {"columns": np.array(explanation.values.columns),
                      "values": explanation.values.to_numpy(),
                      "base": explanation.base.to_numpy(),
                      "transformed": explanation.transformed.to_numpy()},
                     gap, TRANSFORMED_TOLERANCE)


def _network(distribution):
    """La xarxa distribucional d'una distribució, aturada sobre tota la validació, i
    explicada paràmetre a paràmetre contra els dies d'entrenament de la llavor."""
    def described(X, y, parts, explained, seed):
        train, validate, test, shown = _parts_of(X, parts, explained)
        fitted = ddnn.fit_rows(distribution, (X[train], y[train]),
                               (X[validate], y[validate]), ddnn_hyperparameters.searched())
        explanation = ddnn.attribute_rows(distribution, fitted, X[shown],
                                          background=X[train])
        # Dues sumes: la base més les contribucions torna cada paràmetre, i la base és la
        # mitjana de cada paràmetre sobre tots els dies d'entrenament, que és el fons
        # que s'ha triat i el que es perdria si se'n retallés una mostra.
        gap = max(np.abs(explanation.base + explanation.values.sum(axis=1)
                         - ddnn.parameters_rows(distribution, fitted, X[shown])).max(),
                  np.abs(explanation.base - ddnn.parameters_rows(
                      distribution, fitted, X[train]).mean(axis=0)).max())
        found = ddnn.parameters_rows(distribution, fitted, X[test])
        return Described(distribution.point(found).ravel(), y[test].to_numpy().ravel(),
                         {"names": np.array(distribution.names),
                          "columns": np.array(explanation.columns),
                          "values": explanation.values, "base": explanation.base,
                          "parameters": found}, gap, TOLERANCE)
    return described


def _floor(X, y, parts, explained, seed):
    """El sòl de l'ordre de mèrit de Trebbien et al. (2023): un polinomi cúbic de la
    càrrega residual a l'hora, la demanda menys l'eòlica i la fotovoltaica previstes."""
    train, _, test, _ = _parts_of(X, parts, explained)
    load = (X["demand"] - X["wind"] - X["solar"]).to_numpy()
    coefficients = np.polyfit(load[train], y[train].to_numpy(), FLOOR_DEGREE)
    return Described(np.polyval(coefficients, load[test]), y[test].to_numpy(),
                     {"coefficients": coefficients}, 0.0, TOLERANCE)


# Qui es descriu: amb quina taula, sobre quins conjunts, com, i amb quins números. El sòl
# només té sentit amb les previsions, i la prova de datació només l'ajusta l'arbre.
Describer = collections.namedtuple("Describer", "table sets describe numbers")


def describers():
    """Cada família que es descriu, per nom."""
    tree = boosting_hyperparameters
    net = linear_hyperparameters
    found = {
        boosting.NAME: Describer(
            "hourly", (FORECASTS, PREVIOUS_DAY, DATE_PROBE), _tree,
            {"trees": tree.TREES, "rate": tree.RATE, "leaves": tree.LEAVES,
             "patience": tree.PATIENCE, "unsearched": True}),
        linear.NAME: Describer(
            "hourly", (FORECASTS, PREVIOUS_DAY), _net,
            {"penalty": net.PENALTY, "mixing": net.MIXING, "max_iter": net.MAX_ITER,
             "unsearched": True}),
    }
    for name, distribution in DISTRIBUTIONS.items():
        found[name] = Describer(
            "daily", (FORECASTS, PREVIOUS_DAY), _network(distribution),
            ddnn_hyperparameters.searched() | {
                "batch": ddnn_hyperparameters.BATCH, "epochs": ddnn_hyperparameters.EPOCHS,
                "patience": ddnn_hyperparameters.PATIENCE, "unsearched": True})
    found[FLOOR] = Describer("hourly", (FORECASTS,), _floor, {"degree": FLOOR_DEGREE})
    return found


def _tables(frame, day_ahead):
    """Les taules de cada conjunt, per nom i per taula."""
    sets = {FORECASTS: matrix.forecasts(frame, day_ahead),
            PREVIOUS_DAY: matrix.previous_day(frame)}
    tables = {name: {"hourly": found.hourly, "daily": found.daily}
              for name, found in sets.items()}
    tables[DATE_PROBE] = {"hourly": matrix.date_probe(frame, day_ahead, DEVELOPMENT[0])}
    return tables


def scored(predicted, actual):
    """La R-quadrat i el MAE sobre totes les files de prova juntes."""
    error = np.asarray(predicted) - np.asarray(actual)
    spread = np.asarray(actual) - np.mean(actual)
    return {"r2": float(1 - np.sum(error ** 2) / np.sum(spread ** 2)),
            "mae": float(np.mean(np.abs(error)))}


def _days(index):
    return index.to_numpy().astype("datetime64[D]")


def _save(path, parts, explained, described, scores):
    """Desa una llavor, primer a un fitxer a part i després al seu nom, de manera que un
    fitxer amb nom és sempre una llavor sencera."""
    arrays = {f"days_{part}": _days(days) for part, days in parts._asdict().items()}
    arrays |= {"days_explained": _days(explained), "predicted": described.predicted,
               "actual": described.actual, "gap": described.gap, **scores}
    for name, value in described.arrays.items():
        value = np.asarray(value)
        arrays[name] = value.astype(np.float32) if value.dtype.kind == "f" and value.ndim > 1 else value
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with open(partial, "wb") as handle:
        np.savez(handle, **arrays)
    partial.replace(path)


def main(arguments):
    commit = run_files.git("rev-parse", "HEAD")
    if arguments.tier == MEASUREMENT and run_files.git("status", "--porcelain"):
        print("l'arbre té canvis sense desar. El directori d'una mesura es diu pel "
              "commit, i amb canvis sense desar el nom no diria quin codi l'ha fet. "
              f"Fes un commit, o fes servir --tier {TEST}.")
        return 1
    table = describers()
    chosen = arguments.family or list(table)
    unknown = sorted(set(chosen) - set(table))
    if unknown:
        print(f"no es descriu cap família {unknown}; n'hi ha {sorted(table)}")
        return 1

    seeds, explained_days = DESCRIPTIONS[arguments.tier]
    where = run_files.run_folder(arguments.tier, STAGE, commit[:7])
    # Reprendre és del nivell de mesura i prou, com al camí endavant: el de prova admet
    # l'arbre brut, i dues còpies diferents del codi en compartirien el directori.
    if arguments.tier == TEST or arguments.replace:
        for family in chosen:
            for name in table[family].sets:
                shutil.rmtree(where / name / family, ignore_errors=True)

    tables = _tables(price.load(), forecasts.load())
    swept = {name: getattr(parameters, name) for name in dir(parameters)
             if name.isupper() and not name.startswith("_")}
    for family in chosen:
        describer = table[family]
        for name in describer.sets:
            folder = where / name / family
            X, y = tables[name][describer.table]
            started, described_now = time.time(), []
            for seed in seeds:
                path = folder / SEED_FILE.format(seed=seed)
                if path.exists():
                    continue
                parts = split(seed, DEVELOPMENT)
                explained = parts.test if explained_days is None else parts.test[:explained_days]
                start = time.time()
                described = describer.describe(X, y, parts, explained, seed)
                if described.gap > described.tolerance:
                    raise ValueError(
                        f"l'explicació de {family} sobre {name}, llavor {seed}, no torna la "
                        f"seva predicció: se n'allunya {described.gap:.3g}, per sobre del "
                        f"marge de {described.tolerance}")
                scores = scored(described.predicted, described.actual)
                _save(path, parts, explained, described, scores)
                described_now.append(seed)
                print(f"{name} {family} llavor {seed}: R² {scores['r2']:.3f}, MAE "
                      f"{scores['mae']:.2f} EUR/MWh, residu de l'explicació "
                      f"{described.gap:.1e}, {time.time() - start:.1f} s", flush=True)
            record = {"commit": commit, "tier": arguments.tier, "set": name,
                      "family": family, "seeds": list(seeds),
                      "explained_days": explained_days, "numbers": describer.numbers,
                      "described_now": described_now,
                      "seconds_now": round(time.time() - started, 1),
                      "parameters": swept}
            (folder / RECORD).write_text(json.dumps(run_files.plain(record), indent=2))
    print(f"la part B és a {where}")
    return 0
