"""El protocol de Trebbien et al. (2023) sobre el període que es descriu.

Les setmanes es barregen per llavor i es reparteixen en entrenament, validació i
prova, i la robustesa és la dispersió entre llavors. A diferència del seu article, el
preu del dia abans és una entrada, i per això l'entrenament perd el dia de cada costat
d'una setmana retinguda.

`epf describe` ajusta cada família explicada sobre cada conjunt i cada llavor, la puntua
sobre les setmanes de prova al costat del sòl de l'ordre de mèrit, de les prediccions
ingènues i de la xarxa de punt, i n'explica les files de prova. Tot es desa per llavor, de manera que les figures surten del que hi ha al disc
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
from epf.data import forecasts, price, schedule
from epf.models.boosting import boosting, boosting_hyperparameters
from epf.models.ddnn import ddnn, ddnn_hyperparameters
from epf.models.ddnn.distributions import FAMILIES as DISTRIBUTIONS
from epf.models.ddnn.distributions import POINT, POINT_FAMILY
from epf.models.linear import linear, linear_hyperparameters
from epf.models.naive import naive
from epf.parameters import (BLACKOUT_WEEK, DAYS_PER_WEEK, DESCRIPTIONS, DEVELOPMENT, GAP,
                            INTERACTION_ROWS, MEASUREMENT, SHARES, SMOKE)
from epf.results import describe_figures, run_files

# Els dies de cada part d'una partició, en l'ordre en què es reparteixen les setmanes.
Split = collections.namedtuple("Split", "training validation test")

# Parts en què es parteix la validació perquè l'entrenament s'aturi quan qualsevol
# d'elles deixa de millorar, les de Trebbien et al. (2023).
VALIDATION_PARTS = 4


def _deal(seed, period, deleted, shares):
    """Els dies del període i, per a cada part, les setmanes que li toquen, en l'ordre
    en què la llavor les ha repartit.

    El període es talla en setmanes de set dies des del primer, i la setmana `deleted`
    en surt i no és de cap part. La resta es barregen amb la llavor i es reparteixen en
    l'ordre de `Split`: a la validació, la seva part de `shares` arrodonida a un múltiple
    de `VALIDATION_PARTS`, perquè es parteixi en parts iguals; a la prova, la seva
    arrodonida; i a l'entrenament, la resta.
    """
    days = pd.date_range(*period, freq="D")
    weeks, remainder = divmod(len(days), DAYS_PER_WEEK)
    if remainder:
        raise ValueError(f"{len(days)} dies no són setmanes senceres")
    first = days.get_indexer([pd.Timestamp(deleted[0])])[0]
    if first < 0 or first % DAYS_PER_WEEK or (
            pd.Timestamp(deleted[1]) - pd.Timestamp(deleted[0])).days != DAYS_PER_WEEK - 1:
        raise ValueError(f"{deleted} no és cap de les setmanes del període")
    kept = np.delete(np.arange(weeks), first // DAYS_PER_WEEK)
    order = np.random.default_rng(seed).permutation(kept)
    _, validation_share, test_share = shares
    validation = VALIDATION_PARTS * round(len(kept) * validation_share / 100
                                          / VALIDATION_PARTS)
    test = round(len(kept) * test_share / 100)
    ends = np.cumsum([len(kept) - validation - test, validation, test])
    return days, [order[start:end] for start, end in zip([0, *ends[:-1]], ends)]


def _days_of(days, weeks):
    """Els dies de les setmanes donades, en ordre de data."""
    return days[np.sort(np.concatenate([np.arange(week * DAYS_PER_WEEK,
                                                  (week + 1) * DAYS_PER_WEEK)
                                        for week in weeks]))]


def split(seed, period, deleted, shares=SHARES, gap=GAP):
    """Els dies d'entrenament, de validació i de prova d'una llavor.

    Després de repartir les setmanes surt de l'entrenament tot dia que tingui un dia
    retingut a `gap` dies o menys, a qualsevol costat: aquest dia no és de cap part. La
    setmana suprimida no és retinguda, i els dies del seu costat es reparteixen com
    qualsevol altre.
    """
    days, dealt = _deal(seed, period, deleted, shares)
    part_of_week = np.full(len(days) // DAYS_PER_WEEK, -1)
    for part, weeks in enumerate(dealt):
        part_of_week[weeks] = part
    part_of_day = np.repeat(part_of_week, DAYS_PER_WEEK)

    training = part_of_day == 0
    held_out = (part_of_day == 1) | (part_of_day == 2)
    beside = np.zeros(len(days), dtype=bool)
    for offset in range(1, gap + 1):
        beside[offset:] |= held_out[:-offset]
        beside[:-offset] |= held_out[offset:]
    return Split(days[training & ~beside], days[part_of_day == 1], days[part_of_day == 2])


def validation_parts(seed, period, deleted, count=VALIDATION_PARTS, shares=SHARES):
    """Les setmanes de validació d'una llavor en `count` parts iguals, en l'ordre en
    què la llavor les ha repartit, de manera que cada part és de setmanes a l'atzar."""
    days, dealt = _deal(seed, period, deleted, shares)
    weeks = dealt[Split._fields.index("validation")]
    if len(weeks) % count:
        raise ValueError(f"{len(weeks)} setmanes de validació no fan {count} parts iguals")
    return [_days_of(days, chunk) for chunk in np.split(weeks, count)]


# -------------------------------------------------------------------- la passada

# La carpeta de les passades dins de cada nivell, `runs/<nivell>/describe/<commit>/`,
# amb els conjunts en carpetes per sota.
STAGE = "describe"

# Els conjunts, que són carpetes de la passada, i el sòl de l'ordre de mèrit, que hi és
# com una família més perquè es puntua igual. `schedule` es diu com el conjunt de dades
# del programa, que és el que hi afegeix, i és una altra cosa.
SCHEDULE, DATE_PROBE = "schedule", "date_probe"
FLOOR = "floor"

# Grau del polinomi del sòl, el cúbic de Trebbien et al. (2023).
FLOOR_DEGREE = 3

# Les unitats d'una contribució: un preu, un preu a l'espai de la transformació de Lago
# et al. (2021), on la base més les contribucions no són un preu, o cap, que són
# l'asimetria i el pes de la cua d'una distribució.
PRICE_UNITS = "EUR/MWh"
TRANSFORMED_UNITS = "espai transformat"
UNITLESS = "sense unitats"

# Marge de la comprovació d'additivitat, en EUR/MWh: molt per sota de qualsevol
# diferència que vulgui dir res en un preu. Es comprova sobre el que hi ha a la memòria,
# abans de desar, i el residu de cada llavor s'imprimeix al costat de les seves
# puntuacions.
TOLERANCE = 1e-3

# Marge de l'additivitat a l'espai transformat, a part perquè és en les seves unitats i
# no en un preu: en EUR/MWh val tant com el de dalt, perquè a prop d'1 la derivada de
# `sinh` és d'1,5 i la MAD de les files d'ajust va en desenes d'euros.
TRANSFORMED_TOLERANCE = 1e-4

# El que surt de descriure una família en una llavor: la predicció i el preu de les
# files de prova en el mateix ordre, el que es desa de l'explicació, i el pitjor residu
# de l'explicació contra la predicció que explica, amb el marge que li toca.
Described = collections.namedtuple("Described", "predicted actual arrays gap tolerance")


def group(column):
    """El grup d'una columna a la importància per grups: el dia abans, cada previsió, cada
    grup del programa, el calendari i, a la prova de datació, el comptador de dies."""
    if column.startswith("lag1_"):
        return "previous_day"
    for name in (*forecasts.INDICATORS, *schedule.GROUPS):
        if column == name or column.startswith(f"{name}_"):
            return name
    return "date" if column == matrix.DAY_COUNTER else "calendar"


def _parts_of(X, parts, explained):
    """Les files de cada part i les que s'expliquen, com a màscares."""
    return [X.index.isin(days) for days in (*parts, explained)]


def _tree(X, y, parts, explained, seed):
    """L'arbre, aturat quan qualsevol de les quatre parts de la validació deixa de
    millorar, i les interaccions sobre una mostra de les files que s'expliquen."""
    train, _, test, shown = _parts_of(X, parts, explained)
    weeks = validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK)
    validation = [(X[part], y[part]) for part in (X.index.isin(days) for days in weeks)]
    model = boosting.fit_rows(X[train], y[train], boosting_hyperparameters.UNSEARCHED,
                              boosting_hyperparameters.MOST_TREES, validation=validation)
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
                     {"columns": np.array(X.columns, dtype=str),
                      "groups": np.array([group(column) for column in X.columns]),
                      "inputs": X[shown].to_numpy(), "units": np.array([PRICE_UNITS]),
                      "values": values, "base": base, "interaction_rows": sample,
                      "interactions": interactions, "trees": model.best_iteration_},
                     gap, TOLERANCE)


def _net(X, y, parts, explained, seed):
    """La xarxa elàstica, amb els números sense cercar del seu fitxer."""
    train, _, test, shown = _parts_of(X, parts, explained)
    model = linear.fit_rows(X[train], y[train], linear_hyperparameters.PENALTY,
                            linear_hyperparameters.MIXING)
    explanation = linear.attribute_rows(model, X[shown])
    gap = np.abs(explanation.base + explanation.values.sum(axis=1)
                 - explanation.transformed).max()
    return Described(linear.predict_rows(model, X[test]).to_numpy(), y[test].to_numpy(),
                     {"columns": np.array(explanation.values.columns, dtype=str),
                      "groups": np.array([group(column)
                                          for column in explanation.values.columns]),
                      "inputs": X.loc[shown, explanation.values.columns].to_numpy(),
                      "units": np.array([TRANSFORMED_UNITS]),
                      "values": explanation.values.to_numpy(),
                      "base": explanation.base.to_numpy(),
                      "transformed": explanation.transformed.to_numpy()},
                     gap, TRANSFORMED_TOLERANCE)


def _network(name, distribution):
    """La xarxa distribucional d'una distribució, aturada sobre tota la validació, i
    explicada paràmetre a paràmetre contra els dies d'entrenament de la llavor.

    Una xarxa per distribució i llavor, on Marcjasz et al. (2023) en mitjanen quatre:
    cada membre seria una cerca més, i el que costa és la cerca.
    """
    def described(X, y, parts, explained, seed):
        train, validate, test, shown = _parts_of(X, parts, explained)
        fitted = ddnn.fit_rows(distribution, (X[train], y[train]),
                               (X[validate], y[validate]),
                               ddnn_hyperparameters.COPIED[name])
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
                          "groups": np.array([group(column)
                                              for column in explanation.columns]),
                          "inputs": explanation.inputs,
                          # La localització i l'escala en EUR/MWh, la resta sense unitats.
                          "units": np.array([PRICE_UNITS, PRICE_UNITS, UNITLESS,
                                             UNITLESS][:len(distribution.names)]),
                          "values": explanation.values, "base": explanation.base,
                          "parameters": found}, gap, TOLERANCE)
    return described


def _point_network(X, y, parts, explained, seed):
    """La xarxa de punt de Marcjasz et al. (2023), la distribucional amb la sortida
    canviada, ajustada sobre les mateixes files i aturada igual."""
    train, validate, test, _ = _parts_of(X, parts, explained)
    fitted = ddnn.fit_rows(POINT, (X[train], y[train]), (X[validate], y[validate]),
                           ddnn_hyperparameters.COPIED[POINT_FAMILY])
    return Described(ddnn.predict_rows(POINT, fitted, X[test]).ravel(),
                     y[test].to_numpy().ravel(), {}, 0.0, TOLERANCE)


def _floor(X, y, parts, explained, seed):
    """El sòl de l'ordre de mèrit de Trebbien et al. (2023): un polinomi cúbic de la
    càrrega residual a l'hora, la demanda menys l'eòlica i la fotovoltaica previstes."""
    train, _, test, _ = _parts_of(X, parts, explained)
    load = (X["demand"] - X["wind"] - X["solar"]).to_numpy()
    coefficients = np.polyfit(load[train], y[train].to_numpy(), FLOOR_DEGREE)
    return Described(np.polyval(coefficients, load[test]), y[test].to_numpy(),
                     {"coefficients": coefficients, "load_training": load[train],
                      "actual_training": y[train].to_numpy()}, 0.0, TOLERANCE)


def _naive(forecast):
    """Una predicció ingènua de Lago et al. (2021), presa de la sèrie del preu als dies de
    prova que té la taula del conjunt, de manera que es puntua sobre les files de totes."""
    def described(price, y, parts, explained, seed):
        days = y.index[y.index.isin(parts.test)]
        return Described(forecast(price, days).ravel(), y.loc[days].to_numpy().ravel(), {},
                         0.0, TOLERANCE)
    return described


# Qui es descriu: amb quina taula, sobre quins conjunts, com, i amb quins números. La
# prova de datació només l'ajusta l'arbre.
Describer = collections.namedtuple("Describer", "table sets describe numbers")


def _copied(name):
    """Els números d'una xarxa mentre no s'ha cercat, d'on es van copiar i els que ningú
    no cerca."""
    networks = ddnn_hyperparameters
    return networks.COPIED[name] | {
        "copied_from": networks.COPIED_FROM[name], "copied_trials": networks.COPIED_TRIALS,
        "batch": networks.BATCH, "epochs": networks.EPOCHS, "patience": networks.PATIENCE,
        "unsearched": True}


def describers():
    """Cada família que es descriu, per nom."""
    tree = boosting_hyperparameters
    net = linear_hyperparameters
    found = {
        tree.NAME: Describer(
            "hourly", (SCHEDULE, DATE_PROBE), _tree,
            {name: value for name, value in tree.UNSEARCHED.items() if name != "trees"}
            | {"most_trees": tree.MOST_TREES, "patience": tree.PATIENCE,
               "unsearched": True}),
        net.NAME: Describer(
            "hourly", (SCHEDULE,), _net,
            {"penalty": net.PENALTY, "mixing": net.MIXING, "max_iter": net.MAX_ITER,
             "unsearched": True}),
    }
    for name, distribution in DISTRIBUTIONS.items():
        found[name] = Describer("daily", (SCHEDULE,), _network(name, distribution),
                                _copied(name))
    found[POINT_FAMILY] = Describer("daily", (SCHEDULE,), _point_network,
                                    _copied(POINT_FAMILY))
    found[FLOOR] = Describer("hourly", (SCHEDULE,), _floor, {"degree": FLOOR_DEGREE})
    for name, (forecast, numbers) in naive.FORECASTS.items():
        found[name] = Describer("price", (SCHEDULE,), _naive(forecast), numbers)
    return found


def _tables(frame, day_ahead, scheduled):
    """Les taules de cada conjunt, per nom i per taula. La del preu és la sèrie sencera
    amb el preu real del conjunt, per a les prediccions ingènues, que copien dies que el
    conjunt no porta."""
    found = matrix.schedule(frame, day_ahead, scheduled)
    return {SCHEDULE: {"hourly": found.hourly, "daily": found.daily,
                       "price": (frame, found.actual)},
            DATE_PROBE: {"hourly": matrix.date_probe(frame, day_ahead, scheduled,
                                                     DEVELOPMENT[0])}}


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
              f"Fes un commit, o fes servir --tier {SMOKE}.")
        return 1
    table = describers()
    chosen = arguments.family or list(table)
    unknown = sorted(set(chosen) - set(table))
    if unknown:
        print(f"no es descriu cap família {unknown}; n'hi ha {sorted(table)}")
        return 1

    seeds, explained_days = DESCRIPTIONS[arguments.tier]
    where = run_files.run_folder(arguments.tier, STAGE, commit[:7])
    # Reprendre és de la mesura i prou: l'assaig admet l'arbre brut, i dues còpies
    # diferents del codi en compartirien el directori.
    if arguments.tier == SMOKE or arguments.replace:
        for family in chosen:
            for name in table[family].sets:
                shutil.rmtree(where / name / family, ignore_errors=True)

    tables = _tables(price.load(), forecasts.load(), schedule.load())
    swept = {name: getattr(parameters, name) for name in dir(parameters)
             if name.isupper() and not name.startswith("_")}
    for family in chosen:
        describer = table[family]
        for name in describer.sets:
            folder = where / name / family
            X, y = tables[name][describer.table]
            started, described_now = time.time(), []
            for seed in seeds:
                path = folder / run_files.SEED.format(seed=seed)
                if path.exists():
                    continue
                parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
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
            (folder / run_files.SEED_RECORD).write_text(
                json.dumps(run_files.plain(record), indent=2))
    describe_figures.draw(where)
    print(f"la passada és a {where}, amb les seves figures")
    return 0
