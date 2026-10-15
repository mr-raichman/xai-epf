"""El protocol de Trebbien et al. (2023) sobre el període que es descriu, amb la
partició de `contract/split.py`: la robustesa és la dispersió entre llavors.

`epf describe` ajusta cada família explicada sobre cada conjunt i cada llavor, la puntua
sobre les setmanes de prova al costat del sòl de l'ordre de mèrit, de les prediccions
ingènues i de la xarxa de punt, i n'explica les files de prova. Una distribució, la de
cada xarxa distribucional i la de la còpia del dia similar, es puntua també pel CRPS i
pel test de cobertura de Kupiec. Tot es desa per llavor, de manera que les figures
surten del que hi ha al disc i no d'un ajust nou. Mentre la cerca no s'ha fet, cada
família hi entra amb uns números que no s'han cercat, i el registre de cada una ho diu.
"""
import collections
import json
import shutil
import time

import numpy as np

from epf import parameters, paths
from epf.contract import features
from epf.contract.explanation import (PRICE_UNITS, TOLERANCE, TRANSFORMED_TOLERANCE,
                                      TRANSFORMED_UNITS, UNITLESS, explained_rows, group,
                                      summed)
from epf.contract.features import DATE_PROBE, SCHEDULE
from epf.contract.metrics import INTERVALS, scored, scored_distribution
from epf.contract.record import plain, save_model
from epf.contract.split import split, validation_parts
from epf.contract.study import configurations
from epf.data import forecasts, price, schedule
from epf.models.boosting import boosting, boosting_hyperparameters
from epf.models.ddnn import ddnn, ddnn_hyperparameters
from epf.models.ddnn.distributions import FAMILIES as DISTRIBUTIONS
from epf.models.ddnn.distributions import POINT, POINT_FAMILY
from epf.models.linear import linear, linear_hyperparameters
from epf.models.naive import naive
from epf.parameters import (BEST, BLACKOUT_WEEK, DESCRIPTIONS, DEVELOPMENT, HOURS_PER_DAY,
                            INTERACTION_ROWS, MEASUREMENT, SMOKE)
from epf.results import describe_figures, run_files

# -------------------------------------------------------------------- la passada

# La carpeta de les passades dins de cada nivell, `runs/<nivell>/describe/<commit>/`,
# amb els conjunts en carpetes per sota.
STAGE = "describe"

# El sòl de l'ordre de mèrit, que és a la passada com una família més perquè es puntua
# igual.
FLOOR = "floor"

# Grau del polinomi del sòl, el cúbic de Trebbien et al. (2023).
FLOOR_DEGREE = 3

# El que surt de descriure una família en una llavor: la predicció i el preu de les
# files de prova en el mateix ordre, el que es desa de l'explicació, el pitjor residu de
# l'explicació contra la predicció que explica, amb el marge que li toca, de les que
# prediuen una distribució les puntuacions de la distribució, i, de les explicades, les
# sumes que es tornen a fer sobre el fitxer desat.
Described = collections.namedtuple(
    "Described", "predicted actual arrays gap tolerance scores stored",
    defaults=(None, None))

# Les parts de la llavor d'on la predicció ingènua pren els errors per fer-se
# probabilística: el que Marcjasz et al. (2023) diuen dins la mostra, l'entrenament i la
# validació, on la còpia no tria res.
IN_SAMPLE = ("training", "validation")


def _parts_of(X, parts, explained):
    """Les files de cada part i les que s'expliquen, com a màscares."""
    return [X.index.isin(days) for days in (*parts, explained)]


def _tree_sums(found, best):
    """Les sumes de l'arbre sobre el seu fitxer: la base més els valors de cada fila
    contra la seva predicció i, a la millor configuració, cada fila d'interaccions
    contra el valor de la seva columna."""
    sums = [(*summed(found["base"], found["values"], 1),
             found["predicted"][explained_rows(found, HOURS_PER_DAY)])]
    if best:
        sums.append((*summed(0.0, found["interactions"], 2),
                     found["values"][found["interaction_rows"]]))
    return sums


def _net_sums(found):
    """La suma de la xarxa elàstica sobre el seu fitxer, a l'espai transformat."""
    return [(*summed(found["base"], found["values"], 1), found["transformed"])]


def _network_sums(found):
    """La suma d'una xarxa distribucional sobre el seu fitxer, contra els paràmetres de
    prova dels dies explicats. La de la base contra la mitjana dels dies d'entrenament
    només es fa a la memòria: el fitxer no en desa els paràmetres."""
    return [(*summed(found["base"], found["values"], 1),
             found["parameters"][explained_rows(found, 1)])]


def _tree(X, y, parts, explained, seed, numbers, best):
    """L'arbre, aturat quan qualsevol de les quatre parts de la validació deixa de
    millorar, i, a la millor configuració de la llavor, les interaccions sobre una mostra
    de les files que s'expliquen."""
    train, _, test, shown = _parts_of(X, parts, explained)
    weeks = validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK)
    validation = [(X[part], y[part]) for part in (X.index.isin(days) for days in weeks)]
    model = boosting.fit_rows(X[train], y[train], numbers,
                              boosting_hyperparameters.MOST_TREES, validation=validation)
    values, base = boosting.attribute_rows(model, X[shown])
    # La base més els valors torna la predicció.
    gap = np.abs(base + values.sum(axis=1) - boosting.predict_rows(model, X[shown])).max()
    arrays = {"columns": np.array(X.columns, dtype=str),
              "groups": np.array([group(column) for column in X.columns]),
              "inputs": X[shown].to_numpy(), "units": np.array([PRICE_UNITS]),
              "values": values, "base": base, "trees": model.best_iteration_}
    if best:
        count = int(shown.sum())
        sample = np.sort(np.random.default_rng(seed).choice(
            count, min(INTERACTION_ROWS, count), replace=False))
        interactions = boosting.interaction_rows(model, X[shown].iloc[sample])
        # Cada fila d'interaccions torna el valor de la seva columna.
        gap = max(gap, np.abs(interactions.sum(axis=2) - values[sample]).max())
        arrays |= {"interaction_rows": sample, "interactions": interactions}
    return Described(boosting.predict_rows(model, X[test]), y[test].to_numpy(), arrays,
                     gap, TOLERANCE, stored=lambda found: _tree_sums(found, best))


def _net(X, y, parts, explained, seed, numbers, best):
    """La xarxa elàstica, amb el càstig i la barreja de la configuració."""
    train, _, test, shown = _parts_of(X, parts, explained)
    model = linear.fit_rows(X[train], y[train], numbers["penalty"], numbers["mixing"])
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
                     gap, TRANSFORMED_TOLERANCE, stored=_net_sums)


def _network(name, distribution):
    """La xarxa distribucional d'una distribució, aturada sobre tota la validació, i
    explicada paràmetre a paràmetre contra els dies d'entrenament de la llavor.

    Una xarxa per distribució i llavor, on Marcjasz et al. (2023) en mitjanen quatre:
    cada membre seria una cerca més, i el que costa és la cerca.
    """
    def described(X, y, parts, explained, seed, numbers, best):
        train, validate, test, shown = _parts_of(X, parts, explained)
        fitted = ddnn.fit_rows(distribution, (X[train], y[train]),
                               (X[validate], y[validate]), numbers)
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
        scores = scored_distribution(
            y[test], lambda probabilities: distribution.quantiles(found, probabilities))
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
                          "parameters": found}, gap, TOLERANCE, scores,
                         stored=_network_sums)
    return described


def _point_network(X, y, parts, explained, seed, numbers, best):
    """La xarxa de punt de Marcjasz et al. (2023), la distribucional amb la sortida
    canviada, ajustada sobre les mateixes files i aturada igual."""
    train, validate, test, _ = _parts_of(X, parts, explained)
    fitted = ddnn.fit_rows(POINT, (X[train], y[train]), (X[validate], y[validate]),
                           numbers)
    return Described(ddnn.predict_rows(POINT, fitted, X[test]).ravel(),
                     y[test].to_numpy().ravel(), {}, 0.0, TOLERANCE)


def _floor(X, y, parts, explained, seed, numbers, best):
    """El sòl de l'ordre de mèrit de Trebbien et al. (2023): un polinomi cúbic de la
    càrrega residual a l'hora, la demanda menys l'eòlica i la fotovoltaica previstes."""
    train, _, test, _ = _parts_of(X, parts, explained)
    load = (X["demand"] - X["wind"] - X["solar"]).to_numpy()
    coefficients = np.polyfit(load[train], y[train].to_numpy(), FLOOR_DEGREE)
    return Described(np.polyval(coefficients, load[test]), y[test].to_numpy(),
                     {"coefficients": coefficients, "load_training": load[train],
                      "actual_training": y[train].to_numpy()}, 0.0, TOLERANCE)


def _naive(forecast, bootstrapped):
    """Una predicció ingènua de Lago et al. (2021), presa de la sèrie del preu als dies de
    prova que té la taula del conjunt, de manera que es puntua sobre les files de totes.

    La que Marcjasz et al. (2023) fan probabilística es puntua també com a distribució,
    amb els seus errors als dies de `IN_SAMPLE`, que es desen.
    """
    def described(price, y, parts, explained, seed, numbers, best):
        days = y.index[y.index.isin(parts.test)]
        copied = forecast(price, days)
        actual = y.loc[days]
        if not bootstrapped:
            return Described(copied.ravel(), actual.to_numpy().ravel(), {}, 0.0, TOLERANCE)
        sample = y.index[np.any([y.index.isin(getattr(parts, part)) for part in IN_SAMPLE],
                                axis=0)]
        errors = y.loc[sample].to_numpy() - forecast(price, sample)
        scores = scored_distribution(
            actual, lambda probabilities: naive.bootstrapped(copied, errors, probabilities))
        return Described(copied.ravel(), actual.to_numpy().ravel(), {"errors": errors}, 0.0,
                         TOLERANCE, scores)
    return described


# Qui es descriu: amb quina taula, sobre quins conjunts i com; els números que cap cerca
# no toca; els de lloc mentre no hi ha cerca; i quantes de les millors configuracions de
# la cerca s'hi ajusten a cada llavor, cap per a una família que no es cerca. La prova de
# datació només l'ajusta l'arbre.
Describer = collections.namedtuple("Describer", "table sets describe fixed placeholder kept")


def _copied(name):
    """Els números d'una xarxa mentre no s'ha cercat, i d'on es van copiar."""
    networks = ddnn_hyperparameters
    return networks.COPIED[name] | {"copied_from": networks.COPIED_FROM[name],
                                    "copied_trials": networks.COPIED_TRIALS}


def describers():
    """Cada família que es descriu, per nom. L'arbre i la xarxa elàstica s'ajusten amb
    les deu millors configuracions de cada llavor, i cada xarxa amb la millor."""
    tree = boosting_hyperparameters
    net = linear_hyperparameters
    networks = ddnn_hyperparameters
    trained = {"batch": networks.BATCH, "epochs": networks.EPOCHS,
               "patience": networks.PATIENCE}
    found = {
        tree.NAME: Describer(
            "hourly", (SCHEDULE, DATE_PROBE), _tree,
            {"most_trees": tree.MOST_TREES, "patience": tree.PATIENCE},
            {name: value for name, value in tree.UNSEARCHED.items() if name != "trees"},
            BEST),
        net.NAME: Describer("hourly", (SCHEDULE,), _net, {"max_iter": net.MAX_ITER},
                            {"penalty": net.PENALTY, "mixing": net.MIXING}, BEST),
    }
    for name, distribution in DISTRIBUTIONS.items():
        found[name] = Describer("daily", (SCHEDULE,), _network(name, distribution),
                                trained, _copied(name), 1)
    found[POINT_FAMILY] = Describer("daily", (SCHEDULE,), _point_network, trained,
                                    _copied(POINT_FAMILY), 1)
    found[FLOOR] = Describer("hourly", (SCHEDULE,), _floor, {"degree": FLOOR_DEGREE},
                             None, 0)
    for name, (forecast, numbers) in naive.FORECASTS.items():
        bootstrapped = name == naive.BOOTSTRAPPED
        found[name] = Describer(
            "price", (SCHEDULE,), _naive(forecast, bootstrapped),
            numbers | ({"bootstrap_from": IN_SAMPLE} if bootstrapped else {}), None, 0)
    return found


def _tables(frame, day_ahead, scheduled):
    """Les taules de cada conjunt, per nom i per taula. La del preu és la sèrie sencera
    amb el preu real del conjunt, per a les prediccions ingènues, que copien dies que el
    conjunt no porta."""
    found = features.tables(frame, day_ahead, scheduled)
    return {SCHEDULE: {"hourly": found.hourly, "daily": found.daily,
                       "price": (frame, found.actual)},
            DATE_PROBE: {"hourly": features.date_probe(frame, day_ahead, scheduled,
                                                     DEVELOPMENT[0])}}


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
    where = paths.run_folder(arguments.tier, STAGE, commit[:7])
    # Reprendre és de la mesura i prou: l'assaig admet l'arbre brut, i dues còpies
    # diferents del codi en compartirien el directori.
    if arguments.tier == SMOKE or arguments.replace:
        for family in chosen:
            for name in table[family].sets:
                shutil.rmtree(where / name / family, ignore_errors=True)

    tables = _tables(price.load(), forecasts.load(), schedule.load())
    # Les configuracions de totes les famílies abans d'ajustar-ne cap, de manera que una
    # cerca a mitges o feta sobre unes altres files es refusa abans de gastar-hi res. La
    # prova de datació llegeix la cerca de l'arbre sobre el seu conjunt.
    try:
        plans = {(family, name): configurations(
                     arguments.tier, family, table[family],
                     tables[SCHEDULE][table[family].table][0].columns, seeds,
                     best_only=name == DATE_PROBE)
                 for family in chosen for name in table[family].sets}
    except ValueError as refused:
        print(refused)
        return 1
    swept = {name: getattr(parameters, name) for name in dir(parameters)
             if name.isupper() and not name.startswith("_")}
    for family in chosen:
        describer = table[family]
        for name in describer.sets:
            folder = where / name / family
            X, y = tables[name][describer.table]
            started, described_now = time.time(), []
            for seed in seeds:
                parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
                explained = parts.test if explained_days is None else parts.test[:explained_days]
                for configuration in plans[family, name][seed]:
                    rank = configuration["rank"]
                    path = folder / paths.SEED.format(seed=seed, rank=rank)
                    if path.exists():
                        continue
                    start = time.time()
                    described = describer.describe(X, y, parts, explained, seed,
                                                   configuration.get("numbers"), rank == 0)
                    if described.gap > described.tolerance:
                        raise ValueError(
                            f"l'explicació de {family} sobre {name}, llavor {seed}, "
                            f"configuració {rank}, no torna la seva predicció: se n'allunya "
                            f"{described.gap:.3g}, per sobre del marge de "
                            f"{described.tolerance}")
                    scores = scored(described.predicted, described.actual)
                    scores |= described.scores or {}
                    save_model(path, parts, explained, described, scores, rank)
                    described_now.append([seed, rank])
                    distribution = "" if described.scores is None else (
                        f", CRPS {scores['crps']:.2f}, hores que passen el test de Kupiec: "
                        + " i ".join(f"{scores[f'kupiec_{level}']} al {level}%"
                                     for level in INTERVALS))
                    print(f"{name} {family} llavor {seed} configuració {rank}: R² "
                          f"{scores['r2']:.3f}, MAE {scores['mae']:.2f} EUR/MWh"
                          f"{distribution}, residu de l'explicació {described.gap:.1e}, "
                          f"{time.time() - start:.1f} s", flush=True)
            record = {"commit": commit, "tier": arguments.tier, "set": name,
                      "family": family, "seeds": list(seeds),
                      "explained_days": explained_days, "fixed": describer.fixed,
                      "configurations": plans[family, name],
                      "described_now": described_now,
                      "seconds_now": round(time.time() - started, 1),
                      "parameters": swept}
            (folder / paths.SEED_RECORD).write_text(
                json.dumps(plain(record), indent=2))
    describe_figures.draw(where)
    print(f"la passada és a {where}, amb les seves figures")
    return 0
