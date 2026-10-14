"""La cerca de Trebbien et al. (2023): a cada llavor i família, configuracions a l'atzar de
l'espai de la família, cada una ajustada amb les files d'entrenament de la llavor i
puntuada sobre les de validació, sense tocar mai les de prova, i les millors desades.

Cada família es puntua com ja es puntuava la seva cerca: l'arbre i la xarxa elàstica pel
MAE en EUR/MWh, i les xarxes per la seva pèrdua de Keras sobre la validació, amb les L1 a
dins, que és el que torna el codi de Marcjasz et al. (2023). Una graella es recorre sense
repetir cap punt; un espai continu es mostreja a l'atzar.

Optuna desa cada prova en un estudi SQLite, de manera que una cerca interrompuda es
reprèn on era. Quan una llavor té totes les seves proves, el seu registre en desa les
millors en la forma en què les pren l'ajust, i sobre quines files es van fer: les
columnes del conjunt i la partició. Una passada no llegeix un registre fet sobre unes
altres files, i una cerca no es reprèn sobre unes altres files.
"""
import collections
import json
import os
import shutil
import time

import numpy as np
import optuna

from epf import matrix, paths
from epf.data import forecasts, price, schedule
from epf.matrix import SCHEDULE
from epf.models.boosting import boosting, boosting_hyperparameters
from epf.models.boosting import search as boosting_search
from epf.models.ddnn import ddnn
from epf.models.ddnn import search as ddnn_search
from epf.models.ddnn.distributions import FAMILIES as DISTRIBUTIONS
from epf.models.ddnn.distributions import POINT, POINT_FAMILY
from epf.models.linear import linear, linear_hyperparameters
from epf.models.linear import search as linear_search
from epf.parameters import (BEST, BLACKOUT_WEEK, DEVELOPMENT, GAP, MEASUREMENT,
                            SEARCH_TRIALS, SEED, SHARES, SMOKE)
from epf.results import run_files
from epf.split import split, validation_parts

# Els fitxers d'una cerca, per llavor: l'estudi d'Optuna i el registre del que ha triat.
STUDY = "seed_{seed}.db"
RECORD = "seed_{seed}.json"

# Una família que es cerca: la taula del conjunt que llegeix, la seva graella si en té, i
# la prova, que d'una prova d'Optuna, les files d'entrenament i les de validació, totes
# dues parelles `(X, y)`, i la llavor de la partició en fa els números de l'ajust i la
# seva puntuació.
Searcher = collections.namedtuple("Searcher", "table grid trial")


def _mae(predicted, actual):
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def _tree(trial, train, validate, seed):
    """Un punt de la graella, aturat quan qualsevol de les quatre parts de la validació
    deixa de millorar, com a la passada, i puntuat pel MAE de tota la validació."""
    (X, y), (X_validate, y_validate) = train, validate
    point = boosting_search.space(trial)
    weeks = validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK)
    parts = [X_validate.index.isin(days) for days in weeks]
    model = boosting.fit_rows(X, y, point, boosting_hyperparameters.MOST_TREES,
                              validation=[(X_validate[part], y_validate[part])
                                          for part in parts])
    return point, _mae(boosting.predict_rows(model, X_validate), y_validate)


def _net(trial, train, validate, seed):
    """Un punt de la graella d'Uniejewski et al. (2016): la barreja i el pas de la seva
    seqüència, que es calcula sobre les files d'entrenament."""
    (X, y), (X_validate, y_validate) = train, validate
    mixing, step = linear_search.space(trial)
    penalty = float(linear_search.penalties(X, y)[mixing][step])
    model = linear.fit_rows(X, y, penalty, mixing)
    return ({"penalty": penalty, "mixing": mixing},
            _mae(linear.predict_rows(model, X_validate), y_validate))


def _network(distribution, penalised):
    """Una configuració de l'espai de l'article, aturada sobre tota la validació i
    puntuada per la pèrdua de Keras sobre ella."""
    def trial_of(trial, train, validate, seed):
        X_validate, y_validate = validate
        tuned = ddnn_search.space(trial, distribution, penalised)
        fitted = ddnn.fit_rows(distribution, train, validate, tuned)
        return tuned, float(fitted.evaluate(ddnn.inputs(X_validate).to_numpy(),
                                            y_validate.to_numpy(dtype=float), verbose=0))
    return trial_of


def searchers():
    """Cada família que es cerca, per nom. La xarxa de punt no cerca cap L1 a la sortida."""
    found = {
        boosting_hyperparameters.NAME: Searcher("hourly", boosting_search.GRID, _tree),
        linear_hyperparameters.NAME: Searcher("hourly", linear_search.GRID, _net),
    }
    for name, distribution in DISTRIBUTIONS.items():
        found[name] = Searcher("daily", None, _network(distribution, penalised=True))
    found[POINT_FAMILY] = Searcher("daily", None, _network(POINT, penalised=False))
    return found


def folder(tier, family):
    """On viu la cerca d'una família en un nivell, sobre l'únic conjunt que es cerca."""
    return paths.SEARCHES / tier / SCHEDULE / family


def made_on(columns):
    """Les files sobre què es fa una cerca: les columnes del conjunt i la partició."""
    return run_files.plain({"columns": list(columns), "period": DEVELOPMENT,
                            "deleted": BLACKOUT_WEEK, "shares": SHARES, "gap": GAP})


def completed(study):
    """Les proves acabades d'un estudi."""
    return [one for one in study.trials if one.state == optuna.trial.TrialState.COMPLETE]


def _seed(*parts):
    """Una llavor d'Optuna de la del projecte i de `parts`, enters o textos, que no canvia
    d'un procés a un altre com el `hash` d'un text."""
    entropy = [SEED]
    for part in parts:
        entropy += list(part.encode()) if isinstance(part, str) else [part]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _sampler(searcher, family, seed, study):
    """Com tria cada prova la seva configuració.

    Una graella es barreja amb una llavor de la família i de la partició, i la prova `k`
    en pren el punt `k`: una cerca represa continua per on anava, i cap punt no es repeteix.
    Un espai continu es mostreja amb una llavor que compta també les proves que l'estudi
    ja porta, perquè una cerca represa no torni a proposar les primeres. El nom de la
    família hi entra perquè dues famílies del mateix espai no proposin les mateixes.
    """
    if searcher.grid is not None:
        return optuna.samplers.GridSampler(
            {name: list(values) for name, values in searcher.grid.items()},
            seed=_seed(family, seed))
    return optuna.samplers.RandomSampler(seed=_seed(family, seed, len(study.trials)))


def run(store, family, searcher, X, y, parts, seed, trials):
    """L'estudi d'una família en una llavor, amb les proves que li falten fins a `trials`.

    Cada prova s'ajusta amb les files d'entrenament de `parts` i es puntua sobre les de
    validació, i la taula no n'hi dóna cap altra. Els números que en surten es desen a la
    prova, en la forma en què els pren l'ajust. Un estudi fet sobre unes altres files es
    refusa en lloc de reprendre's.
    """
    study = optuna.create_study(study_name=f"{family}_seed_{seed}", direction="minimize",
                                storage=f"sqlite:///{store}", load_if_exists=True)
    now = made_on(X.columns)
    if not study.trials:
        study.set_user_attr("made_on", now)
    elif study.user_attrs.get("made_on") != now:
        raise ValueError(f"l'estudi de {family}, llavor {seed}, es va fer sobre unes altres "
                         f"files que les d'ara: torna'l a començar amb --replace")
    study.sampler = _sampler(searcher, family, seed, study)
    train, validate = X.index.isin(parts.training), X.index.isin(parts.validation)

    def objective(trial):
        start = time.time()
        numbers, score = searcher.trial(trial, (X[train], y[train]),
                                        (X[validate], y[validate]), seed)
        trial.set_user_attr("numbers", run_files.plain(numbers))
        # La càrrega de la màquina en acabar, perquè la durada de la prova, que Optuna desa,
        # porti les condicions en què es va prendre.
        trial.set_user_attr("load", os.getloadavg())
        print(f"{family} llavor {seed} prova {len(completed(study)) + 1} de {trials}: "
              f"{score:.4g}, {time.time() - start:.1f} s", flush=True)
        return score

    remaining = trials - len(completed(study))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study


def record(study, family, seed, tier, commit):
    """El registre d'una cerca acabada: les `BEST` millors proves, de la millor a la
    pitjor, amb els seus números i la seva puntuació, quantes proves les van triar i sobre
    quines files."""
    done = sorted(completed(study), key=lambda one: one.value)
    return {"commit": commit, "tier": tier, "set": SCHEDULE, "family": family,
            "seed": seed, "trials": len(done), "made_on": study.user_attrs["made_on"],
            "best": [{"numbers": one.user_attrs["numbers"], "score": one.value,
                      "trial": one.number} for one in done[:BEST]]}


def save(found, tier):
    """Desa un registre, primer a un fitxer a part i després al seu nom, de manera que un
    registre amb nom és sempre una cerca acabada."""
    path = folder(tier, found["family"]) / RECORD.format(seed=found["seed"])
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(found, indent=2) + "\n")
    partial.replace(path)


def searched(tier, family, seed, columns):
    """El registre de la cerca d'una família en una llavor, o `None` si no n'hi ha cap
    d'acabada.

    Un registre fet sobre unes altres columnes o una altra partició es refusa: els seus
    números s'haurien triat per unes altres files, i res no ho diria.
    """
    path = folder(tier, family) / RECORD.format(seed=seed)
    if not path.exists():
        return None
    found = json.loads(path.read_text())
    if found["made_on"] != made_on(columns):
        raise ValueError(f"la cerca de {family}, llavor {seed}, es va fer sobre unes altres "
                         f"files que les d'ara: torna-la a fer amb epf search --replace")
    return found


def main(arguments):
    commit = run_files.git("rev-parse", "HEAD")
    if arguments.tier == MEASUREMENT and run_files.git("status", "--porcelain"):
        print("l'arbre té canvis sense desar. Una cerca de mesura desa el commit que la fa, "
              "i amb canvis sense desar no diria quin codi l'ha feta. Fes un commit, o fes "
              f"servir --tier {SMOKE}.")
        return 1
    table = searchers()
    chosen = arguments.family or list(table)
    unknown = sorted(set(chosen) - set(table))
    if unknown:
        print(f"no es cerca cap família {unknown}; n'hi ha {sorted(table)}")
        return 1

    seeds, trials = SEARCH_TRIALS[arguments.tier]
    if arguments.seed:
        outside = sorted(set(arguments.seed) - set(seeds))
        if outside:
            print(f"les llavors {outside} no són del nivell {arguments.tier}, que en té "
                  f"{list(seeds)}")
            return 1
        seeds = [seed for seed in seeds if seed in arguments.seed]
    # Reprendre és de la mesura i prou, com a la passada.
    if arguments.tier == SMOKE or arguments.replace:
        for family in chosen:
            shutil.rmtree(folder(arguments.tier, family), ignore_errors=True)

    tables = matrix.schedule(price.load(), forecasts.load(), schedule.load())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for family in chosen:
        searcher = table[family]
        X, y = getattr(tables, searcher.table)
        where = folder(arguments.tier, family)
        where.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            study = run(where / STUDY.format(seed=seed), family, searcher, X, y,
                        split(seed, DEVELOPMENT, BLACKOUT_WEEK), seed, trials)
            found = record(study, family, seed, arguments.tier, commit)
            save(found, arguments.tier)
            print(f"{family} llavor {seed}: {found['trials']} proves, la millor "
                  f"{found['best'][0]['score']:.4g}", flush=True)
    print(f"les cerques són a {paths.SEARCHES / arguments.tier / SCHEDULE}")
    return 0
