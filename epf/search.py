"""La cerca de Trebbien et al. (2023): a cada llavor i família, configuracions a l'atzar de
l'espai de la família, cada una ajustada amb les files d'entrenament de la llavor i
puntuada sobre les de validació, sense tocar mai les de prova, i les millors desades.

Cada família es puntua com ja es puntuava la seva cerca: l'arbre i la xarxa elàstica pel
MAE en EUR/MWh, i les xarxes per la seva pèrdua de Keras sobre la validació, amb les L1 a
dins, que és el que torna el codi de Marcjasz et al. (2023). Una graella es recorre sense
repetir cap punt; un espai continu es mostreja a l'atzar.
"""
import collections
import shutil

import optuna

from epf import paths
from epf.contract import features
from epf.contract.features import SCHEDULE
from epf.contract.metrics import mae
from epf.contract.split import split, validation_parts
from epf.contract.study import STUDY, folder, record, run, save
from epf.data import forecasts, price, schedule
from epf.models.boosting import boosting, boosting_hyperparameters
from epf.models.boosting import search as boosting_search
from epf.models.ddnn import ddnn
from epf.models.ddnn import search as ddnn_search
from epf.models.ddnn.distributions import FAMILIES as DISTRIBUTIONS
from epf.models.ddnn.distributions import POINT, POINT_FAMILY
from epf.models.linear import linear, linear_hyperparameters
from epf.models.linear import search as linear_search
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, MEASUREMENT, SEARCH_TRIALS, SMOKE
from epf.results import run_files

# Una família que es cerca: la taula del conjunt que llegeix, la seva graella si en té, i
# la prova, que d'una prova d'Optuna, les files d'entrenament i les de validació, totes
# dues parelles `(X, y)`, i la llavor de la partició en fa els números de l'ajust i la
# seva puntuació.
Searcher = collections.namedtuple("Searcher", "table grid trial")


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
    return point, mae(boosting.predict_rows(model, X_validate), y_validate)


def _net(trial, train, validate, seed):
    """Un punt de la graella d'Uniejewski et al. (2016): la barreja i el pas de la seva
    seqüència, que es calcula sobre les files d'entrenament."""
    (X, y), (X_validate, y_validate) = train, validate
    mixing, step = linear_search.space(trial)
    penalty = float(linear_search.penalties(X, y)[mixing][step])
    model = linear.fit_rows(X, y, penalty, mixing)
    return ({"penalty": penalty, "mixing": mixing},
            mae(linear.predict_rows(model, X_validate), y_validate))


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

    tables = features.tables(price.load(), forecasts.load(), schedule.load())
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
