"""El que comparteixen les cerques de tots els models.

Optuna desa cada intent en un study SQLite, de manera que una cerca interrompuda
continua on era. Quan una llavor té tots els seus intents, el seu registre en desa els
millors, en la forma en què els pren l'ajust, i sobre quines files es van fer: les
columnes de la matriu i la partició. Una cerca continua sobre un altre codi, un altre
dataset o unes altres biblioteques, i cada intent en desa les empremtes, però no sobre
unes altres files; i una passada no llegeix un registre fet sobre unes altres files.
"""

import json
import os

import numpy as np
import optuna

from epf import Refusal, paths
from epf.contract.record import plain, write_record
from epf.parameters import BEST, BLACKOUT_WEEK, DEVELOPMENT, GAP, SEED, SHARES

# Els fitxers d'una cerca, per llavor: el study d'Optuna i el registre del que ha triat.
STUDY = "seed_{seed}.db"
RECORD = "seed_{seed}.json"


def folder(smoke, family):
    """Torna la carpeta de la cerca d'una família, la d'assaig si `smoke`."""
    return paths.SEARCHES / f"{paths.SMOKE_PREFIX if smoke else ''}{family}"


def made_on(columns):
    """Torna les files d'una cerca: les columnes de la matriu i la partició."""
    return plain(
        {
            "columns": list(columns),
            "period": DEVELOPMENT,
            "deleted": BLACKOUT_WEEK,
            "shares": SHARES,
            "gap": GAP,
        }
    )


def completed(study):
    return [
        one for one in study.trials if one.state == optuna.trial.TrialState.COMPLETE
    ]


def _seed(*parts):
    """Torna una llavor d'Optuna feta de la del projecte i de `parts`, enters o textos.

    No canvia d'un procés a un altre, com faria el `hash` d'un text.
    """
    entropy = [SEED]
    for part in parts:
        entropy += list(part.encode()) if isinstance(part, str) else [part]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _sampler(searcher, family, seed, study):
    """Torna el sampler d'un study: la graella del cercador, `GRID`, o l'espai.

    Una graella es barreja amb una llavor de la família i de la partició, i l'intent `k`
    en pren el punt `k`: una cerca continuada segueix per on anava, i cap punt no es
    repeteix. Un espai continu es mostreja amb una llavor que compta també els intents
    que el study ja porta, perquè una cerca continuada no torni a proposar els primers.
    El nom de la família hi entra perquè dues famílies del mateix espai no proposin les
    mateixes.
    """
    if searcher.GRID is not None:
        return optuna.samplers.GridSampler(
            {name: list(values) for name, values in searcher.GRID.items()},
            seed=_seed(family, seed),
        )
    return optuna.samplers.RandomSampler(seed=_seed(family, seed, len(study.trials)))


def run(store, family, searcher, X, y, parts, seed, trials, fingerprints):
    """Fa els intents que falten fins a `trials` al study d'una família en una llavor.

    Torna el study. `searcher` és el `search.py` del model de la família: la seva
    graella, `GRID`, i l'intent, `trial`. Cada intent s'ajusta amb les files
    d'entrenament de `parts` i es puntua sobre les de validació, i la matriu no n'hi
    dona cap altra. Els números que en surten es desen a l'intent, en la forma en què
    els pren l'ajust, amb les empremtes, `fingerprints`, del codi, del dataset i de les
    biblioteques que el fan. Refusa un study fet sobre unes altres files.
    """
    # El camí amb `/`, que és el que la URL d'SQLite admet també a Windows.
    study = optuna.create_study(
        study_name=f"{family}_seed_{seed}",
        direction="minimize",
        storage=f"sqlite:///{store.as_posix()}",
        load_if_exists=True,
    )
    now = made_on(X.columns)
    if not study.trials:
        study.set_user_attr("made_on", now)
    elif study.user_attrs.get("made_on") != now:
        raise Refusal(
            f"el study es va fer sobre altres files; esborrar {paths.relative(store)}"
        )
    study.sampler = _sampler(searcher, family, seed, study)
    train, validate = X.index.isin(parts.training), X.index.isin(parts.validation)

    def objective(optuna_trial):
        numbers, score = searcher.trial(
            family, optuna_trial, (X[train], y[train]), (X[validate], y[validate]), seed
        )
        optuna_trial.set_user_attr("numbers", plain(numbers))
        # La càrrega de la màquina en acabar l'intent; Windows no la dona, i hi queda
        # buida.
        optuna_trial.set_user_attr(
            "load", os.getloadavg() if hasattr(os, "getloadavg") else None
        )
        for name, value in fingerprints.items():
            optuna_trial.set_user_attr(name, value)
        return score

    remaining = trials - len(completed(study))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study


def record(study, family, seed, smoke):
    """Torna el registre d'una cerca acabada, amb els seus `BEST` millors intents."""
    done = sorted(completed(study), key=lambda one: one.value)
    return {
        "family": family,
        "smoke": smoke,
        "seed": seed,
        "trials": len(done),
        "made_on": study.user_attrs["made_on"],
        "best": [
            {
                "numbers": one.user_attrs["numbers"],
                "score": one.value,
                "trial": one.number,
                "code": one.user_attrs["code"],
                "dataset": one.user_attrs["dataset"],
                "environment": one.user_attrs["environment"],
            }
            for one in done[:BEST]
        ],
    }


def save(found, smoke):
    """Desa el registre d'una cerca acabada, `found`, sencer o gens."""
    write_record(
        folder(smoke, found["family"]) / RECORD.format(seed=found["seed"]), found
    )


def searched(smoke, family, seed, columns):
    """Torna el registre de la cerca d'una família en una llavor, o `None` si no en té.

    Refusa un registre fet sobre unes altres columnes, `columns`, o una altra partició:
    els seus números s'haurien triat per unes altres files, i res no ho diria.
    """
    path = folder(smoke, family) / RECORD.format(seed=seed)
    if not path.exists():
        return None
    found = json.loads(path.read_text(encoding="utf-8"))
    if found["made_on"] != made_on(columns):
        study = path.with_name(STUDY.format(seed=seed))
        raise Refusal(
            f"la cerca de {family} llavor {seed} es va fer sobre altres files; "
            f"esborrar {paths.relative(path)} i {paths.relative(study)}, cercar amb "
            f'"uv run epf search"'
        )
    return found


def configurations(smoke, plan, columns, seeds):
    """Torna les configuracions que s'ajusten a cada llavor, de la millor a la pitjor.

    Cada una porta el seu lloc a la cerca. `plan` és l'entrada de la família a
    `FAMILIES` del seu model: les `kept` millors de la cerca de la família `searched_as`
    del mateix tipus, si totes les llavors en tenen; els números de lloc, `placeholder`,
    marcats, si cap llavor no en té; i una sola sense números per a una família que no
    es cerca, `kept` zero. `columns` són les de la matriu sobre la qual es cerca, que la
    cerca compara amb les seves. Refusa una cerca que només té algunes llavors: la
    dispersió barrejaria llavors cercades i sense cercar, i res no ho diria.
    """
    if not plan["kept"]:
        return {seed: [{"rank": 0}] for seed in seeds}
    family = plan["searched_as"]
    found = {seed: searched(smoke, family, seed, columns) for seed in seeds}
    missing = [seed for seed, record in found.items() if record is None]
    if len(missing) == len(seeds):
        return {
            seed: [{"rank": 0, "numbers": plan["placeholder"], "unsearched": True}]
            for seed in seeds
        }
    if missing:
        asked = (
            f"la llavor {missing[0]}"
            if len(missing) == 1
            else f"les llavors {', '.join(map(str, missing))}"
        )
        raise Refusal(
            f'la cerca de {family} no té {asked}; acabar amb "uv run epf search"'
        )
    return {
        seed: [
            {
                "rank": rank,
                "numbers": one["numbers"],
                "score": one["score"],
                "trial": one["trial"],
                "trials": record["trials"],
            }
            for rank, one in enumerate(record["best"][: plan["kept"]])
        ]
        for seed, record in found.items()
    }
