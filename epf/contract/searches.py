"""El que comparteixen les cerques de tots els models.

Optuna desa cada trial en un study SQLite, de manera que una cerca interrompuda continua
on era. Només el procés pare hi escriu: en demana cada trial i en desa el resultat, i
un altre procés l'ajusta. Quan una seed té tots els seus trials, el seu registre els
desa tots, del millor al pitjor, en la forma en què els pren l'ajust, i sobre quines
files es van fer: les columnes de la matriu i la partició, i amb quina `PROJECT_SEED`.
Una cerca continua sobre un altre codi, un altre dataset o unes altres biblioteques, i
cada trial en desa les empremtes, però no sobre unes altres files ni amb una altra
`PROJECT_SEED`; i una passada no llegeix un registre fet així.
"""

import json
import os

import numpy as np
import optuna

from epf import Refusal, paths
from epf.contract.record import plain, write_record
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, GAP, PROJECT_SEED, SHARES

# Els fitxers d'una cerca, per seed: el study d'Optuna i el registre del que ha triat.
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


def _different(made, now):
    """Torna en què es va fer diferent una cerca, `made`, o `None` si no ho va fer.

    `now` són les files d'ara. Una cerca que no diu la seva `PROJECT_SEED` es va fer
    amb una altra.
    """
    if made.get("made_on") != now:
        return "sobre altres files"
    if made.get("PROJECT_SEED") != PROJECT_SEED:
        return "amb una altra PROJECT_SEED"
    return None


def _seed(*parts):
    """Torna una seed d'Optuna feta de la del projecte i de `parts`, enters o textos.

    No canvia d'un procés a un altre, com faria el `hash` d'un text.
    """
    entropy = [PROJECT_SEED]
    for part in parts:
        entropy += list(part.encode()) if isinstance(part, str) else [part]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _sampler(searcher, family, seed, study):
    """Torna el sampler d'un study: la graella del cercador, `GRID`, o l'espai.

    Una graella es barreja amb una seed de la família i de la partició, i el trial `k`
    en pren el punt `k`: una cerca continuada segueix per on anava, i cap punt no es
    repeteix. Un espai continu es mostreja amb una seed que compta també els trials que
    el study ja porta, perquè una cerca continuada no torni a proposar els primers. El
    nom de la família hi entra perquè dues famílies del mateix espai no proposin les
    mateixes.
    """
    if searcher.GRID is not None:
        return _Grid(
            {name: list(values) for name, values in searcher.GRID.items()},
            seed=_seed(family, seed),
        )
    return optuna.samplers.RandomSampler(seed=_seed(family, seed, len(study.trials)))


class _Grid(optuna.samplers.GridSampler):
    """La graella d'Optuna, que no atura el study quan se n'acaba l'últim punt.

    Ho faria amb `study.stop`, que fora d'`optimize` falla, i el pare demana i desa cada
    trial fora d'`optimize`. Amb més trials que punts, Optuna en torna a fer algun.
    """

    def after_trial(self, study, trial, state, values):
        pass


def _name(family, seed):
    """Torna el nom del study d'una família en una seed."""
    return f"{family}_seed_{seed}"


def _storage(store):
    """Torna l'adreça d'Optuna del fitxer d'un study, `store`."""
    # El camí amb `/`, que és el que la URL d'SQLite admet també a Windows.
    return f"sqlite:///{store.as_posix()}"


def opened(store, family, searcher, seed, columns):
    """Torna el study d'una família en una seed, i el crea si encara no hi és.

    `searcher` és el `search.py` del model de la família, i `columns` les columnes de la
    matriu sobre la qual es cerca. Refusa un study fet sobre unes altres files o amb una
    altra `PROJECT_SEED`.
    """
    study = optuna.create_study(
        study_name=_name(family, seed),
        direction="minimize",
        storage=_storage(store),
        load_if_exists=True,
    )
    now = made_on(columns)
    if not study.trials:
        study.set_user_attr("made_on", now)
        study.set_user_attr("PROJECT_SEED", PROJECT_SEED)
    why = _different(study.user_attrs, now)
    if why:
        raise Refusal(f"el study es va fer {why}; esborrar {paths.relative(store)}")
    study.sampler = _sampler(searcher, family, seed, study)
    return study


def asked(study, searcher, family):
    """Demana un trial al study i el torna amb els valors que el cercador en treu."""
    optuna_trial = study.ask()
    return optuna_trial, searcher.drawn(family, optuna_trial)


def tried(searcher, family, drawn, X, y, parts, seed):
    """Ajusta els valors d'un trial, `drawn`, i en torna els números i la puntuació.

    S'ajusta amb les files d'entrenament de `parts` i es puntua sobre les de validació,
    i la matriu no n'hi dona cap altra. Els números són els valors en la forma en què
    els pren l'ajust.
    """
    train, validate = X.index.isin(parts.training), X.index.isin(parts.validation)
    return searcher.trial(
        family, drawn, (X[train], y[train]), (X[validate], y[validate]), seed
    )


def told(study, optuna_trial, numbers, score, fingerprints):
    """Desa al study el resultat d'un trial i el dona per acabat.

    Hi desa els números, la càrrega de la màquina i les empremtes, `fingerprints`, del
    codi, del dataset i de les biblioteques que el fan.
    """
    optuna_trial.set_user_attr("numbers", plain(numbers))
    # La càrrega quan el trial torna; Windows no la dona, i hi queda buida.
    optuna_trial.set_user_attr(
        "load", os.getloadavg() if hasattr(os, "getloadavg") else None
    )
    for name, value in fingerprints.items():
        optuna_trial.set_user_attr(name, value)
    study.tell(optuna_trial, score)


def abandoned(study, optuna_trial):
    """Dona per fallat al study un trial que no ha tornat cap resultat."""
    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)


def record(study, family, seed, smoke):
    """Torna el registre d'una cerca acabada, amb tots els seus trials, del millor."""
    done = sorted(completed(study), key=lambda one: one.value)
    return {
        "family": family,
        "smoke": smoke,
        "seed": seed,
        "trials": len(done),
        "made_on": study.user_attrs["made_on"],
        "PROJECT_SEED": study.user_attrs["PROJECT_SEED"],
        "ranked": [
            {
                "numbers": one.user_attrs["numbers"],
                "score": one.value,
                "trial": one.number,
                "code": one.user_attrs["code"],
                "dataset": one.user_attrs["dataset"],
                "environment": one.user_attrs["environment"],
            }
            for one in done
        ],
    }


def save(found, smoke):
    """Desa el registre d'una cerca acabada, `found`, sencer o gens."""
    write_record(
        folder(smoke, found["family"]) / RECORD.format(seed=found["seed"]), found
    )


def searched(smoke, family, seed, columns):
    """Torna el registre de la cerca d'una família en una seed, o `None` si no en té.

    Refusa un registre fet sobre unes altres columnes, `columns`, o una altra partició,
    o amb una altra `PROJECT_SEED`: els seus números s'haurien triat per unes altres
    files o amb uns altres trials, i res no ho diria.
    """
    path = folder(smoke, family) / RECORD.format(seed=seed)
    if not path.exists():
        return None
    found = json.loads(path.read_text(encoding="utf-8"))
    why = _different(found, made_on(columns))
    if why:
        study = path.with_name(STUDY.format(seed=seed))
        raise Refusal(
            f"la cerca de {family} seed {seed} es va fer {why}; "
            f"esborrar {paths.relative(path)} i {paths.relative(study)}, cercar amb "
            f'"uv run epf search"'
        )
    return found


def configurations(smoke, plan, columns, seeds, best):
    """Torna les configuracions que s'ajusten a cada seed, de la millor a la pitjor.

    Cada una porta el seu lloc a la cerca. `plan` és l'entrada de la família a
    `FAMILIES` del seu model: les `kept` millors de la cerca de la família `searched_as`
    del mateix tipus, o les `best` millors si `kept` és `"BEST"`, si totes les seeds en
    tenen; els números de lloc, `placeholder`, marcats, si cap seed no en té; i una sola
    sense números per a una família que no es cerca, `kept` zero. `columns` són les de
    la matriu sobre la qual es cerca, que la cerca compara amb les seves, i `best` el
    `BEST` de la comanda. Refusa una cerca que només té algunes seeds: la dispersió
    barrejaria seeds cercades i sense cercar, i res no ho diria.
    """
    if not plan["kept"]:
        return {seed: [{"rank": 0}] for seed in seeds}
    kept = best if plan["kept"] == "BEST" else plan["kept"]
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
            f"la seed {missing[0]}"
            if len(missing) == 1
            else f"les seeds {', '.join(map(str, missing))}"
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
            for rank, one in enumerate(record["ranked"][:kept])
        ]
        for seed, record in found.items()
    }
