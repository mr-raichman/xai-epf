"""El que comparteixen les cerques de tots els models.

Optuna desa cada trial en un study SQLite, de manera que una cerca interrompuda continua
on era. Només el procés pare hi escriu: en demana cada trial i en desa el resultat, i
un altre procés l'ajusta. Quan una seed té tots els seus trials, el seu registre els
desa tots, del millor al pitjor, en la forma en què els pren l'ajust, sobre quines
files es van fer, les columnes de la matriu, la partició i els seus dies, amb quina
`PROJECT_SEED`, i sobre quin espai i amb quins números fixos. Una cerca continua sobre
un altre codi o unes altres biblioteques, i cada trial en desa les empremtes, però no
sobre un altre dataset, unes altres files, un altre espai o uns altres números fixos,
ni amb una altra `PROJECT_SEED`.
"""

import hashlib
import json
import os

import optuna

from epf import Refusal, paths
from epf.contract.record import plain, write_record
from epf.contract.split import split, validation_parts
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, GAP, PROJECT_SEED, SHARES, seeded

# Els fitxers d'una cerca, per seed: el study d'Optuna i el registre del que ha triat.
STUDY = "seed_{seed}.db"
RECORD = "seed_{seed}.json"


def folder(smoke, family):
    """Torna la carpeta de la cerca d'una família, la d'assaig si `smoke`."""
    return paths.SEARCHES / f"{paths.SMOKE_PREFIX if smoke else ''}{family}"


def _dealt(seed):
    """Torna la SHA-256 dels dies que una seed reparteix a cada part, en ISO.

    Les parts són l'entrenament, cada part de la validació en l'ordre de la seed, i la
    prova.
    """
    parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
    every = [
        parts.training,
        *validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK),
        parts.test,
    ]
    text = json.dumps([days.strftime("%Y-%m-%d").tolist() for days in every])
    return hashlib.sha256(text.encode()).hexdigest()


def made_on(columns, seed):
    """Torna les files de la cerca d'una seed.

    Són les columnes de la matriu, `columns`, la partició i els dies que la seed hi
    reparteix.
    """
    return plain(
        {
            "columns": list(columns),
            "period": DEVELOPMENT,
            "deleted": BLACKOUT_WEEK,
            "shares": SHARES,
            "gap": GAP,
            # Els dies mateixos: una altra derivació, o una altra versió de NumPy, en
            # repartiria uns altres amb la mateixa seed.
            "split": _dealt(seed),
        }
    )


def completed(study):
    return [
        one for one in study.trials if one.state == optuna.trial.TrialState.COMPLETE
    ]


def _different(made, now, searcher):
    """Torna en què es va fer diferent una cerca, `made`, o `None` si no ho va fer.

    `now` són les files d'ara, i `searcher` el `search.py` d'ara, amb l'espai i els
    números fixos. Una cerca que no diu la seva `PROJECT_SEED`, el seu espai o els seus
    números fixos es va fer amb uns altres.
    """
    if made.get("made_on") != now:
        return "sobre altres files"
    if made.get("space") != plain(searcher.SPACE):
        return "sobre un altre espai"
    if made.get("fixed") != plain(searcher.FIXED):
        return "amb uns altres números fixos"
    if made.get("PROJECT_SEED") != PROJECT_SEED:
        return "amb una altra PROJECT_SEED"
    return None


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
            seed=seeded("sampler", family, seed),
        )
    return optuna.samplers.RandomSampler(
        seed=seeded("sampler", family, seed, len(study.trials))
    )


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


def opened(store, family, searcher, seed, columns, dataset):
    """Torna el study d'una família en una seed, i el crea si encara no hi és.

    `searcher` és el `search.py` del model de la família, `columns` les columnes de la
    matriu sobre la qual es cerca, i `dataset` l'empremta del dataset d'ara. Refusa un
    study fet sobre unes altres files, un altre espai o uns altres números fixos, amb
    una altra `PROJECT_SEED` o amb algun trial d'un altre dataset: compararia trials
    que no es poden comparar.
    """
    study = optuna.create_study(
        study_name=_name(family, seed),
        direction="minimize",
        storage=_storage(store),
        load_if_exists=True,
    )
    now = made_on(columns, seed)
    if not study.trials:
        study.set_user_attr("made_on", now)
        study.set_user_attr("space", plain(searcher.SPACE))
        study.set_user_attr("fixed", plain(searcher.FIXED))
        study.set_user_attr("PROJECT_SEED", PROJECT_SEED)
    why = _different(study.user_attrs, now, searcher)
    datasets = {one.user_attrs.get("dataset") for one in completed(study)}
    if not why and datasets - {dataset}:
        why = "amb un altre dataset"
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
        "space": study.user_attrs["space"],
        "fixed": study.user_attrs["fixed"],
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


def configurations(found, kept):
    """Torna les configuracions que s'ajusten a una seed, de la millor a la pitjor.

    Cada una porta el seu lloc a la cerca. Són les `kept` millors del registre de la
    cerca de la seed, `found`, o, amb `kept` zero, una sola sense números, la d'una
    família que no es cerca.
    """
    if not kept:
        return [{"rank": 0}]
    return [
        {
            "rank": rank,
            "numbers": one["numbers"],
            "score": one["score"],
            "trial": one["trial"],
            "trials": found["trials"],
        }
        for rank, one in enumerate(found["ranked"][:kept])
    ]
