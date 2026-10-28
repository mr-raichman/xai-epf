"""El que comparteixen les cerques de tots els models.

Optuna desa cada trial en un study SQLite, de manera que una cerca interrompuda continua
on era. Només el procés pare hi escriu: en demana cada trial i en desa el resultat, i
un altre procés l'ajusta. Quan una seed té tots els seus trials, el seu registre els
desa tots, del millor al pitjor, en la forma en què els pren l'ajust, sobre quines
files es van fer, l'information set, les columnes de la matriu, la partició i els seus
dies, amb quina `PROJECT_SEED`, i sobre quin espai i amb quins números fixos. Una cerca
continua sobre un altre codi o unes altres biblioteques, i cada trial en desa les
empremtes, però no sobre unes altres matrius, unes altres files, un altre espai o uns
altres números fixos, ni amb una altra `PROJECT_SEED`. Cada information set té les
seves cerques.
"""

import hashlib
import json
import math
import os

import optuna

from epf import Refusal, paths
from epf.paths import plain, write_record
from epf.contract import matrix
from epf.information import recorded
from epf.parameters import DEVELOPMENT, PROJECT_SEED, SHARES, seeded

# Els fitxers d'una cerca, per seed: el study d'Optuna i el registre del que ha triat.
STUDY = "seed_{seed}.db"
RECORD = "seed_{seed}.json"


def folder(smoke, set_name, family):
    """Torna la carpeta de la cerca d'una família en un information set.

    La d'assaig si `smoke`.
    """
    return paths.searches(smoke, set_name) / family


def _dealt(seed, built):
    """Torna la SHA-256 dels dies que una seed reparteix a cada part, en ISO.

    Les parts són l'entrenament, la validació i la prova de la partició de les matrius
    `built`.
    """
    parts = matrix.partition(built, seed)
    text = json.dumps([days.strftime("%Y-%m-%d").tolist() for days in parts])
    return hashlib.sha256(text.encode()).hexdigest()


def made_on(columns, built, seed):
    """Torna les files de la cerca d'una seed.

    Són l'information set de les matrius `built`, les columnes de la matriu, `columns`,
    la partició, sense els seus dies suprimits, i els dies que la seed hi reparteix.
    """
    return plain(
        {
            "information": recorded(built.information),
            "columns": list(columns),
            "period": DEVELOPMENT,
            "deleted": list(built.deleted),
            "shares": SHARES,
            "gap": built.information.gap,
            # Els dies mateixos: una altra derivació, o una altra versió de NumPy, en
            # repartiria uns altres amb la mateixa seed.
            "split": _dealt(seed, built),
        }
    )


def completed(study):
    return [
        one for one in study.trials if one.state == optuna.trial.TrialState.COMPLETE
    ]


def diverged(study):
    """Torna els trials d'un study que han divergit, sense cap puntuació finita."""
    return [one for one in study.trials if one.user_attrs.get("diverged")]


def made_otherwise(study, columns, built, seed, searcher, matrices):
    """Torna en què es va fer diferent el study d'una seed, o `None` si no ho va fer.

    `columns` són les columnes de la matriu sobre la qual es cerca, `built` les matrius
    d'ara, `searcher` el `search.py` d'ara, amb l'espai i els números fixos, i
    `matrices` l'empremta de les matrius d'ara. Diferent vol dir sobre unes altres
    files, un altre espai o uns altres números fixos, amb una altra `PROJECT_SEED` o amb
    algun trial d'unes altres matrius: els seus trials no es podrien comparar amb els
    d'ara. Un study que no diu la seva `PROJECT_SEED`, el seu espai o els seus números
    fixos es va fer amb uns altres.
    """
    made = study.user_attrs
    if made.get("made_on") != made_on(columns, built, seed):
        return "sobre altres files"
    if made.get("space") != plain(searcher.SPACE):
        return "sobre un altre espai"
    if made.get("fixed") != plain(searcher.FIXED):
        return "amb uns altres números fixos"
    if made.get("PROJECT_SEED") != PROJECT_SEED:
        return "amb una altra PROJECT_SEED"
    found = {
        one.user_attrs.get("matrices") for one in completed(study) + diverged(study)
    }
    if found - {matrices}:
        return "sobre unes altres matrius"
    return None


def trials(searcher, cap):
    """Torna els trials de la cerca d'un cercador a cada seed.

    Són tots els punts de la seva graella, `GRID`, o, sense, els seus `TRIALS`; i com
    a molt `cap`, si no és `None`.
    """
    if searcher.GRID is None:
        own = searcher.TRIALS
    else:
        own = math.prod(len(values) for values in searcher.GRID.values())
    return own if cap is None else min(own, cap)


def _sampler(searcher, family, seed, study):
    """Torna el sampler d'un study: la graella del cercador, `GRID`, o TPE a l'espai.

    Una graella dona a cada trial, a l'atzar amb una seed de la família i de la
    partició, un punt que cap altre trial no ha començat, de manera que cap punt no es
    repeteix. TPE pren una seed que compta també els trials que el study ja porta,
    perquè una cerca continuada no torni a proposar els primers. El nom de la família
    hi entra perquè dues famílies del mateix espai no proposin les mateixes.
    """
    if searcher.GRID is not None:
        return _Grid(
            {name: list(values) for name, values in searcher.GRID.items()},
            seed=seeded("sampler", family, seed),
        )
    return optuna.samplers.TPESampler(
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


def stored(store, family, seed):
    """Torna el study d'una família en una seed que hi ha a `store`, o `None`.

    No el crea mai: sense el fitxer, torna `None`.
    """
    if not store.exists():
        return None
    return optuna.load_study(study_name=_name(family, seed), storage=_storage(store))


def opened(store, family, searcher, seed, columns, built, matrices):
    """Torna el study d'una família en una seed, i el crea si encara no hi és.

    `searcher` és el `search.py` del model de la família, `columns` les columnes de la
    matriu sobre la qual es cerca, `built` les matrius d'ara i `matrices` la seva
    empremta. Refusa un study fet d'una altra manera, com diu `made_otherwise`.
    """
    study = optuna.create_study(
        study_name=_name(family, seed),
        direction="minimize",
        storage=_storage(store),
        load_if_exists=True,
    )
    if not study.trials:
        study.set_user_attr("made_on", made_on(columns, built, seed))
        study.set_user_attr("space", plain(searcher.SPACE))
        study.set_user_attr("fixed", plain(searcher.FIXED))
        study.set_user_attr("PROJECT_SEED", PROJECT_SEED)
    why = made_otherwise(study, columns, built, seed, searcher, matrices)
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
    els pren l'ajust. Torna també on s'ha aturat l'ajust, el que en desa el cercador.
    """
    train, validate = X.index.isin(parts.training), X.index.isin(parts.validation)
    return searcher.trial(
        family, drawn, (X[train], y[train]), (X[validate], y[validate]), seed
    )


def told(study, optuna_trial, numbers, score, stop, fingerprints):
    """Desa al study el resultat d'un trial i el dona per acabat.

    Hi desa els números, on s'ha aturat l'ajust, `stop`, cada valor pel seu nom, la
    càrrega de la màquina i les empremtes, `fingerprints`, del codi, del dataset, de les
    matrius i de les biblioteques que el fan. Un trial sense una puntuació finita ha
    divergit: el dona per fallat, marcat, i diu si ho ha fet.
    """
    optuna_trial.set_user_attr("numbers", plain(numbers))
    for name, value in stop.items():
        optuna_trial.set_user_attr(name, plain(value))
    # La càrrega quan el trial torna; Windows no la dona, i hi queda buida.
    optuna_trial.set_user_attr(
        "load", os.getloadavg() if hasattr(os, "getloadavg") else None
    )
    for name, value in fingerprints.items():
        optuna_trial.set_user_attr(name, value)
    if math.isfinite(score):
        study.tell(optuna_trial, score)
        return False
    # Optuna també falla un trial que rep un NaN, però n'avisa a sobre del bloc i el
    # deixa igual que un d'abandonat.
    optuna_trial.set_user_attr("diverged", True)
    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
    return True


def abandoned(study, optuna_trial):
    """Dona per fallat al study un trial que no ha tornat cap resultat."""
    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)


def record(study, family, seed, smoke):
    """Torna el registre d'una cerca acabada, amb tots els seus trials, del millor.

    Els seus `trials` compten també els que han divergit, com la cerca.
    """
    done = sorted(completed(study), key=lambda one: one.value)
    return {
        "family": family,
        "smoke": smoke,
        "seed": seed,
        "trials": len(done) + len(diverged(study)),
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
                "matrices": one.user_attrs["matrices"],
                "environment": one.user_attrs["environment"],
            }
            for one in done
        ],
    }


def save(found, smoke, set_name):
    """Desa el registre d'una cerca acabada, `found`, sencer o gens."""
    write_record(
        folder(smoke, set_name, found["family"]) / RECORD.format(seed=found["seed"]),
        found,
    )


def configurations(found, kept):
    """Torna les configuracions que s'ajusten a una seed, de la millor a la pitjor.

    Cada una porta el seu lloc entre les que s'ajusten. Són les `kept` millors
    diferents del registre de la cerca de la seed, `found`, o, amb `kept` zero, una
    sola sense números, la d'una família que no es cerca.
    """
    if not kept:
        return [{"rank": 0}]
    chosen = []
    for one in found["ranked"]:
        # Un altre punt que puntua fins a l'últim dígit com un de guardat n'és el mateix
        # model, com 900 fulles que les files per fulla deixen en 200.
        if len(chosen) < kept and all(
            one["score"] != other["score"] for other in chosen
        ):
            chosen.append(one)
    return [
        {
            "rank": rank,
            "numbers": one["numbers"],
            "score": one["score"],
            "trial": one["trial"],
            "trials": found["trials"],
        }
        for rank, one in enumerate(chosen)
    ]
