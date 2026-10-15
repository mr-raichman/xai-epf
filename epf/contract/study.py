"""El que comparteixen les cerques de tots els models: un estudi d'Optuna per llavor i
família, el registre de les seves millors configuracions, i quines configuracions ajusta
una passada a cada llavor.

Optuna desa cada prova en un estudi SQLite, de manera que una cerca interrompuda es
reprèn on era. Quan una llavor té totes les seves proves, el seu registre en desa les
millors en la forma en què les pren l'ajust, i sobre quines files es van fer: les
columnes del conjunt i la partició. Una passada no llegeix un registre fet sobre unes
altres files, i una cerca no es reprèn sobre unes altres files.
"""
import json
import os
import time

import numpy as np
import optuna

from epf import paths
from epf.contract.features import SCHEDULE
from epf.contract.record import plain
from epf.parameters import BEST, BLACKOUT_WEEK, DEVELOPMENT, GAP, SEED, SHARES

# Els fitxers d'una cerca, per llavor: l'estudi d'Optuna i el registre del que ha triat.
STUDY = "seed_{seed}.db"
RECORD = "seed_{seed}.json"


def folder(tier, family):
    """On viu la cerca d'una família en un nivell, sobre l'únic conjunt que es cerca."""
    return paths.SEARCHES / tier / SCHEDULE / family


def made_on(columns):
    """Les files sobre què es fa una cerca: les columnes del conjunt i la partició."""
    return plain({"columns": list(columns), "period": DEVELOPMENT,
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
        trial.set_user_attr("numbers", plain(numbers))
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


def configurations(tier, family, describer, columns, seeds, best_only):
    """Les configuracions que s'ajusten a cada llavor, de la millor a la pitjor, cada una
    amb el seu lloc a la cerca.

    Les de la cerca de la família al nivell, si totes les llavors en tenen, `kept` de
    cada una, o la millor sola amb `best_only`; els números de lloc, marcats, si no en té
    cap cap llavor; i una sola sense números per a una família que no es cerca. Una cerca
    que només té algunes llavors es refusa: la dispersió barrejaria llavors cercades i
    sense cercar, i res no ho diria. `columns` són les de la taula sobre la qual es cerca
    la família, que la cerca compara amb les seves.
    """
    if not describer.kept:
        return {seed: [{"rank": 0}] for seed in seeds}
    found = {seed: searched(tier, family, seed, columns) for seed in seeds}
    missing = [seed for seed, record in found.items() if record is None]
    if len(missing) == len(seeds):
        return {seed: [{"rank": 0, "numbers": describer.placeholder, "unsearched": True}]
                for seed in seeds}
    if missing:
        raise ValueError(f"la cerca de {family} no té les llavors {missing}: acaba-la amb "
                         f"epf search abans de descriure-la")
    kept = 1 if best_only else describer.kept
    return {seed: [{"rank": rank, "numbers": one["numbers"], "score": one["score"],
                    "trial": one["trial"], "trials": record["trials"]}
                   for rank, one in enumerate(record["best"][:kept])]
            for seed, record in found.items()}
