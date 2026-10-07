"""Els models ajustats que una execució deixa, perquè l'atribució no els hagi de refer.

Un fitxer per dia, escrit pel camí endavant des del mateix ajust amb què va predir, de
manera que l'explicació que en surt és de la predicció que té al costat i no d'un ajust
nou. Això importa perquè un ajust no es repeteix sol: `boosting.lightgbm` ho va haver de
demanar explícitament.

El directori és `runs/<nivell>/<etapa>/fitted/<família>/` i només n'hi ha un per nivell,
etapa i família: el camí endavant l'esborra abans d'escriure-hi. L'etapa hi és perquè un
ajust de només preu no pot explicar una execució amb fonaments, i sense ella les dues
escriurien al mateix lloc amb el mateix commit.

La marca diu de quina execució són. Qui els llegeix la compara amb el directori que li
donen i refusa si no quadra, perquè atribuir amb l'ajust d'una altra execució dóna
números que semblen bons.
"""
import json
import shutil

import joblib

from epf import paths

FOLDER = "fitted"
STAMP = "fitted.json"
MODEL = "{day}.joblib"

# El mateix nivell de compressió amb què es va mesurar què costa desar-los.
COMPRESS = 3


def folder(tier, stage, family):
    """On viuen els ajustos d'una família, al costat dels directoris de commit."""
    return paths.RUNS / tier / stage / FOLDER / family


def clear(where):
    """Deixa el directori buit: només hi ha els ajustos de l'última execució."""
    if where.exists():
        shutil.rmtree(where)
    where.mkdir(parents=True)


def save(where, day, model):
    joblib.dump(model, where / MODEL.format(day=day.date()), compress=COMPRESS)


def load(where, day):
    return joblib.load(where / MODEL.format(day=day.date()))


def describe(commit, tier, stage, family, days):
    """La marca: de quina execució són aquests ajustos."""
    return {"commit": commit, "tier": tier, "stage": stage, "family": family,
            "days": [str(day.date()) for day in days]}


def save_stamp(where, record):
    (where / STAMP).write_text(json.dumps(record, indent=2) + "\n")


def load_stamp(where):
    try:
        return json.loads((where / STAMP).read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no hi ha cap ajust desat a {where}; torna a executar la família") from None


def refuse_mismatch(where, record):
    """Refusa si els ajustos desats no són de l'execució que es vol explicar.

    Els ajustos se substitueixen a cada execució del mateix nivell i etapa, així que
    els que hi ha poden ser d'un commit posterior al directori que s'explica. Amb
    l'additivitat sola no es veuria: quadraria amb l'ajust equivocat.
    """
    stamp = load_stamp(where)
    wrong = [key for key in ("commit", "tier", "stage", "family")
             if stamp.get(key) != record[key]]
    if wrong:
        raise ValueError(
            f"els ajustos de {where} són d'una altra execució: "
            + ", ".join(f"{key} {stamp.get(key)} i no {record[key]}" for key in wrong))
    return stamp
