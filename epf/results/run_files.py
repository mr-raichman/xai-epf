"""Què hi ha dins del directori d'una execució, i on és.

Un directori per configuració i família, més el de la combinació, perquè el que es
compara és una execució sencera: totes les famílies d'una configuració, de cop. El
període és dins de `run.json` i dins del nom que se li posa a la configuració.

Tot el que hi ha aquí es deriva de les prediccions i del preu, de manera que es pot
tornar a calcular sense tornar a ajustar res. L'atribució no: aquella surt del model
que va predir el dia i la desa `attribution/stored.py`.
"""
import json

import numpy as np
import pandas as pd

from epf import paths

# El nom de l'índex de tota taula d'una execució, que és el dia de lliurament. Ve de la
# sèrie de preus i es torna a posar en llegir el que es desa sense índex de pandas.
DAY = "day"

# La columna de l'hora de les taules llargues d'una execució. La de `matrix.py` és la
# de les matrius de característiques; aquesta és la mateixa paraula perquè és la
# mateixa cosa, i no s'importa d'allà perquè `results/` no depèn de les matrius.
HOUR = "hour"

FORECASTS = "forecasts_{name}.csv"
METRICS = "metrics.csv"
RECORD = "run.json"

# Els paràmetres de la distribució que una família distribucional ha predit cada dia,
# una fila per dia i hora i una columna per paràmetre. Van a part i no a
# `forecasts_<nom>.csv` perquè no són preus: una escala comparada amb el preu real
# donaria un MAE que no vol dir res, i tot el que llegeix un pronòstic l'espera en
# EUR/MWh. El CRPS és qui els demana.
#
# CSV i no un `.npz` perquè el camí endavant es reprèn: s'hi afegeix un dia cada
# vegada, com a les prediccions, i es talla amb elles. Un `.npz` no s'hi pot afegir
# res, de manera que una execució represa n'hauria desat només els dies de l'última
# sessió i ningú no ho hauria notat.
DISTRIBUTION = "distribution_{name}.csv"

# La combinació d'una execució: el nom del pronòstic i el del seu directori, que és un
# de més dins del de la configuració. Viu aquí, i no a `combination.py`, perquè és un
# dels noms que un directori d'execució porta a dins, i perquè qui l'ha de poder
# reconèixer sense muntar-la és el que escriu i el que llegeix (`architecture.md` 8.1).
COMBINATION = "combination"

# Com es tornen a llegir els CSV. El lector per defecte de pandas pot errar l'últim
# bit d'un float, i amb això una reconstrucció donava mètriques que es movien a la 16a
# xifra significativa. Hi passen tots els números d'una execució: les mètriques es
# calculen de les prediccions rellegides del disc, de manera que una execució represa
# dóna el mateix que una de seguida, i una reconstrucció reprodueix l'execució bit a
# bit, que és el que fa que un diff entre les dues vulgui dir alguna cosa.
EXACT = "round_trip"


def run_folder(tier, stage, commit):
    """On viu una execució sencera: hi pengen les famílies i les figures que les
    comparen. El commit la nomena, perquè és el que fixa tots els seus números."""
    return paths.RUNS / tier / stage / commit


def family_folders(paths):
    """Els directoris amb registre que hi ha darrere dels camins donats.

    Un camí que ja ho és val tal qual; un que no ho és aporta els seus
    subdirectoris que sí que ho són, de manera que tant el directori d'una
    configuració com un glob que l'obre valen igual. El glob també agafa les
    figures de la configuració, que no són directoris i se salten.

    El de la combinació hi entra com els altres, que és el que fa que surti a la taula
    de `epf show-run` i es refaci com un més, i és per això que mitjanar-la és l'únic
    lloc que l'ha de distingir (`combination.combined`).
    """
    found = []
    for path in paths:
        if not path.is_dir():
            continue
        if has_record(path):
            found.append(path)
        else:
            found += recorded(path)
    if not found:
        raise ValueError(f"cap directori de família a {[str(path) for path in paths]}")
    return found


def is_combination(folder):
    """Si el directori és el de la combinació d'una configuració i no el d'una família."""
    return folder.name == COMBINATION


def has_record(folder):
    """Si el directori porta registre, que és el que diu que l'execució és sencera.

    `run.json` s'escriu l'últim, de manera que un directori a mig caminar no en té i
    tot el que llegeix una execució el salta sense haver de saber que es pot reprendre.
    """
    return (folder / RECORD).exists()


def recorded(where):
    """Els directoris amb registre que hi ha dins d'una configuració, amb la combinació."""
    return sorted(path for path in where.iterdir() if has_record(path)) \
        if where.exists() else []


def save_forecasts(folder, forecasts):
    for name, frame in sorted(forecasts.items()):
        frame.to_csv(folder / FORECASTS.format(name=name))


def load_forecasts(folder):
    """Els pronòstics que hi ha desats, pel nom que duien quan es van escriure."""
    found = sorted(folder.glob(FORECASTS.format(name="*")))
    if not found:
        raise ValueError(f"cap fitxer de prediccions a {folder}")
    prefix = FORECASTS.format(name="")[:-len(".csv")]
    return {path.stem.removeprefix(prefix): pd.read_csv(path, index_col=0,
                                                       parse_dates=True,
                                                       float_precision=EXACT)
            for path in found}


def append_forecasts(folder, frames):
    """Afegeix un dia a cada fitxer de prediccions, una taula d'una fila per pronòstic.

    El camí endavant escriu a mesura que camina, de manera que el directori d'una
    família és el seu propi magatzem per reprendre-la: `metrics.csv` i `run.json`
    arriben al final i `run.json` és el que diu que és sencera.

    Una fila escrita així surt byte a byte igual que la mateixa fila dins del `to_csv`
    de la taula sencera, que és el que fa que una execució represa deixi el mateix
    fitxer que una de seguida. Això és mesurat i no raonat: `tests/test_run_files.py`
    ho fixa.
    """
    for name, frame in sorted(frames.items()):
        path = folder / FORECASTS.format(name=name)
        starting = not path.exists()
        frame.to_csv(path, mode="w" if starting else "a", header=starting)


def last_common_day(folder):
    """L'últim dia que tenen tots els pronòstics del directori, o `None` si no n'hi ha.

    El mínim dels últims i no el màxim: si una execució va caure entre dos fitxers,
    manda el curt, perquè reprendre des del llarg alinearia un pronòstic amb els dies
    equivocats, que és el que `walk.forward` refusa dins de la memòria.
    """
    if not folder.exists() or not any(folder.glob(FORECASTS.format(name="*"))):
        return None
    forecasts = load_forecasts(folder)
    last = [frame.index[-1] for frame in forecasts.values() if not frame.empty]
    if len(last) != len(forecasts):
        return None
    return min(last)


def drop_after(folder, day):
    """Deixa tot el que el camí endavant va escrivint acabat el mateix dia.

    Els pronòstics i, on n'hi hagi, els paràmetres de la distribució: tots dos
    s'escriuen dia a dia, de manera que tots dos es poden haver passat del dia pel qual
    es reprèn.
    """
    for name, frame in load_forecasts(folder).items():
        if frame.index[-1] > day:
            frame[frame.index <= day].to_csv(folder / FORECASTS.format(name=name))
    for path in sorted(folder.glob(DISTRIBUTION.format(name="*"))):
        table = pd.read_csv(path, parse_dates=[DAY], float_precision=EXACT)
        kept = table[table[DAY] <= day]
        if len(kept) != len(table):
            kept.to_csv(path, index=False)


def append_distribution(folder, name, day, parameters, names):
    """Afegeix els paràmetres d'un dia, `(hores, paràmetres)`, una fila per hora."""
    path = folder / DISTRIBUTION.format(name=name)
    rows = pd.DataFrame(parameters, columns=list(names))
    rows.insert(0, HOUR, range(len(rows)))
    rows.insert(0, DAY, day)
    starting = not path.exists()
    rows.to_csv(path, mode="w" if starting else "a", header=starting, index=False)


def load_distribution(folder, name):
    """Els dies i els paràmetres desats, `(dies, hores, paràmetres)`, i els seus noms.

    Es tornen a muntar en el mateix ordre en què es van escriure, que és el de l'hora
    dins del dia: si una fila hagués arribat fora d'ordre, el CRPS compararia una hora
    amb el preu d'una altra.
    """
    table = pd.read_csv(folder / DISTRIBUTION.format(name=name),
                        parse_dates=[DAY], float_precision=EXACT)
    names = [column for column in table.columns if column not in (DAY, HOUR)]
    days = pd.DatetimeIndex(table[DAY].unique(), name=DAY)
    hours = table[HOUR].to_numpy().reshape(len(days), -1)
    if not (hours == np.arange(hours.shape[1])).all():
        raise ValueError(f"les hores de {DISTRIBUTION.format(name=name)} no van en "
                         "ordre dins de cada dia")
    return days, table[names].to_numpy().reshape(len(days), hours.shape[1],
                                                 len(names)), names


def save_metrics(folder, table):
    table.to_csv(folder / METRICS, index=False)


def load_metrics(folder):
    return pd.read_csv(folder / METRICS, float_precision=EXACT)


def summary(table):
    """Les mètriques d'un sol valor, un pronòstic per fila i una mètrica per columna.

    L'ordre de les columnes és el que porta la taula i no l'alfabètic, de manera
    que qui decideix com es llegeix una execució és `evaluation.metrics`, que és qui
    les calcula.
    """
    singles = table[table["hour"].isna() | (table["hour"] == "")]
    order = list(singles["metric"].drop_duplicates())
    wide = singles.pivot(index="forecast", columns="metric", values="value")
    return wide[order].to_string(float_format=lambda value: f"{value:9.3f}")


def headlines(folder, forecasts):
    """Els noms que una família compara a les figures, del seu `run.json`.

    Un registre d'abans que ho digués dóna tots els pronòstics que hi ha, que és el
    que es dibuixava aleshores.
    """
    return load_record(folder).get("headlines", sorted(forecasts))


def drawn(forecasts):
    """«1 pronòstic» o «n pronòstics», perquè el que imprimeix un script és català."""
    return "1 pronòstic" if forecasts == 1 else f"{forecasts} pronòstics"


def save_record(folder, record):
    (folder / RECORD).write_text(json.dumps(record, indent=2) + "\n")


def load_record(folder):
    return json.loads((folder / RECORD).read_text())
