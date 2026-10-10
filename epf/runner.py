"""Camina el període endavant amb totes les famílies i desa el que n'ha sortit.

Què s'executa surt de dos llocs: `epf/parameters.py`, que porta el període i el
mostreig de cada nivell, i `models/<nom>/<nom>_hyperparameters.py`, que porta els
números de cada model. Per defecte hi passen totes les famílies i `--family` ho
restringeix.

Les prediccions s'escriuen dia a dia i `run.json` s'escriu l'últim, de manera que el
directori d'una família és el seu propi magatzem per reprendre-la: una execució de
mesura interrompuda continua d'on era i una família ja sencera se salta. Una família
que falli es salta i s'informa al final, i el seu directori es queda a mig fer, que és
el que permet reprendre-la.

El directori es diu pel commit, que és el que fixa tots els números de dins, així
que una mesura no arrenca amb l'arbre brut: el nom mentiria. És també per això que
reprendre és només del nivell de mesura, l'únic que exigeix l'arbre net.
"""
import argparse
import datetime
import functools
import shutil
import subprocess
import sys
import time
import traceback

import pandas as pd

from epf import cli, matrix, parameters, paths
from epf.attribution import fitted
from epf.data.outliers import report as neighbour_report
from epf.data.price import NAME, load
from epf.evaluation import PERCENTILES, distribution_metrics, metrics
from epf.models.families import families, hyperparameters, matrices
from epf.models.naive import naive
from epf.results import combination, figures, run_files
from epf.walk import forward

# Els dos nivells, que `parameters.py` anomena i als quals fixa el període.
MEASUREMENT, TEST = parameters.MEASUREMENT, parameters.TEST


def run(family, name, tables, tier, store, folder, after, report):
    """Camí endavant d'una família: cada dia escrit a mesura que surt.

    Un ajust per dia i tots els pronòstics d'aquell ajust, de manera que les cinc
    finestres de LEAR i els tres pronòstics ingenus no caminen el període una vegada
    cada un.

    L'ajust es desa a `store` abans de predir-hi, de manera que el que queda al disc és
    el model que ha fet la predicció i no un de nou; l'atribució en surt després, a
    `epf attribute`. Només es desa si la família té manera d'atribuir-se: un ajust que
    ningú no explicarà és disc per res. Es desa abans que la fila arribi a `report`, de
    manera que una caiguda pot deixar un ajust sense fila però mai una fila sense
    ajust: el dia que es reprèn només sobreescriu el seu.

    Torna els dies que ha caminat, que són menys que els del període quan reprèn.
    """
    X, y = tables[family.matrix]

    def predict(X, y, when):
        model = family.fit(X, y, when)
        if family.attribute is not None:
            fitted.save(store, when, model)
        if family.distribution is not None:
            # Es demana la xarxa dues vegades per dia, aquí i a `predict`: són dues
            # crides sobre una sola fila, i el preu de deixar `predict` genèric.
            run_files.append_distribution(
                folder, name, when, family.distribution.of_day(model, X, when),
                family.distribution.names)
        return family.predict(model, X, when)

    first, last, every = parameters.PERIODS[tier]
    return forward(predict, X, y, first, last, every=every, after=after,
                   on_day=functools.partial(report, name))


def _git(*arguments):
    return subprocess.run(("git", *arguments), cwd=paths.ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _plain(value):
    """El valor tal com es pot escriure a JSON, sense perdre què era."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (datetime.date, datetime.timedelta)):
        return str(value)
    return value


def described(commit, tier, name, chosen, family, forecasts, walk_seconds, walked):
    """El que `run.json` ha de dir perquè una xifra es pugui tornar al codi que la va fer.

    Els paràmetres de l'execució i els números del model hi van tal com estaven, de
    manera que el fitxer es pot llegir sense anar a buscar el commit, encara que el
    commit basti per recuperar-los.

    `walk_seconds` és el camí endavant i no la família sencera: és la part que
    creix amb el període, i per tant la que diu què costarà una passada més llarga.
    Les figures no hi són.
    """
    swept = {name: getattr(parameters, name) for name in dir(parameters)
             if name.isupper() and not name.startswith("_")}
    return {
        "commit": commit,
        "stage": parameters.PRICE_ONLY,
        "tier": tier,
        "family": name,
        # Quines famílies ha triat la invocació, i no només aquesta. L'últim bit d'una
        # mètrica es mou amb la selecció, de manera que el commit tot sol no diu quina
        # crida va donar aquests números.
        "chosen": sorted(chosen),
        # Quins dels pronòstics del directori són models ajustats, i per tant què
        # mitjana la combinació, i quins es comparen a les figures. Van al disc perquè
        # una reconstrucció els pugui refer sense el registre de famílies a mà.
        "members": sorted(family.members),
        "headlines": sorted(family.headlines),
        "forecasts": sorted(forecasts),
        "days": len(next(iter(forecasts.values()))),
        # Els dies caminats en aquesta sessió, que són menys que `days` quan l'execució
        # s'ha represa, i els segons d'aquests mateixos dies. Els de la sessió
        # anterior no es van desar enlloc, perquè `run.json` s'escriu al final i una
        # execució interrompuda no en té; amb les dues xifres, la divisió segueix
        # donant el cost per dia i cap de les dues no menteix.
        "walked_days": walked,
        "walk_seconds": round(walk_seconds, 1),
        "parameters": _plain(swept),
        "hyperparameters": _plain(hyperparameters(name)),
    }


def _distribution_rows(folder, name, family, actual):
    """Les mètriques que només tenen sentit per a una predicció distribucional.

    `None` per a una família que prediu 24 preus i prou, que són totes llevat de les
    dues distribucionals. Es llegeixen del disc i no de la memòria, com les
    prediccions, de manera que una execució represa en dóna les mateixes.
    """
    if family.distribution is None:
        return None
    days, parameters, _ = run_files.load_distribution(folder, name)
    quantiles = family.distribution.quantiles(parameters, PERCENTILES)
    return distribution_metrics(name, actual.loc[days], quantiles)


def write(folder, forecasts, actual, record, extra=None):
    """Tanca el directori d'una família: les mètriques, i el registre l'últim.

    Les prediccions ja hi són, escrites dia a dia pel camí endavant, i arriben aquí
    rellegides del disc: així una execució represa dóna els mateixos números que una de
    seguida i no hi ha dues versions dels mateixos, la de la memòria i la del disc.

    `run.json` s'escriu després de `metrics.csv` perquè és el que diu que el directori
    és sencer: si les mètriques peten, el que queda és un directori a mig fer, que
    ningú no llegeix i que es pot reprendre.

    Torna la taula que ha desat, que és la que s'imprimeix.
    """
    table = metrics(forecasts, actual, naive.weekly_frame(actual))
    if extra is not None:
        table = pd.concat([table, extra], ignore_index=True)
    run_files.save_metrics(folder, table)
    run_files.save_record(folder, record)
    return table


def parse():
    """Els arguments quan s'arrenca com a mòdul, amb `python -m epf.runner`.

    Els declara `cli.run_arguments`, de manera que `epf run` i el mòdul prenen
    exactament els mateixos i no n'hi ha dues declaracions.
    """
    return cli.run_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()


def main(arguments):
    commit = _git("rev-parse", "HEAD")
    if arguments.tier == MEASUREMENT and _git("status", "--porcelain"):
        print("l'arbre té canvis sense desar. El directori d'una mesura es diu pel "
              "commit, i amb canvis sense desar el nom no diria quin codi l'ha fet. "
              f"Fes un commit, o fes servir --tier {TEST}.")
        return 1

    # `load` ja ha passat el contracte. L'informe de veïnatge no refusa res i no és
    # cap comprovació: el llegeix Teo.
    frame = load()
    neighbour_report(NAME, frame)
    data = matrix.price_only(frame)

    tables = matrices(data)
    actual = data.actual
    table = families()
    chosen = arguments.family or list(table)

    where = run_files.run_folder(arguments.tier, parameters.PRICE_ONLY, commit[:7])
    folders = {family: where / family for family in chosen}

    # Reprendre és del nivell de mesura i prou. El de prova admet l'arbre brut i el
    # directori es diu igualment pel commit, de manera que dues còpies diferents del
    # codi en comparteixen el nom: reprendre-hi continuaria un camí fet per un altre
    # codi. Una execució de prova sempre comença de zero, i allà `--replace` és el que
    # passa i no una tria.
    resuming = arguments.tier == MEASUREMENT and not arguments.replace
    complete = {family for family in chosen if run_files.has_record(folders[family])}
    if resuming and complete == set(chosen):
        print(f"aquesta execució ja hi és sencera a {where}: no es toca res. "
              "Amb --replace es torna a començar de zero.")
        return 0

    started = {}

    def report(name, predicted_day, frames):
        run_files.append_forecasts(folders[name], frames)
        done = started[name] = started.get(name, 0) + 1
        print(f"  {name} {predicted_day.date()}  {done} dies", flush=True)

    first, last, every = parameters.PERIODS[arguments.tier]
    failed = []
    for family in chosen:
        folder = folders[family]
        explains = table[family].attribute is not None
        store = fitted.folder(arguments.tier, parameters.PRICE_ONLY, family)

        if resuming and family in complete:
            print(f"{family}: ja hi és sencera a {folder}, no es torna a caminar",
                  flush=True)
            continue

        after = run_files.last_common_day(folder) if resuming else None
        if after is None:
            # Se substitueix ara i no al final, que és el que fa que el directori es
            # pugui anar escrivint. Els ajustos se'n van amb ell: només valen per al
            # camí endavant que els va escriure.
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True)
            if explains:
                fitted.clear(store)
            print(f"{family}: de {first} a {last}, un dia de cada {every}", flush=True)
        else:
            # Els pronòstics que havien arribat més enllà del dia que tots tenen es
            # tallen, de manera que el camí continua amb tots alineats.
            run_files.drop_after(folder, after)
            print(f"{family}: es reprèn després del {after.date()}, fins a {last}",
                  flush=True)

        clock = time.perf_counter()
        started.clear()
        try:
            walked = run(table[family], family, tables, arguments.tier, store,
                         folder, after, report)
            seconds = time.perf_counter() - clock
            forecasts = run_files.load_forecasts(folder)
            if explains:
                fitted.save_stamp(store, fitted.describe(
                    commit, arguments.tier, parameters.PRICE_ONLY, family,
                    next(iter(forecasts.values())).index))
            measured = write(folder, forecasts, actual,
                             described(commit, arguments.tier, family, chosen,
                                       table[family], forecasts, seconds, len(walked)),
                             extra=_distribution_rows(folder, family, table[family],
                                                      actual))
            figures.draw(folder, actual)
        except Exception:
            traceback.print_exc()
            failed.append(family)
            # El directori es queda com està, a mig caminar i sense registre: ningú no
            # el llegeix com una execució i la pròxima crida el reprèn d'on era.
            print(f"{family}: ha fallat, es podrà reprendre de {folder}", flush=True)
        else:
            print(f"{family}: {time.perf_counter() - clock:.0f} s en total, "
                  f"{seconds:.0f} s de camí endavant de {len(walked)} dies, "
                  f"desat a {folder}", flush=True)
            print(run_files.summary(measured), flush=True)

    # Les figures que comparen famílies es dibuixen amb les que hi ha al disc i no
    # amb les d'aquesta invocació, de manera que tornar a executar-ne una de sola
    # refresca la comparació amb les altres que ja hi eren.
    done = run_files.recorded(where)
    if done:
        counted = combination.save(where, done, actual)
        forecasts = combination.comparison(run_files.recorded(where))
        figures.shared(where, forecasts, actual)
        print(f"figures de {run_files.drawn(len(forecasts))} a {where}"
              + (f", amb la combinació de {counted} membres ajustats" if counted
                 else ", sense combinació"), flush=True)
        # L'atribució és una segona passada i el directori encara no en porta cap.
        print(f"per explicar-la: .venv/bin/epf attribute {where}", flush=True)

    if failed:
        print(f"ha fallat: {', '.join(failed)}")
        # Les que han anat bé ja estan desades: el codi de sortida només hi és
        # perquè una execució que ha perdut alguna cosa no sembli bona a qui la
        # miri des de fora, que no veu el missatge de dalt.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(parse()))
