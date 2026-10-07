"""Camina el període endavant amb totes les famílies i desa el que n'ha sortit.

Què s'executa surt de dos llocs: `epf/parameters.py`, que porta el període i el
mostreig de cada nivell, i `models/<nom>/<nom>_hyperparameters.py`, que porta els
números de cada model. Per defecte hi passen totes les famílies i `--family` ho
restringeix.

Una família que falli es salta i s'informa al final, i el seu directori només
apareix quan tot el seu camí endavant ha anat bé, de manera que un fitxer de
prediccions curt no es pugui llegir mai com una execució sencera.

El directori es diu pel commit, que és el que fixa tots els números de dins, així
que una mesura no arrenca amb l'arbre brut: el nom mentiria.
"""
import argparse
import datetime
import functools
import shutil
import subprocess
import sys
import time
import traceback

from epf import matrix, parameters, paths
from epf.attribution import fitted
from epf.data.outliers import report as neighbour_report
from epf.data.price import NAME, load
from epf.evaluation import metrics
from epf.models.families import families, hyperparameters, matrices
from epf.models.naive import naive
from epf.results import figures, run_files
from epf.results.combination import add_combination, comparison
from epf.walk import forward

# Els dos nivells, que `parameters.py` anomena i als quals fixa el període.
MEASUREMENT, TEST = parameters.MEASUREMENT, parameters.TEST


def run(family, name, tables, tier, store, report):
    """Camí endavant d'una família: les prediccions de tots els seus pronòstics.

    Un ajust per dia i tots els pronòstics d'aquell ajust, de manera que les cinc
    finestres de LEAR i els tres pronòstics ingenus no caminen el període una vegada
    cada un.

    L'ajust es desa a `store` abans de predir-hi, de manera que el que queda al disc és
    el model que ha fet la predicció i no un de nou; l'atribució en surt després, a
    `epf-attribute`. Només es desa si la família té manera d'atribuir-se: un ajust que
    ningú no explicarà és disc per res.
    """
    X, y = tables[family.matrix]

    def predict(X, y, when):
        model = family.fit(X, y, when)
        if family.attribute is not None:
            fitted.save(store, when, model)
        return family.predict(model, X, when)

    first, last, every = parameters.PERIODS[tier]
    return forward(predict, X, y, first, last, every=every,
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


def described(commit, tier, name, chosen, family, forecasts, walk_seconds):
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
        "walk_seconds": round(walk_seconds, 1),
        "parameters": _plain(swept),
        "hyperparameters": _plain(hyperparameters(name)),
    }


def occupied(folders):
    """Els directoris de la llista que ja tenen res a dins, amb quants fitxers tenen."""
    return [(folder, len(list(folder.iterdir()))) for folder in folders
            if folder.exists() and any(folder.iterdir())]


def confirm(taken):
    """Pregunta abans de substituir res, anomenant cada directori i què hi ha a dins.

    Es pregunta un sol cop i abans de començar res, de manera que dir que no deixi el
    disc exactament com estava. Un comptador no n'hi ha prou: reconèixer el que no
    s'havia de tocar és tota la gràcia d'anomenar-ho.
    """
    if len(taken) == 1:
        print("1 directori d'aquesta configuració ja hi és i se substituirà sencer:")
    else:
        print(f"{len(taken)} directoris d'aquesta configuració ja hi són i se "
              "substituiran sencers:")
    for folder, files in taken:
        print(f"  {folder}  ({files} fitxers)")
    return input("escriu «si» per substituir-los: ").strip().lower() in ("si", "sí")


def write(folder, forecasts, actual, record):
    """Desa tot el que ha deixat una família, un cop ja ha acabat.

    Substitueix el directori sencer, i només ara: si la família hagués fallat, el que
    hi havia abans encara hi seria. L'atribució no hi és: la deixa `epf-attribute`
    llegint els ajustos desats.

    Les mètriques es calculen abans de tocar el disc, perquè una que peti no deixi un
    directori amb prediccions i sense `metrics.csv`, que des de fora sembla una
    execució sencera. Torna la taula que ha desat, que és la que s'imprimeix.
    """
    table = metrics(forecasts, actual, naive.weekly_frame(actual))
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    run_files.save_forecasts(folder, forecasts)
    run_files.save_metrics(folder, table)
    run_files.save_record(folder, record)
    return table


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=(TEST, MEASUREMENT), required=True,
                        help="de prova, que no dóna cap xifra, o de mesura, que sí")
    parser.add_argument("--price-only", action="store_true", required=True,
                        help="conjunt d'informació de només preu, l'únic que hi ha")
    parser.add_argument("--family", action="append",
                        help="només aquesta família; es pot repetir")
    parser.add_argument("--replace", action="store_true",
                        help="substitueix el que ja hi hagi sense preguntar")
    return parser.parse_args()


def main():
    args = parse()
    commit = _git("rev-parse", "HEAD")
    if args.tier == MEASUREMENT and _git("status", "--porcelain"):
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
    chosen = args.family or list(table)

    # Tots els directoris alhora, perquè la pregunta es faci un sol cop i abans de
    # començar res.
    where = run_files.run_folder(args.tier, parameters.PRICE_ONLY, commit[:7])
    folders = {family: where / family for family in chosen}
    taken = occupied(folders.values())
    if taken and not args.replace and not confirm(taken):
        print("no es toca res")
        return 0

    started = {}

    def report(name, predicted_day):
        done = started[name] = started.get(name, 0) + 1
        print(f"  {name} {predicted_day.date()}  {done} dies", flush=True)

    first, last, every = parameters.PERIODS[args.tier]
    failed = []
    for family in chosen:
        print(f"{family}: de {first} a {last}, un dia de cada {every}", flush=True)
        clock = time.perf_counter()
        started.clear()
        try:
            # Un sol directori d'ajustos per família, substituït sencer: el que hi
            # queda és sempre de l'última execució d'aquest nivell i etapa. Només el
            # tenen les famílies que s'atribueixen; les altres no desen cap ajust i no
            # han de deixar un directori buit amb una marca.
            explains = table[family].attribute is not None
            store = fitted.folder(args.tier, parameters.PRICE_ONLY, family)
            if explains:
                fitted.clear(store)
            forecasts = run(table[family], family, tables, args.tier, store, report)
            if explains:
                fitted.save_stamp(store, fitted.describe(
                    commit, args.tier, parameters.PRICE_ONLY, family,
                    next(iter(forecasts.values())).index))
            walked = time.perf_counter() - clock
            folder = folders[family]
            measured = write(folder, forecasts, actual,
                             described(commit, args.tier, family, chosen,
                                       table[family], forecasts, walked))
            figures.draw(folder, actual)
        except Exception:
            traceback.print_exc()
            failed.append(family)
            # Res a mitges: si ha petat després de començar a escriure, el
            # directori se'n va sencer, i amb ell els ajustos a mig desar. Sense la
            # marca ja no es podrien llegir, però són desenes de MB de res.
            shutil.rmtree(folders[family], ignore_errors=True)
            shutil.rmtree(fitted.folder(args.tier, parameters.PRICE_ONLY, family),
                          ignore_errors=True)
            print(f"{family}: ha fallat, no se'n desa res", flush=True)
        else:
            print(f"{family}: {time.perf_counter() - clock:.0f} s en total, "
                  f"{walked:.0f} s de camí endavant, desat a {folder}", flush=True)
            print(run_files.summary(measured), flush=True)

    # Les figures que comparen famílies es dibuixen amb les que hi ha al disc i no
    # amb les d'aquesta invocació, de manera que tornar a executar-ne una de sola
    # refresca la comparació amb les altres que ja hi eren.
    done = sorted(path for path in where.iterdir() if path.is_dir()) if where.exists() else []
    if done:
        counted = add_combination(done, actual)
        figures.shared(where, comparison(done, actual), actual)
        print(f"figures de {run_files.drawn(len(done))} a {where}"
              + (f", i la combinació de {counted} membres ajustats" if counted
                 else ", sense combinació"), flush=True)
        # L'atribució és una segona passada i el directori encara no en porta cap.
        print(f"per explicar-la: .venv/bin/epf-attribute {where}", flush=True)

    if failed:
        print(f"ha fallat i no ha desat res: {', '.join(failed)}")
        # Les que han anat bé ja estan desades: el codi de sortida només hi és
        # perquè una execució que ha perdut alguna cosa no sembli bona a qui la
        # miri des de fora, que no veu el missatge de dalt.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
