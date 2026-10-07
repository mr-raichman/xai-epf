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
import collections
import datetime
import functools
import importlib
import shutil
import subprocess
import sys
import time
import traceback

from epf import matrix, parameters, paths
from epf.data.outliers import report as neighbour_report
from epf.data.price import NAME, load
from epf.evaluation import metrics, walk_forward
from epf.models import attribution
from epf.models.boosting import boosting
from epf.models.lear import lear
from epf.models.mlp import mlp
from epf.models.naive import naive
from epf.results import figures, run_files
from epf.results.combination import add_combination, comparison

DAILY, HOURLY = "daily", "hourly"

# Els dos nivells, que `parameters.py` anomena i als quals fixa el període.
MEASUREMENT, TEST = parameters.MEASUREMENT, parameters.TEST

# Les tres funcions d'un pronòstic, cada una amb una feina. `attribute` és `None` on
# no hi ha cap mètode: les prediccions ingènues perquè la seva atribució seria la
# columna que copien, i l'MLP perquè TreeSHAP no s'hi aplica i cap mètode exacte és
# assumible dia a dia; si mai és la família que s'interpreta, la importància per
# permutació, que no mira dins del model, hi arriba sense desar res a cada execució.
Forecast = collections.namedtuple("Forecast", "fit predict attribute")

# `fits` diu si la família ajusta un model. Les ingènues no: copien una columna
# que ja es coneixia, i per això no entren a la combinació. No se'n deriva de `fit`,
# que hi és a totes, ni d'`attribute`, que l'MLP no té tot i ajustar.
Family = collections.namedtuple("Family", "matrix forecasts fits")


def families():
    """Cada família: quina matriu fa servir i, per pronòstic, les seves tres funcions.

    Cada model porta els seus números al seu propi
    `models/<nom>/<nom>_hyperparameters.py` i els pren per defecte, de manera que
    aquí no se'n lliga cap i el camí endavant tampoc no en sap res.
    """
    return {
        "naive": Family(DAILY, {
            "previous_day": Forecast(naive.fit, naive.previous_day, None),
            "previous_week": Forecast(naive.fit, naive.previous_week, None),
            "similar_day": Forecast(naive.fit, naive.similar_day, None),
        }, fits=False),
        "lear": Family(DAILY, {
            "lear": Forecast(lear.fit, lear.predict, lear.attribute)}, fits=True),
        "lightgbm": Family(HOURLY, {
            "lightgbm": Forecast(boosting.fit, boosting.predict, boosting.attribute)},
            fits=True),
        "mlp": Family(DAILY, {"mlp": Forecast(mlp.fit, mlp.predict, None)}, fits=True),
    }


def hyperparameters(family):
    """Els números del model d'una família, llegits del seu propi fitxer.

    Tot el que hi és en majúscules, que és la forma que tenen; així `run.json` els
    recull sense que ningú n'hagi de mantenir una llista.
    """
    try:
        module = importlib.import_module(f"epf.models.{family}.{family}_hyperparameters")
    except ModuleNotFoundError:
        return {}
    return {name: getattr(module, name) for name in dir(module) if name.isupper()}


def run(family, matrices, tier, report):
    """Camí endavant d'una família: els seus pronòstics i la seva atribució."""
    X, y = matrices[family.matrix]
    forecasts, attributions = {}, {}
    for name, of_day in family.forecasts.items():
        kept = []

        # Un ajust per dia, i tant la predicció com l'atribució surten d'aquell ajust
        # i no d'un de nou. És el que fa que l'explicació sigui de la predicció que
        # va al CSV al seu costat.
        def predict(X, y, when):
            model = of_day.fit(X, y, when)
            if of_day.attribute is not None:
                kept.append(of_day.attribute(model, X, when))
            return of_day.predict(model, X, when)

        first, last, every = parameters.PERIODS[tier]
        forecasts[name] = walk_forward(predict, X, y, first, last, every=every,
                                       on_day=functools.partial(report, name))
        if kept:
            attributions[name] = kept
    return forecasts, attributions


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


def described(commit, tier, family, chosen, fits, forecasts, walk_seconds):
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
        "family": family,
        # Quines famílies ha triat la invocació, i no només aquesta. L'últim bit d'una
        # mètrica es mou amb la selecció, de manera que el commit tot sol no diu quina
        # crida va donar aquests números.
        "chosen": sorted(chosen),
        # Si la família ajusta un model o si copia una columna ja coneguda: és el
        # que decideix si entra a la combinació, i va al disc perquè una
        # reconstrucció la pugui refer sense tenir els paràmetres a mà.
        "fits": fits,
        "forecasts": sorted(forecasts),
        "days": len(next(iter(forecasts.values()))),
        "walk_seconds": round(walk_seconds, 1),
        "parameters": _plain(swept),
        "hyperparameters": _plain(hyperparameters(family)),
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


def write(folder, forecasts, attributions, actual, record):
    """Desa tot el que ha deixat una família, un cop ja ha acabat.

    Substitueix el directori sencer, i només ara: si la família hagués fallat, el que
    hi havia abans encara hi seria.

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
    for name, kept in attributions.items():
        attribution.save(folder, name, forecasts[name].index, kept)
        worst = attribution.refuse_inconsistent(folder, name, forecasts[name])
        if worst is not None:
            print(f"  atribució de {name}: additivitat a {worst:.1e} EUR/MWh",
                  flush=True)
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

    matrices = {DAILY: data.daily, HOURLY: data.hourly}
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
            forecasts, attributions = run(table[family], matrices, args.tier, report)
            walked = time.perf_counter() - clock
            folder = folders[family]
            measured = write(folder, forecasts, attributions, actual,
                             described(commit, args.tier, family, chosen,
                                       table[family].fits, forecasts, walked))
            figures.draw(folder, actual)
        except Exception:
            traceback.print_exc()
            failed.append(family)
            # Res a mitges: si ha petat després de començar a escriure, el
            # directori se'n va sencer.
            shutil.rmtree(folders[family], ignore_errors=True)
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

    if failed:
        print(f"ha fallat i no ha desat res: {', '.join(failed)}")
        # Les que han anat bé ja estan desades: el codi de sortida només hi és
        # perquè una execució que ha perdut alguna cosa no sembli bona a qui la
        # miri des de fora, que no veu el missatge de dalt.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
