"""Executa l'etapa E1 i desa el que n'ha sortit.

Què s'executa surt de `epf/config/experiments.py`: cada configuració hi porta el
seu període i tots els seus números. Per defecte hi passen totes les
configuracions i totes les famílies, i `--config` i `--family` ho restringeixen.

Una família que falli es salta i s'informa al final, i el seu directori només
apareix quan tot el seu camí endavant ha anat bé, de manera que un fitxer de
prediccions curt no es pugui llegir mai com una execució sencera.

El directori d'una execució de mesura es diu pel commit, així que una mesura no
arrenca amb l'arbre brut: el nom mentiria.
"""
import argparse
import collections
import datetime
import functools
import shutil
import subprocess
import sys
import time
import traceback

from epf import run as run_files
from epf.config import experiments, paths
from epf.e1 import features, figures
from epf.evaluation import metrics, walk_forward
from epf.models import attribution, boosting, lear, mlp, naive

DAILY, HOURLY = "daily", "hourly"

# Els dos nivells els anomena la configuració, que és qui en fixa els períodes.
MEASUREMENT, TEST = experiments.MEASUREMENT, experiments.TEST

Family = collections.namedtuple("Family", "matrix forecasts")

# Els tres pronòstics ingenus són una sola família de la graella i van junts en un
# sol directori.
NAIVE = {"previous_day": naive.previous_day,
         "previous_week": naive.previous_week,
         "similar_day": naive.similar_day}


def _unattributed(forecast):
    """Adapta un pronòstic que no desa atribució a la forma que espera el camí
    endavant.

    Les prediccions ingènues no n'hi desen perquè la predicció és una columna
    copiada i la seva atribució seria aquella mateixa columna. L'MLP no n'hi desa
    perquè TreeSHAP no s'hi aplica i cap mètode exacte és assumible dia a dia; si
    mai és la família que s'interpreta, la importància per permutació, que no mira
    dins del model, hi arriba sense desar res a cada execució."""
    def day(X, y, when):
        return forecast(X, y, when), None
    return day


def families(settings):
    """Cada família: quina matriu fa servir i, per pronòstic, què fa amb un dia.

    `settings` és una configuració de `experiments.py`, i cada família en pren els
    seus números aquí, de manera que el camí endavant no en sap res.

    La funció d'un dia torna els 24 preus i l'atribució del model que els ha predit.
    """
    light = functools.partial(boosting.lightgbm, trees=settings["trees"],
                              rate=settings["rate"], leaves=settings["leaves"],
                              seed=settings["seed"])
    return {
        "naive": Family(DAILY, {name: _unattributed(forecast)
                                for name, forecast in NAIVE.items()}),
        "lear": Family(DAILY, {"lear": functools.partial(
            lear.explain, windows=settings["lear_windows"],
            max_iter=settings["lear_max_iter"])}),
        "lightgbm": Family(HOURLY, {"lightgbm": functools.partial(
            boosting.explain, light, window=settings["window"])}),
        "mlp": Family(DAILY, {"mlp": _unattributed(functools.partial(
            mlp.forecast, window=settings["mlp_window"],
            hidden=settings["mlp_hidden"], members=settings["mlp_members"],
            seed=settings["seed"], alpha=settings["mlp_alpha"],
            learning_rate=settings["mlp_learning_rate"],
            validation_fraction=settings["mlp_validation_fraction"],
            no_change=settings["mlp_no_change"],
            max_iter=settings["mlp_max_iter"]))}),
    }


def run(family, matrices, settings, tier, report):
    """Camí endavant d'una família: els seus pronòstics i la seva atribució."""
    X, y = matrices[family.matrix]
    forecasts, attributions = {}, {}
    for name, of_day in family.forecasts.items():
        kept = []

        # El camí endavant torna un valor per dia, i l'atribució surt del mateix
        # ajust que la predicció, així que es recull aquí al costat.
        def predict(X, y, when):
            prices, found = of_day(X, y, when)
            if found is not None:
                kept.append(found)
            return prices

        first, last, every = settings["periods"][tier]
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
    if isinstance(value, datetime.date):
        return str(value)
    return value


def described(commit, tier, name, settings, family, forecasts, walk_seconds):
    """El que `run.json` ha de dir perquè una xifra es pugui tornar al codi que la va fer.

    La configuració sencera hi va tal com estava, de manera que el fitxer es pot
    llegir sense anar a buscar el commit, encara que el commit sigui al nom del
    directori i basti per recuperar-la.

    `walk_seconds` és el camí endavant i no la família sencera: és la part que
    creix amb el període, i per tant la que diu què costarà una passada més llarga.
    Les figures no hi són.
    """
    return {
        "commit": commit,
        "stage": features.STAGE,
        "tier": tier,
        "configuration": name,
        "family": family,
        "forecasts": sorted(forecasts),
        "days": len(next(iter(forecasts.values()))),
        "walk_seconds": round(walk_seconds, 1),
        **{key: _plain(value) for key, value in sorted(settings.items())},
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
    run_files.save_record(folder, record)
    return table


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=(TEST, MEASUREMENT), required=True,
                        help="de prova, que no dóna cap xifra, o de mesura, que sí")
    parser.add_argument("--config", action="append",
                        help="només aquesta configuració; es pot repetir")
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

    daily, hourly = features.load()
    matrices = {DAILY: daily, HOURLY: hourly}
    actual = daily[1]
    chosen = {name: experiments.configuration(name)
              for name in (args.config or list(experiments.CONFIGURATIONS))}

    # Tots els directoris de totes les configuracions triades, perquè la pregunta
    # es faci un sol cop i abans de començar res.
    folders = {(name, family): run_files.folder(args.tier, features.STAGE,
                                                f"{commit[:7]}_{name}", family)
               for name in chosen
               for family in (args.family or list(families(chosen[name])))}
    taken = occupied(folders.values())
    if taken and not args.replace and not confirm(taken):
        print("no es toca res")
        return 0

    started = {}

    def report(name, predicted_day):
        done = started[name] = started.get(name, 0) + 1
        print(f"  {name} {predicted_day.date()}  {done} dies", flush=True)

    failed = []
    for name, settings in chosen.items():
        table = families(settings)
        first, last, every = settings["periods"][args.tier]
        for family in (args.family or list(table)):
            print(f"{name}/{family}: de {first} a {last}, un dia de cada {every}",
                  flush=True)
            clock = time.perf_counter()
            started.clear()
            try:
                forecasts, attributions = run(table[family], matrices, settings,
                                              args.tier, report)
                walked = time.perf_counter() - clock
                folder = folders[(name, family)]
                measured = write(folder, forecasts, attributions, actual,
                                 described(commit, args.tier, name, settings, family,
                                           forecasts, walked))
                figures.draw(folder, actual)
            except Exception:
                traceback.print_exc()
                failed.append(f"{name}/{family}")
                # Res a mitges: si ha petat després de començar a escriure, el
                # directori se'n va sencer.
                shutil.rmtree(folders[(name, family)], ignore_errors=True)
                print(f"{name}/{family}: ha fallat, no se'n desa res", flush=True)
            else:
                print(f"{name}/{family}: {time.perf_counter() - clock:.0f} s en total, "
                      f"{walked:.0f} s de camí endavant, desat a {folder}", flush=True)
                print(run_files.summary(measured), flush=True)

        # Les figures que comparen famílies es dibuixen amb les que hi ha al disc i
        # no amb les d'aquesta invocació, de manera que tornar a executar-ne una de
        # sola refresca la comparació amb les altres que ja hi eren.
        where = run_files.configuration_folder(args.tier, features.STAGE,
                                               f"{commit[:7]}_{name}")
        done = sorted(path for path in where.iterdir() if path.is_dir()) if where.exists() else []
        if done:
            figures.compare(where, done, actual)
            print(f"{name}: comparació de {len(done)} famílies a {where}", flush=True)

    if failed:
        print(f"ha fallat i no ha desat res: {', '.join(failed)}")
        # Les que han anat bé ja estan desades: el codi de sortida només hi és
        # perquè una execució que ha perdut alguna cosa no sembli bona a qui la
        # miri des de fora, que no veu el missatge de dalt.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
