"""Executa l'etapa E1 sobre un període i desa el que n'ha sortit.

Per defecte hi passa totes les famílies. Una que falli es salta i s'informa al
final, i el directori d'una família només apareix quan tot el seu camí endavant ha
anat bé, de manera que un fitxer de prediccions curt no es pugui llegir mai com una
execució sencera.
"""
import argparse
import collections
import datetime
import functools
import subprocess
import time
import traceback

from epf import run as run_files
from epf.config import paths
from epf.e1 import features, figures
from epf.evaluation import metrics, walk_forward
from epf.models import attribution, boosting, forest, lear, naive

DAILY, HOURLY = "daily", "hourly"

# Una execució de fum és petita i ràpida a propòsit, i el seu error no es desa, ni
# es cita, ni es compara enlloc. Només una de mesura dóna un resultat.
SMOKE, MEASUREMENT = "smoke", "measurement"

Family = collections.namedtuple("Family", "matrix forecasts")

# Els tres pronòstics ingenus són una sola família de la graella i van junts en un
# sol directori.
NAIVE = {"previous_day": naive.previous_day,
         "previous_week": naive.previous_week,
         "similar_day": naive.similar_day}


def _unattributed(forecast):
    """Un pronòstic que no ajusta res: la predicció és una columna copiada, i la seva
    atribució seria aquella mateixa columna."""
    def day(X, y, when):
        return forecast(X, y, when), None
    return day


def families(args):
    """Cada família: quina matriu fa servir i, per pronòstic, què fa amb un dia.

    La funció d'un dia torna els 24 preus i l'atribució del model que els ha predit.
    """
    light = functools.partial(boosting.lightgbm, trees=args.trees, rate=args.rate,
                              leaves=args.leaves, seed=args.seed)
    extreme = functools.partial(boosting.xgboost, trees=args.trees, rate=args.rate,
                                depth=args.depth, seed=args.seed)
    return {
        "naive": Family(DAILY, {name: _unattributed(forecast)
                                for name, forecast in NAIVE.items()}),
        "lear": Family(DAILY, {"lear": functools.partial(lear.explain, windows=args.windows)}),
        "forest": Family(DAILY, {"forest": functools.partial(
            forest.explain, window=args.window, trees=args.trees,
            features_per_split=args.features_per_split, seed=args.seed)}),
        "lightgbm": Family(HOURLY, {"lightgbm": functools.partial(
            boosting.explain, light, window=args.window)}),
        "xgboost": Family(HOURLY, {"xgboost": functools.partial(
            boosting.explain, extreme, window=args.window)}),
    }


def run(family, matrices, args, report):
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

        forecasts[name] = walk_forward(predict, X, y, args.first, args.last,
                                       every=args.every,
                                       on_day=functools.partial(report, name))
        if kept:
            attributions[name] = kept
    return forecasts, attributions


def _git(*arguments):
    return subprocess.run(("git", *arguments), cwd=paths.ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def described(args, family, forecasts):
    """El que `run.json` ha de dir perquè una xifra es pugui tornar al codi que la va fer."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "stage": features.STAGE,
        "tier": args.tier,
        "config": args.config,
        "family": family,
        "forecasts": sorted(forecasts),
        "period": [str(args.first), str(args.last)],
        "days": len(next(iter(forecasts.values()))),
        "every": args.every,
        "window": args.window,
        "windows": [window or 0 for window in args.windows],
        "trees": args.trees,
        "rate": args.rate,
        "leaves": args.leaves,
        "depth": args.depth,
        "features_per_split": args.features_per_split,
        "seed": args.seed,
    }


def write(folder, forecasts, attributions, actual, record):
    """Desa tot el que ha deixat una família, un cop ja ha acabat."""
    folder.mkdir(parents=True, exist_ok=True)
    run_files.save_forecasts(folder, forecasts)
    run_files.save_metrics(folder, metrics(forecasts, actual))
    for name, kept in attributions.items():
        attribution.save(folder, name, forecasts[name].index, kept)
    run_files.save_record(folder, record)


def day(text):
    return datetime.date.fromisoformat(text)


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    # Sense valor per defecte a propòsit: un període implícit és com els anys de
    # prova acaben fent-se servir per decidir coses.
    parser.add_argument("--from", dest="first", type=day, required=True,
                        help="primer dia a predir")
    parser.add_argument("--to", dest="last", type=day, required=True,
                        help="últim dia a predir")
    parser.add_argument("--tier", choices=(SMOKE, MEASUREMENT), required=True,
                        help="de fum, que no dóna cap xifra, o de mesura, que sí")
    parser.add_argument("--config", required=True,
                        help="nom d'aquesta configuració dins del directori de l'execució")
    parser.add_argument("--family", action="append",
                        help="només aquesta família; es pot repetir")
    parser.add_argument("--every", type=int, default=1,
                        help="pren un dia de cada N")
    parser.add_argument("--window", type=int, default=0,
                        help="dies de calibratge dels arbres, 0 per tot l'historial")
    parser.add_argument("--windows", type=int, nargs="+",
                        default=[window or 0 for window in lear.WINDOWS],
                        help="finestres de LEAR en dies, 0 per tot l'historial")
    parser.add_argument("--trees", type=int, default=forest.TREES)
    parser.add_argument("--rate", type=float, default=boosting.LEARNING_RATE)
    parser.add_argument("--leaves", type=int, default=boosting.LEAVES)
    parser.add_argument("--depth", type=int, default=boosting.MAX_DEPTH)
    parser.add_argument("--features-per-split", type=float, default=forest.FEATURES_PER_SPLIT)
    parser.add_argument("--seed", type=int, default=forest.SEED)
    args = parser.parse_args()
    args.window = args.window or None
    args.windows = tuple(window or None for window in args.windows)
    return args


def main():
    args = parse()
    daily, hourly = features.load()
    matrices = {DAILY: daily, HOURLY: hourly}
    actual = daily[1]
    table = families(args)
    chosen = args.family or list(table)

    started = {}

    def report(name, predicted_day):
        done = started[name] = started.get(name, 0) + 1
        print(f"  {name} {predicted_day.date()}  {done} dies", flush=True)

    failed = []
    for name in chosen:
        print(f"{name}: de {args.first} a {args.last}, configuració {args.config}", flush=True)
        clock = time.perf_counter()
        try:
            forecasts, attributions = run(table[name], matrices, args, report)
            folder = run_files.folder(args.tier, features.STAGE, args.config, name)
            write(folder, forecasts, attributions, actual,
                  described(args, name, forecasts))
            figures.draw(folder, actual)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            print(f"{name}: ha fallat, no se'n desa res", flush=True)
        else:
            print(f"{name}: {time.perf_counter() - clock:.0f} s, desat a {folder}", flush=True)

    if failed:
        print(f"famílies que han fallat i no han desat res: {', '.join(failed)}")


if __name__ == "__main__":
    main()
