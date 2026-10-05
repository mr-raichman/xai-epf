"""Passada completa del model de referència sobre un període, dia a dia.

Imprimeix l'error i no desa res: els números els llegeix qui executa l'script.
"""
import argparse
import datetime
import functools
import time

from epf.config import experiments
from epf.data import prices
from epf.evaluation import mae, mae_per_hour, walk_forward
from epf.features import matrix
from epf.models import lear

def day(text):
    return datetime.date.fromisoformat(text)


parser = argparse.ArgumentParser(description=__doc__)
# Sense valor per defecte a propòsit: un període implícit és com els anys de prova
# acaben fent-se servir per decidir coses.
parser.add_argument("--from", dest="first", type=day, required=True,
                    help="primer dia a predir")
parser.add_argument("--to", dest="last", type=day, required=True,
                    help="últim dia a predir")
parser.add_argument("--windows", type=int, nargs="+",
                    default=[w or 0 for w in experiments.LEAR_WINDOWS],
                    help="finestres de calibratge en dies, 0 per tot l'historial; "
                         "amb més d'una es fa servir la mitjana de les seves prediccions")
parser.add_argument("--every", type=int, default=1,
                    help="pren un dia de cada N")
args = parser.parse_args()

windows = tuple(window or None for window in args.windows)
if len(windows) == 1:
    predict = functools.partial(lear.forecast, window=windows[0],
                                max_iter=experiments.BASE["lear_max_iter"])
else:
    predict = functools.partial(lear.forecast_ensemble, windows=windows,
                                max_iter=experiments.BASE["lear_max_iter"])

X, y = matrix.build(prices.load(), experiments.LAGS)
names = ", ".join("tot l'historial" if w is None else f"{w} dies" for w in windows)
print(f"LEAR de {args.first} a {args.last}, un dia de cada {args.every}")
print(f"finestres: {names}")

started = time.perf_counter()
done = [0]


def report(predicted_day):
    done[0] += 1
    elapsed = time.perf_counter() - started
    print(f"  {predicted_day.date()}  {done[0]} dies, {elapsed / done[0]:.1f} s/dia", flush=True)


predicted = walk_forward(predict, X, y, args.first, args.last, every=args.every, on_day=report)

print()
print(f"{len(predicted)} dies predits en {time.perf_counter() - started:.0f} s")
print(f"MAE: {mae(predicted, y):.3f} EUR/MWh")
print("MAE per hora:")
for hour, error in mae_per_hour(predicted, y).items():
    print(f"  {hour}  {error:6.3f}")
