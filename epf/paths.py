"""Rutes on viuen les dades i les claus."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ROOT / "api-keys"
DATA = ROOT / "data"
OMIE_PRICES = DATA / "omie" / "marginalpdbc"
ESIOS = DATA / "esios"

# Sortides, totes dues fora del paquet perquè la roda es construeix de `epf/` i
# s'hi enduria qualsevol cosa que hi pengés, i totes dues fora de git.
RUNS = ROOT / "runs"
PLOTS = ROOT / "plots"
