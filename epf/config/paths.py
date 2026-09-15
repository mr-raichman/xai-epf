"""Rutes on viuen les dades i les claus."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
KEYS = ROOT / "api-keys"
DATA = ROOT / "data"
OMIE_PRICES = DATA / "omie" / "marginalpdbc"
ESIOS = DATA / "esios"
