"""Rutes on viuen les dades."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OMIE_PRICES = DATA / "omie" / "marginalpdbc"
