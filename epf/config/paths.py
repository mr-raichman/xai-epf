"""On viuen les dades. Res del repositori no s'escriu aquí (decisió 40)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OMIE_PRICES = DATA / "omie" / "marginalpdbc"
