"""Rutes on viuen les dades, les claus i les sortides."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ROOT / "api-keys"
DATA = ROOT / "data"

# Un conjunt de dades, una carpeta. La del preu és plana perquè és un sol conjunt
# amb dues fonts, i els seus noms no xoquen: `marginalpdbc_<dia>.<versió>` d'OMIE i
# `<any>.json` d'ESIOS. La resta d'indicadors d'ESIOS encara no són cap conjunt amb
# nom i es queden un directori per indicador.
PRICE = DATA / "price"
ESIOS = DATA / "esios"

# Figures de les dades, al costat de les dades que descriuen. Com tot el que hi ha
# sota `data/`, són fora de git.
PLOTS = DATA / "plots"

# Fora del paquet, perquè la roda es construeix de `epf/` i s'hi enduria qualsevol
# cosa que hi pengés, i fora de git.
RUNS = ROOT / "runs"
