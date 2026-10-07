"""Rutes on viuen els fitxers baixats, les claus i les sortides."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ROOT / "api-keys"

# Els fitxers tal com els serveix la font. Només hi escriuen els descarregadors i
# res no s'hi desa processat, de manera que el nom diu com hi han arribat i no cap
# contrast amb una còpia transformada que no existeix. Un conjunt de dades, una
# carpeta: la del preu és plana perquè és un sol conjunt amb dues fonts, i els seus
# noms no xoquen: `marginalpdbc_<dia>.<versió>` d'OMIE i `<any>.json` d'ESIOS. La
# resta d'indicadors d'ESIOS encara no són cap conjunt amb nom i es queden un
# directori per indicador.
DOWNLOADS = ROOT / "downloads"
PRICE = DOWNLOADS / "price"
ESIOS = DOWNLOADS / "esios"

# Figures de les dades d'origen, una carpeta per conjunt. Fora de l'arbre que es
# baixa, perquè una entrada i una sortida no comparteixen directori.
FIGURES = ROOT / "figures"
PRICE_FIGURES = FIGURES / "price"

# Fora del paquet, perquè la roda es construeix de `epf/` i s'hi enduria qualsevol
# cosa que hi pengés, i fora de git.
RUNS = ROOT / "runs"
