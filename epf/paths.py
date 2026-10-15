"""Rutes on viuen els fitxers baixats, les claus i les sortides."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ROOT / "api-keys"

# Els fitxers tal com els serveix la font. Només hi escriuen els descarregadors i
# res no s'hi desa processat, de manera que el nom diu com hi han arribat i no cap
# contrast amb una còpia transformada que no existeix. Un conjunt de dades, una
# carpeta: la del preu és plana perquè és un sol conjunt amb dues fonts, i els seus
# noms no xoquen: `marginalpdbc_<dia>.<versió>` d'OMIE i `<any>.json` d'ESIOS. La de
# les previsions i la del programa en tenen una per indicador, perquè un fitxer d'ESIOS
# només porta el període al nom. La resta d'indicadors d'ESIOS encara no són cap
# conjunt amb nom i es queden un directori per indicador.
DOWNLOADS = ROOT / "downloads"
PRICE = DOWNLOADS / "price"
FORECASTS = DOWNLOADS / "forecasts"
SCHEDULE = DOWNLOADS / "schedule"
ESIOS = DOWNLOADS / "esios"

# Figures de les dades d'origen, una carpeta per conjunt. Fora de l'arbre que es
# baixa, perquè una entrada i una sortida no comparteixen directori.
FIGURES = ROOT / "figures"
PRICE_FIGURES = FIGURES / "price"

# Fora del paquet, perquè la roda es construeix de `epf/` i s'hi enduria qualsevol
# cosa que hi pengés, i fora de git.
RUNS = ROOT / "runs"

# Els fitxers d'una passada: un model per fitxer, la llavor i el lloc de la seva
# configuració a la cerca de la llavor, 0 la millor, i el registre de la família al costat.
SEED = "seed_{seed}_{rank}.npz"
SEED_RECORD = "record.json"


def run_folder(tier, stage, commit):
    """On viu una passada sencera: hi pengen els conjunts, les famílies i les figures
    que les comparen."""
    return RUNS / tier / stage / commit


# Les cerques, al costat de les passades i no a dins: una cerca és d'un conjunt i d'una
# partició, i no d'una passada, i la llegeixen totes les que es facin sobre aquelles files.
SEARCHES = ROOT / "searches"
