"""Rutes on viuen els fitxers baixats, les claus i les sortides."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ROOT / "api-keys"

# Els fitxers tal com els serveix la font, una carpeta per indicador amb el seu nom,
# `downloads/<indicador>/<any>.json`, perquè un fitxer d'ESIOS només porta l'any al nom; a
# la del preu, també els d'OMIE, `marginalpdbc_<dia>.<versió>`, que no xoquen. Només hi
# escriu la baixada i res no s'hi desa processat, de manera que el nom diu com hi han
# arribat i no cap contrast amb una còpia transformada que no existeix.
DOWNLOADS = ROOT / "downloads"

# El conjunt de dades fet dels fitxers baixats, el registre dels fitxers d'on es va fer i
# les figures que el descriuen. Fora de l'arbre que es baixa, perquè una entrada i una
# sortida no comparteixen directori.
DATA = ROOT / "data"
DATASET = DATA / "dataset.csv"
DATASET_RECORD = DATA / "dataset.json"
DATA_FIGURES = DATA / "figures"

# Fora del paquet, perquè la roda es construeix de `epf/` i s'hi enduria qualsevol
# cosa que hi pengés, i fora de git.
RUNS = ROOT / "runs"

# Els fitxers d'una passada: un model per fitxer, la llavor i el lloc de la seva
# configuració a la cerca de la llavor, 0 la millor, i el registre de la família al costat.
SEED = "seed_{seed}_{rank}.npz"
SEED_RECORD = "record.json"


def run_folder(tier, stage, commit):
    """On viu una passada sencera: hi pengen les famílies i les figures que les
    comparen."""
    return RUNS / tier / stage / commit


# Les cerques, al costat de les passades i no a dins: una cerca és d'una entrada i d'una
# partició, i no d'una passada, i la llegeixen totes les que es facin sobre aquelles files.
SEARCHES = ROOT / "searches"
