"""Rutes on viuen els fitxers baixats, les claus i les sortides, com es diu cada
passada, i com s'escriu sencer un fitxer de text."""

import pathlib

# El paquet, de tots els fitxers `.py` del qual es fa l'empremta del codi, i l'arrel del
# projecte, on és.
PACKAGE = pathlib.Path(__file__).resolve().parent
ROOT = PACKAGE.parent
KEYS = ROOT / "api-keys"

# La versió exacta de cada biblioteca, que `uv run` instal·la abans de cada ordre.
LOCK = ROOT / "uv.lock"

# Els fitxers tal com els serveix la font, una carpeta per indicador amb el seu nom,
# `downloads/<indicador>/<any>.json`, perquè un fitxer d'ESIOS només porta l'any al nom.
# Només hi escriu la baixada i res no s'hi desa processat, de manera que el nom diu com
# hi han arribat i no cap contrast amb una còpia transformada que no existeix.
DOWNLOADS = ROOT / "downloads"

# El conjunt de dades fet dels fitxers baixats, el registre dels fitxers d'on es va fer
# i les figures que el descriuen, una carpeta per sèrie. Fora de l'arbre que es baixa,
# perquè una entrada i una sortida no comparteixen directori.
DATA = ROOT / "data"
DATASET = DATA / "dataset.csv"
DATASET_RECORD = DATA / "dataset.json"
DATA_FIGURES = DATA / "figures"

# Fora del paquet, perquè la roda es construeix de `epf/` i s'hi enduria qualsevol
# cosa que hi pengés, i fora de git.
RUNS = ROOT / "runs"

# Els fitxers d'una passada: un model per fitxer, la llavor i el lloc de la seva
# configuració a la cerca de la llavor, 0 la millor, i el registre de la família al
# costat; i, a dalt de la passada, cada matriu que hi van llegir els models, la diària i
# la per hora.
SEED = "seed_{seed}_{rank}.npz"
SEED_RECORD = "record.json"
MATRIX = "matrix_{name}.csv"

# El fitxer on cada procés d'una passada o d'una cerca escriu la seva sortida, i on van
# les traces de les seves fallades.
WORKERS_LOG = "workers.log"

# Una passada es diu pel dia local en què va començar i pel seu número de dues xifres
# dins del dia, `2026-10-14_01`, de manera que la de nom més gran és la més nova. La
# d'un assaig porta davant `SMOKE_PREFIX` i es compta a part, i la cerca d'assaig d'una
# família també.
SMOKE_PREFIX = "assaig_"


def runs(smoke):
    """Les passades d'un tipus, de la més antiga a la més nova."""
    return sorted(RUNS.glob(f"{SMOKE_PREFIX if smoke else ''}????-??-??_??"))


def next_run(smoke, day):
    """La carpeta de la passada següent d'un tipus el dia `day`: el número més alt del
    dia més u, o 01 si encara no n'hi ha cap."""
    # shortcut: dues xifres, 99 passades d'un tipus per dia; la 100a tindria un nom fora
    # del patró de `runs` i la següent hi tornaria a escriure. Una tercera xifra, si mai
    # un dia en necessita més.
    prefix = f"{SMOKE_PREFIX if smoke else ''}{day.isoformat()}_"
    numbers = [
        int(path.name.removeprefix(prefix))
        for path in runs(smoke)
        if path.name.startswith(prefix)
    ]
    return RUNS / f"{prefix}{max(numbers, default=0) + 1:02d}"


# Les cerques, al costat de les passades i no a dins: una cerca és d'una entrada i d'una
# partició, i no d'una passada, i la llegeixen totes les que es facin sobre aquelles
# files.
SEARCHES = ROOT / "searches"


def relative(path):
    """El camí d'un fitxer de l'arrel tal com el veu qui hi treballa, relatiu a l'arrel
    i amb `/`."""
    return pathlib.Path(path).relative_to(ROOT, walk_up=True).as_posix()


def write_whole(path, text):
    """Escriu un text en UTF-8 primer a un fitxer a part i després al seu nom, de manera
    que un fitxer amb nom és sempre sencer."""
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
