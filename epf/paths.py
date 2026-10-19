"""Els camins de cada fitxer i carpeta del projecte."""

import pathlib

# L'arrel del repositori, de la qual surt cada camí: la carpeta que conté el paquet,
# perquè uv l'instal·la editable; instal·lat d'una altra manera, seria `site-packages`.
PACKAGE = pathlib.Path(__file__).resolve().parent
ROOT = PACKAGE.parent
KEYS = ROOT / "api-keys"

# La versió exacta de cada biblioteca, que `uv run` instal·la abans de cada comanda.
LOCK = ROOT / "uv.lock"

# Els fitxers tal com els serveix ESIOS, on només escriu la baixada: una carpeta per
# indicador, `downloads/<indicador>/<any>.json`.
DOWNLOADS = ROOT / "downloads"

# El dataset, el registre dels fitxers baixats d'on es va fer, i les figures que el
# descriuen, una carpeta per sèrie.
DATA = ROOT / "data"
DATASET = DATA / "dataset.csv"
DATASET_RECORD = DATA / "dataset.json"
DATA_FIGURES = DATA / "figures"

RUNS = ROOT / "runs"
SEARCHES = ROOT / "searches"

# Els fitxers d'una passada: el model ajustat d'una seed i d'una configuració, pel lloc
# de la configuració a la cerca de la seed, 0 la millor; el registre de la família, al
# costat; i cada matriu que hi van llegir els models, la diària i la per hora, a dalt de
# la passada.
SEED = "seed_{seed}_{rank}.npz"
SEED_RECORD = "record.json"
MATRIX = "matrix_{name}.csv"

# El fitxer on cada procés d'una passada o d'una cerca escriu la seva sortida, i on van
# les traces de les seves fallades.
WORKERS_LOG = "workers.log"

# Una passada es diu pel dia local en què va començar i pel seu número de dues xifres
# dins del dia, `2026-10-14_01`. La d'un assaig, i la cerca d'assaig d'una família,
# porten davant `SMOKE_PREFIX` i es compten a part.
SMOKE_PREFIX = "assaig_"


def runs(smoke):
    """Torna les passades d'assaig o de mesura, de la més antiga a la més nova."""
    return sorted(RUNS.glob(f"{SMOKE_PREFIX if smoke else ''}????-??-??_??"))


def next_run(smoke, day):
    """Torna la carpeta de la passada següent d'assaig o de mesura del dia `day`.

    El seu número és el més alt del dia més u, o 01 si encara no n'hi ha cap.
    """
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


def relative(path):
    """Torna un camí relatiu a l'arrel i amb `/`, com el mostra la consola."""
    return pathlib.Path(path).relative_to(ROOT, walk_up=True).as_posix()


def write_whole(path, text):
    """Escriu un text en UTF-8 sense deixar mai el fitxer a mitges.

    Un procés que s'atura mentre escriu deixa el fitxer com era, i `<nom>.part` al
    costat.
    """
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
