"""Els camins que fan servir diversos mòduls, i l'escriptura d'un fitxer sencer."""

import datetime
import hashlib
import json
import pathlib

# L'arrel del repositori, de la qual surt cada camí: la carpeta que conté el paquet,
# perquè uv l'instal·la editable; instal·lat d'una altra manera, seria `site-packages`.
PACKAGE = pathlib.Path(__file__).resolve().parent
ROOT = PACKAGE.parent
KEYS = ROOT / "api-keys"

# La versió exacta de cada biblioteca, que `uv run` instal·la abans de cada comanda.
LOCK = ROOT / "uv.lock"

# Tot el que escriuen les comandes, i a part, la passada i les cerques d'un assaig.
OUTPUT = ROOT / "output"
SMOKE_OUTPUT = ROOT / "assaig_output"

# Els fitxers tal com els serveix ESIOS, on només escriu la baixada: una carpeta per
# indicador, `output/downloads/<sèrie>_<número>/<any>.json`.
DOWNLOADS = OUTPUT / "downloads"

# El dataset, el registre dels fitxers baixats d'on es va fer, i les figures que el
# descriuen, una carpeta per sèrie.
DATA = OUTPUT / "data"
DATASET = DATA / "dataset.csv"
DATASET_RECORD = DATA / "dataset.json"
DATA_FIGURES = DATA / "figures"

# Els fitxers d'una passada: el model ajustat d'una seed i d'una configuració, pel lloc
# de la configuració a la cerca de la seed, 0 la millor; el registre de la família, al
# costat; i cada matriu que hi van llegir els models, la diària i la per hora, a dalt de
# la passada.
MODEL = "seed_{seed}_{rank}.npz"
FAMILY_RECORD = "record.json"
MATRIX = "matrix_{name}.csv"

# El fitxer on cada procés d'una passada escriu la seva sortida, i on van les traces de
# les seves fallades.
WORKERS_LOG = "workers.log"

# Una passada es diu pel seu information set, en minúscules, i el dia i el mes locals
# en què va començar, `with_price_27-10`, i la d'un assaig porta davant
# `SMOKE_PREFIX`.
SMOKE_PREFIX = "assaig_"


def runs(smoke, set_name):
    """Torna la carpeta de les passades d'assaig o de mesura d'un information set."""
    return (SMOKE_OUTPUT if smoke else OUTPUT) / "runs" / set_name.lower()


def searches(smoke, set_name):
    """Torna la carpeta de les cerques d'assaig o de mesura d'un information set.

    Hi ha una carpeta per família.
    """
    return (SMOKE_OUTPUT if smoke else OUTPUT) / "searches" / set_name.lower()


def named_runs(smoke, set_name):
    """Torna les passades d'assaig o de mesura d'un information set amb nom de passada.

    Per ordre de nom. Cap altra carpeta de `runs(smoke, set_name)` no és una passada que
    es pugui continuar.
    """
    prefix = SMOKE_PREFIX if smoke else ""
    return sorted(runs(smoke, set_name).glob(f"{prefix}{set_name.lower()}_??-??"))


def next_run(smoke, set_name, day):
    """Torna la carpeta d'una passada d'un information set que comença el dia `day`.

    D'assaig si `smoke`.
    """
    prefix = SMOKE_PREFIX if smoke else ""
    return runs(smoke, set_name) / f"{prefix}{set_name.lower()}_{day:%d-%m}"


def relative(path):
    """Torna un camí relatiu a l'arrel i amb `/`, com el mostra la consola."""
    return pathlib.Path(path).relative_to(ROOT, walk_up=True).as_posix()


def digest(path):
    """Torna la SHA-256 d'un fitxer, en hexadecimal."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partial(path):
    """Torna on s'escriu un fitxer abans de portar el seu nom, `<nom>.part`."""
    return path.with_name(path.name + ".part")


def write_whole(path, text):
    """Escriu un text en UTF-8 sense deixar mai el fitxer a mitges.

    Un procés que s'atura mentre escriu deixa el fitxer com era, i `<nom>.part` al
    costat.
    """
    partial(path).write_text(text, encoding="utf-8")
    partial(path).replace(path)


def plain(value):
    """Torna un valor tal com es pot escriure a JSON, sense perdre què era.

    Converteix les dates, les tuples i les claus, i deixa que qualsevol altra cosa faci
    fallar l'escriptura, en lloc d'escriure-la com a text sense dir-ho.
    """
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, datetime.date):
        return str(value)
    return value


def write_record(path, content):
    """Escriu el registre `content` a JSON, sencer o gens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_whole(path, json.dumps(plain(content), indent=2) + "\n")
