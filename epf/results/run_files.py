"""Què hi ha dins del directori d'una passada, i on és.

El commit nomena la passada, perquè és el que fixa tots els seus números, i a dins hi
ha un model per fitxer i el registre de cada família al costat.
"""
import datetime
import subprocess

from epf import paths

# Els fitxers d'una passada: un model per fitxer, la llavor i el lloc de la seva
# configuració a la cerca de la llavor, 0 la millor, i el registre de la família al costat.
SEED = "seed_{seed}_{rank}.npz"
SEED_RECORD = "record.json"


def git(*arguments):
    """La sortida d'una ordre de git sobre el repositori. Una passada es diu pel commit
    que la fa."""
    return subprocess.run(("git", *arguments), cwd=paths.ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def plain(value):
    """El valor tal com es pot escriure a JSON, sense perdre què era."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, datetime.date):
        return str(value)
    return value


def run_folder(tier, stage, commit):
    """On viu una passada sencera: hi pengen els conjunts, les famílies i les figures
    que les comparen."""
    return paths.RUNS / tier / stage / commit
