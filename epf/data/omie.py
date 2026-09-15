"""Descàrrega i lectura dels preus del mercat diari d'OMIE.

Segons l'especificació `formato_ficheros_inf_pub` d'OMIE, cada línia de dades és
`Any;Mes;Dia;Període;MarginalPT;MarginalES`, i el nom del fitxer acaba en `.v`,
on `v` és la versió.
"""
import datetime
import io
import pathlib
import re
import statistics
import zipfile

import pandas as pd
import requests

from epf.config.market import HOURS_PER_DAY, QUARTERS_PER_HOUR
from epf.config.paths import OMIE_PRICES
from epf.data.calendar import HOUR_COLUMNS, repair_dst

# Índex de MarginalES, el preu espanyol. L'índex 4 és el portuguès.
SPAIN_FIELD = 5

BASE = "https://www.omie.es/es"
DOWNLOAD = f"{BASE}/file-download?parents=marginalpdbc&filename="
LISTING = f"{BASE}/file-access-list?realdir=marginalpdbc"

FILENAME = re.compile(r"marginalpdbc_(\d{8})\.(\d+)")
# Anys que OMIE publica en un sol zip anual en lloc de fitxers diaris.
YEAR_ARCHIVE = re.compile(r"marginalpdbc_(\d{4})\.zip")


def _day_of(stem):
    return datetime.date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))


def latest_version_per_day(directory=OMIE_PRICES):
    """Fitxer de cada dia, en la seva versió més alta."""
    chosen = {}
    for path in pathlib.Path(directory).glob("marginalpdbc_*"):
        match = FILENAME.fullmatch(path.name)
        if match is None:
            continue
        day, version = _day_of(match[1]), int(match[2])
        if day not in chosen or version > chosen[day][0]:
            chosen[day] = (version, path)
    return {day: path for day, (_, path) in sorted(chosen.items())}


def read_periods(path):
    """Preus espanyols del fitxer, un per període, en l'ordre del fitxer."""
    with open(path, errors="ignore") as handle:
        return [float(line.split(";")[SPAIN_FIELD]) for line in handle if line[:1].isdigit()]


def to_hourly(periods):
    """Redueix els períodes quart-horaris a hores, amb la mitjana de cada quatre."""
    if len(periods) <= HOURS_PER_DAY + 1:
        return periods
    return [statistics.mean(periods[i * QUARTERS_PER_HOUR:(i + 1) * QUARTERS_PER_HOUR])
            for i in range(len(periods) // QUARTERS_PER_HOUR)]


def load(directory=OMIE_PRICES):
    """Preus en EUR/MWh, un dia de lliurament per fila i una hora per columna."""
    chosen = latest_version_per_day(directory)
    if not chosen:
        raise FileNotFoundError(f"no hi ha cap fitxer de preus a {directory}")
    rows = [repair_dst(to_hourly(read_periods(path))) for path in chosen.values()]
    index = pd.DatetimeIndex(list(chosen), name="day")
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def _fetch(filename, directory):
    """Baixa un fitxer si encara no hi és. Torna si s'ha escrit."""
    target = pathlib.Path(directory) / filename
    if target.exists():
        return False
    response = requests.get(DOWNLOAD + filename, timeout=60)
    if response.status_code != 200 or not response.content.startswith(b"MARGINALPDBC"):
        print(f"no s'ha pogut baixar {filename}: HTTP {response.status_code}")
        return False
    target.write_bytes(response.content)
    return True


def download(directory=OMIE_PRICES):
    """Actualitza el directori de preus amb tot el que llista OMIE."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in directory.glob("marginalpdbc_*")}

    listing = requests.get(LISTING, timeout=60)
    listing.raise_for_status()

    for year in sorted(set(YEAR_ARCHIVE.findall(listing.text))):
        if any(name.startswith(f"marginalpdbc_{year}") for name in existing):
            continue
        response = requests.get(f"{DOWNLOAD}marginalpdbc_{year}.zip", timeout=300)
        if response.status_code != 200:
            print(f"no s'ha pogut baixar l'any {year}: HTTP {response.status_code}")
            continue
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(directory)
        print(f"{year} desempaquetat")

    wanted = sorted({f"marginalpdbc_{m[1]}.{m[2]}" for m in FILENAME.finditer(listing.text)})
    written = sum(_fetch(name, directory) for name in wanted if name not in existing)
    print(f"{len(wanted)} fitxers diaris al llistat, {written} de nous")

    days = latest_version_per_day(directory)
    print(f"{len(days)} dies a disc, de {min(days)} a {max(days)}")
