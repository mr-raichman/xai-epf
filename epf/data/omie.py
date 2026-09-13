"""Descàrrega i lectura dels preus del mercat diari d'OMIE.

El format el descriu OMIE a `formato_ficheros_inf_pub`: cada línia de dades és
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
from epf.data.calendar import repair_dst

# El cinquè camp és MarginalPT, el portuguès. Els dos coincideixen quan la
# interconnexió no està congestionada i difereixen un terç dels dies, així que
# llegir el que no toca és correcte prou sovint com per semblar-ho (decisió 43).
SPAIN_FIELD = 5

BASE = "https://www.omie.es/es"
DOWNLOAD = f"{BASE}/file-download?parents=marginalpdbc&filename="
LISTING = f"{BASE}/file-access-list?realdir=marginalpdbc"

# Els anys anteriors només es publiquen empaquetats; a partir d'aquí, un fitxer
# per dia, i el llistat és l'únic lloc que diu quina versió té cada un.
ZIP_YEARS = range(2018, 2023)

FILENAME = re.compile(r"marginalpdbc_(\d{8})\.(\d+)")
HOUR_COLUMNS = [f"h{hour:02d}" for hour in range(HOURS_PER_DAY)]


def _day_of(stem):
    return datetime.date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))


def latest_version_per_day(directory=OMIE_PRICES):
    """Els fitxers agrupats per dia, quedant-se amb la versió més alta (decisió 44).

    Cinc dies del període es publiquen en versió 2 o 3, i tres d'ells no existeixen
    en cap versió inferior.
    """
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
    """Els preus espanyols del fitxer, un per període, en l'ordre en què hi surten."""
    with open(path, errors="ignore") as handle:
        return [float(line.split(";")[SPAIN_FIELD]) for line in handle if line[:1].isdigit()]


def to_hourly(periods):
    """Redueix els períodes a hores.

    Des de 2025-10-01 el mercat casa quatre productes per hora i el fitxer en porta
    96, o 92 i 100 els dies de canvi d'hora. Es promedien en grups de quatre
    (decisió 29): agafar el preu en punt hi escriuria un error amb forma de corba
    solar, +17.75 EUR/MWh de biaix a les 09 i -16.66 a les 19.
    """
    if len(periods) <= HOURS_PER_DAY + 1:
        return periods
    return [statistics.mean(periods[i * QUARTERS_PER_HOUR:(i + 1) * QUARTERS_PER_HOUR])
            for i in range(len(periods) // QUARTERS_PER_HOUR)]


def load(directory=OMIE_PRICES):
    """Els preus com un DataFrame de dies per hores, en EUR/MWh, indexat per data.

    La data és la del lliurament, no la de la subhasta, que va ser el dia abans.
    """
    chosen = latest_version_per_day(directory)
    if not chosen:
        raise FileNotFoundError(f"no hi ha cap fitxer de preus a {directory}")
    rows = [repair_dst(to_hourly(read_periods(path))) for path in chosen.values()]
    index = pd.DatetimeIndex(list(chosen), name="day")
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def _fetch(filename, directory):
    """Baixa un fitxer si encara no hi és. Torna si s'ha escrit alguna cosa."""
    target = pathlib.Path(directory) / filename
    if target.exists():
        return False
    response = requests.get(DOWNLOAD + filename, timeout=60)
    if response.status_code != 200 or not response.content.startswith(b"MARGINALPDBC"):
        print(f"  no s'ha pogut baixar {filename}: HTTP {response.status_code}")
        return False
    target.write_bytes(response.content)
    return True


def download(directory=OMIE_PRICES):
    """Posa al dia el directori de preus. És l'únic que hi escriu (decisió 40).

    Els anys empaquetats es baixen sencers un sol cop. Per als diaris es llegeix el
    llistat d'OMIE, que és l'únic lloc que diu quines versions existeixen: endevinar
    `.1` fa semblar que falten dies que no falten.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in directory.glob("marginalpdbc_*")}

    for year in ZIP_YEARS:
        if any(name.startswith(f"marginalpdbc_{year}") for name in existing):
            continue
        response = requests.get(f"{DOWNLOAD}marginalpdbc_{year}.zip", timeout=300)
        if response.status_code != 200:
            print(f"  no s'ha pogut baixar l'any {year}: HTTP {response.status_code}")
            continue
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(directory)
        print(f"  {year} desempaquetat")

    listing = requests.get(LISTING, timeout=60)
    listing.raise_for_status()
    wanted = sorted({f"marginalpdbc_{m[1]}.{m[2]}" for m in FILENAME.finditer(listing.text)})
    written = sum(_fetch(name, directory) for name in wanted if name not in existing)
    print(f"  {len(wanted)} fitxers al llistat, {written} de nous")

    days = latest_version_per_day(directory)
    print(f"  {len(days)} dies a disc, de {min(days)} a {max(days)}")
