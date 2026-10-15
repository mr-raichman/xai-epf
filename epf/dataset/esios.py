"""Com es baixen i es llegeixen els fitxers d'ESIOS, l'API de Red Eléctrica.

Es desa la resposta de l'API tal com arriba, un fitxer per any, amb el que el servidor
agrega de cada hora: la mitjana o, d'una energia servida per quarts d'hora, la suma. Quins
indicadors es baixen i es llegeixen, i com els serveix ESIOS, ho diu la taula de
`build.py`, un `Indicator` per fila; aquí hi ha el client, amb la clau que llegeix del
fitxer `api-keys` de l'arrel, i el lector: la geografia, la graella d'hores, que refusa
qualsevol valor fora de l'hora en punt, la regla del zero amb la seva àncora i la
reparació dels canvis d'hora.

Hi ha també el rellotge del mercat, que el lector és el primer a fer servir: la zona
horària i la reparació dels dies de canvi d'hora.
"""
import collections
import datetime
import json
import pathlib
import statistics
import zoneinfo

import pandas as pd
import requests

from epf.parameters import HOUR_COLUMNS, HOURS_PER_DAY
from epf.paths import KEYS, write_whole

# La zona horària del mercat diari espanyol, en la qual es compten els dies i les hores.
MARKET_TIMEZONE = zoneinfo.ZoneInfo("Europe/Madrid")

# Hora local que falta l'últim diumenge de març i es repeteix l'últim d'octubre.
DST_HOUR = 2

URL = "https://api.esios.ree.es/indicators/"
ACCEPT = "application/json; application/vnd.esios-api-v1+json"

# Primer any que es demana, més enrere del que la font arriba a servir: la decisió
# és baixar cada font tan enrere com vagi, i una resposta buida es desa igual i no
# es torna a demanar mai més.
FIRST_YEAR = 2010

# Temps després de tancar un any a partir del qual el seu fitxer es dona per complet.
PUBLICATION_MARGIN = datetime.timedelta(days=2)

# Com serveix ESIOS un indicador: el seu número; com n'agrega el servidor cada hora,
# `average` o `sum`, el seu `time_agg`; la geografia que se'n pren i, amb `only`, si ha de
# ser l'única del fitxer o, sense, si es tria entre les que porta; el nom de l'indicador
# que li fa d'àncora, o `None`; i les hores que ESIOS no hi va publicar, que no s'omplen.
Indicator = collections.namedtuple("Indicator", "id aggregation geography only anchor gaps")

# Previsions que ESIOS sobreescriu amb actualitzacions posteriors al tancament de
# la subhasta. No són admissibles com a variables d'entrada.
REVISED_FORECASTS = {460, 541, 542, 543, 10034, 10249}


def periods(today):
    """Nom del fitxer, primer dia i últim dia de cada any que es demana, fins al de `today`.

    L'any en curs es demana sencer: el que ESIOS publica per endavant, com les previsions
    de l'endemà, hi arriba igual.
    """
    for year in range(FIRST_YEAR, today.year + 1):
        yield str(year), datetime.date(year, 1, 1), datetime.date(year, 12, 31)


def _is_complete(path, last):
    """Un fitxer és complet si es va baixar prou després de tancar-se el seu període."""
    if not path.exists():
        return False
    closed = datetime.datetime.combine(last + datetime.timedelta(days=1), datetime.time(),
                                       MARKET_TIMEZONE) + PUBLICATION_MARGIN
    return datetime.datetime.fromtimestamp(path.stat().st_mtime, MARKET_TIMEZONE) >= closed


def _json_or_empty(response):
    try:
        return response.json()
    except ValueError:
        return {}


def api_key(name, path=KEYS):
    """Valor de la línia `name=valor` del fitxer de claus de l'arrel."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"no existeix {path}; ha de contenir la línia {name}=...") from None
    for line in lines:
        found, separator, value = line.partition("=")
        if separator and found.strip() == name and value.strip():
            return value.strip()
    raise KeyError(f"falta la línia {name}=... a {path}")


def session_for(key=None):
    """Sessió amb la clau i la capçalera que l'API demana."""
    session = requests.Session()
    session.headers.update({"x-api-key": key or api_key("ESIOS_KEY"), "Accept": ACCEPT})
    return session


def download(indicator, folder, session=None):
    """Baixa cada any d'un indicador que encara no tingui un fitxer complet, amb l'agregació
    que l'indicador demana.

    Un indicador i no tots: qui en vol diversos els recorre, i diu on va cada un.
    """
    session = session or session_for()
    folder = pathlib.Path(folder)
    written = 0
    for name, first, last in periods(datetime.date.today()):
        path = folder / f"{name}.json"
        if _is_complete(path, last):
            continue
        # ESIOS interpreta les dates en hora local.
        response = session.get(URL + str(indicator.id), timeout=600, params={
            "start_date": f"{first}T00:00", "end_date": f"{last}T23:59",
            "time_trunc": "hour", "time_agg": indicator.aggregation})
        if response.status_code != 200 or "indicator" not in _json_or_empty(response):
            print(f"no s'ha pogut baixar {folder.name} de {name}: "
                  f"HTTP {response.status_code}", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        write_whole(path, response.text)
        written += 1
    print(f"{folder.name}: {written} fitxers baixats", flush=True)


def files(folder):
    """Els fitxers desats d'un indicador, un per any, en ordre.

    Només els del seu format, `<any>.json`, i no el `.part` que hi hagi deixat una baixada
    interrompuda.
    """
    paths = sorted(pathlib.Path(folder).glob("????.json"))
    if not paths:
        raise FileNotFoundError(f"no hi ha cap any d'ESIOS desat a {folder}")
    return paths


def values(paths):
    """Tots els valors desats d'un indicador, de tots els seus fitxers."""
    return [value for path in paths for value in json.loads(path.read_text(encoding="utf-8"))["indicator"]["values"]]


def admissible(values, indicator):
    """Els valors d'un indicador que se'n poden fer servir: els de la seva geografia.

    Una previsió revisada es refusa sencera. Amb `only`, un fitxer que porti cap altra
    geografia es refusa: vol dir que ESIOS ja no serveix l'indicador com diu la taula, i
    el que se'n triés podria ser una altra cosa amb el mateix nom. Sense, la geografia es
    tria entre les que porta el fitxer, com la del preu, que en porta sis.
    """
    if indicator.id in REVISED_FORECASTS:
        raise ValueError(f"l'indicador {indicator.id} és una previsió revisada, no admissible")
    if indicator.only:
        others = {value["geo_name"] for value in values} - {indicator.geography}
        if others:
            raise ValueError(f"l'indicador {indicator.id} porta altres geografies que "
                             f"{indicator.geography}: {sorted(others)}")
    return [value for value in values if value["geo_name"] == indicator.geography]


def published_instants(values):
    """Els instants, en UTC, en què hi ha algun valor."""
    return set(pd.to_datetime([value["datetime"] for value in values], format="ISO8601",
                              utc=True))


def repair_dst(hours):
    """Torna sempre 24 hores, corregint els dies de canvi d'hora amb mitjanes.

    El dia de 23 hores rep la mitjana de les dues veïnes a l'hora que falta, i el
    de 25 fusiona les dues hores repetides en la seva mitjana.
    """
    if len(hours) == HOURS_PER_DAY:
        return hours
    if len(hours) == HOURS_PER_DAY - 1:
        inserted = statistics.mean(hours[DST_HOUR - 1:DST_HOUR + 1])
        return hours[:DST_HOUR] + [inserted] + hours[DST_HOUR:]
    if len(hours) == HOURS_PER_DAY + 1:
        merged = statistics.mean(hours[DST_HOUR:DST_HOUR + 2])
        return hours[:DST_HOUR] + [merged] + hours[DST_HOUR + 2:]
    raise ValueError(f"un dia de {len(hours)} hores, s'esperaven "
                     f"{HOURS_PER_DAY - 1}, {HOURS_PER_DAY} o {HOURS_PER_DAY + 1}")


def hourly(values, anchor=None):
    """Valors d'ESIOS com a dies per hores.

    La clau és `datetime`, que porta el desplaçament horari: `datetime_utc` etiqueta
    malament la primera 02:00 del canvi d'hora d'octubre. Un valor fora de l'hora en punt
    es refusa: la graella només té hores, i el deixaria fora sense dir-ho, com en deixaria
    tres de cada quatre d'un indicador servit per quarts d'hora. Dos valors al mateix
    instant es refusen: d'una sola geografia, són una hora repetida, i sumats la doblarien
    sense dir-ho. Amb tots dos refusats, cap dia local no porta més valors que hores té.
    Amb `anchor`, els instants que va publicar l'àncora, una hora absent val zero si
    l'àncora la va publicar, del primer dia publicat de l'indicador a l'últim de l'àncora,
    també passat l'últim de l'indicador.
    """
    if not values:
        return pd.DataFrame(columns=HOUR_COLUMNS, index=pd.DatetimeIndex([], name="day"))
    instants = pd.to_datetime([value["datetime"] for value in values], format="ISO8601", utc=True)
    off = instants[instants != instants.floor("h")]
    if len(off):
        raise ValueError(f"{len(off)} valors fora de l'hora en punt, el primer {off[0]}")
    repeated = instants[instants.duplicated()]
    if len(repeated):
        raise ValueError(f"{len(repeated)} instants amb més d'un valor, el primer {repeated[0]}")
    series = pd.Series([value["value"] for value in values], index=instants)

    published = set(series.index.tz_convert(MARKET_TIMEZONE).date)
    first, last = min(published), max(published)
    if anchor:
        last = max(last, max(pd.DatetimeIndex(list(anchor)).tz_convert(MARKET_TIMEZONE).date))

    start = pd.Timestamp(first).tz_localize(MARKET_TIMEZONE)
    end = pd.Timestamp(last).tz_localize(MARKET_TIMEZONE) + pd.DateOffset(days=1)
    grid = pd.date_range(start, end, freq="h", inclusive="left")
    zero = grid.tz_convert("UTC").isin(list(anchor or ()))
    frame = pd.DataFrame({"day": grid.tz_localize(None).normalize(),
                          "value": series.reindex(grid.tz_convert("UTC")).to_numpy(),
                          "zero": zero})
    frame.loc[frame["zero"], "value"] = frame.loc[frame["zero"], "value"].fillna(0.0)
    zero_days = set(frame.loc[frame["zero"], "day"].dt.date)
    frame = frame[frame["day"].dt.date.isin(published | zero_days)]

    rows = {day: repair_dst(hours.tolist()) for day, hours in frame.groupby("day")["value"]}
    index = pd.DatetimeIndex(list(rows), name="day")
    return pd.DataFrame(list(rows.values()), index=index, columns=HOUR_COLUMNS)
