"""Descàrrega i lectura de les sèries de Red Eléctrica a l'API d'ESIOS.

Es desa la resposta de l'API tal com arriba, un fitxer per període, anual o mensual
en les sèries de `MONTHLY`, amb les mitjanes horàries que calcula el servidor.

El directori es dona, no es deriva de l'indicador: la sèrie del preu viu plana amb
la resta del seu conjunt de dades i no en un directori propi. `series_folder` és el
que hi viu a `ESIOS`, que és la resta d'indicadors.

Aquí hi ha el client, amb la clau que llegeix del fitxer `api-keys` de l'arrel, i els
fets de com ESIOS serveix cada sèrie. Quins indicadors formen un conjunt de dades és del
conjunt, i es declara al seu mòdul.
"""
import datetime
import json
import pathlib

import pandas as pd
import requests

from epf.market import MARKET_TIMEZONE
from epf.paths import ESIOS, KEYS
from epf.calendar import HOUR_COLUMNS, repair_dst

URL = "https://api.esios.ree.es/indicators/"
ACCEPT = "application/json; application/vnd.esios-api-v1+json"

# Primer any que es demana, més enrere del que la font arriba a servir: la decisió
# és baixar cada font tan enrere com vagi, i una resposta buida es desa igual i no
# es torna a demanar mai més.
FIRST_YEAR = 2010

# Temps després de tancar un any a partir del qual el seu fitxer es dona per complet.
PUBLICATION_MARGIN = datetime.timedelta(days=2)

DEMAND = 1293
PBF_WIND = 12

# Indicador i la seva àncora. ESIOS omet les hores que valen zero. Sense àncora,
# la sèrie no pot valer zero i una hora absent queda absent. Amb àncora, una hora
# absent val zero si l'àncora la va publicar.
SERIES = {
    # Demanda real, i la previsió D+1 d'eòlica més fotovoltaica.
    DEMAND: None, 10358: None,
    # Generació en temps real.
    551: None, 1295: None, 10206: None, 10351: None, 10352: None,
    1294: DEMAND, 546: DEMAND, 547: DEMAND, 548: DEMAND, 549: DEMAND, 550: DEMAND,
    1296: DEMAND, 1297: DEMAND, 2078: DEMAND, 2079: DEMAND, 2344: DEMAND,
    # Saldo net de les interconnexions en temps real.
    10207: DEMAND, 10208: DEMAND, 10209: DEMAND,
    # Programa base de funcionament (PBF), per tecnologia i interconnexió. Les nou que
    # suma el conjunt del programa són al seu mòdul; l'eòlica hi és igualment, perquè és
    # l'àncora de les altres.
    PBF_WIND: None,
    **{indicator: PBF_WIND for indicator in (5, 6, 10, 13, 14, 15, 16, *range(18, 36))},
    10104: PBF_WIND, 10113: PBF_WIND, 10122: PBF_WIND, 10186: PBF_WIND, 462: PBF_WIND,
    # Potència disponible, publicada per província.
    **{indicator: PBF_WIND for indicator in range(472, 480)}, 10001: PBF_WIND,
    # Capacitat d'intercanvi.
    1844: PBF_WIND, 1845: PBF_WIND, 1846: PBF_WIND, 1848: PBF_WIND, 1849: PBF_WIND,
    1850: PBF_WIND, 506: PBF_WIND, 507: PBF_WIND, 10241: PBF_WIND, 10242: PBF_WIND,
    10018: PBF_WIND, 10019: PBF_WIND,
}

# Sèries publicades per província, que se sumen.
PROVINCES = {*range(472, 480), 10001}

# Sèries la resposta anual de les quals supera el límit de 60 s del servidor
# (HTTP 504). Es demanen i es desen per mesos.
MONTHLY = {472, 473, 477, 10001}

# Previsions que ESIOS sobreescriu amb actualitzacions posteriors al tancament de
# la subhasta. No són admissibles com a variables d'entrada.
REVISED_FORECASTS = {460, 541, 542, 543, 10034, 10249}


def periods(indicator, today):
    """Nom del fitxer, primer dia i últim dia de cada període que es demana, fins a l'any de `today`.

    L'any en curs es demana sencer: la potència disponible es publica per endavant.
    """
    for year in range(FIRST_YEAR, today.year + 1):
        if indicator not in MONTHLY:
            yield str(year), datetime.date(year, 1, 1), datetime.date(year, 12, 31)
            continue
        for month in range(1, 13):
            first = datetime.date(year, month, 1)
            following = (first + datetime.timedelta(days=31)).replace(day=1)
            yield f"{year}-{month:02d}", first, following - datetime.timedelta(days=1)


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
        lines = path.read_text().splitlines()
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


def series_folder(indicator, directory=ESIOS):
    """On viu un indicador que no pertany a cap conjunt de dades amb nom."""
    return pathlib.Path(directory) / str(indicator)


def download(indicator, folder, session=None, aggregation="average"):
    """Baixa cada període d'un indicador que encara no tingui un fitxer complet.

    Un indicador i no tots: qui en vol diversos els recorre, i és el conjunt de
    dades qui diu quins són seus i on van, i com ha d'agregar el servidor el que publica
    per sota de l'hora: la mitjana, o la suma per a una energia per quart d'hora.
    """
    session = session or session_for()
    folder = pathlib.Path(folder)
    written = 0
    for name, first, last in periods(indicator, datetime.date.today()):
        path = folder / f"{name}.json"
        if _is_complete(path, last):
            continue
        # ESIOS interpreta les dates en hora local.
        response = session.get(URL + str(indicator), timeout=600, params={
            "start_date": f"{first}T00:00", "end_date": f"{last}T23:59",
            "time_trunc": "hour", "time_agg": aggregation})
        if response.status_code != 200 or "indicator" not in _json_or_empty(response):
            print(f"no s'ha pogut baixar l'indicador {indicator} de {name}: "
                  f"HTTP {response.status_code}", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_text(response.text)
        partial.replace(path)
        written += 1
    print(f"indicador {indicator}: {written} fitxers baixats", flush=True)


def values(folder, monthly=False):
    """Tots els valors desats d'una sèrie, de tots els seus períodes.

    Només els fitxers del format de la sèrie: un fitxer anual oblidat al costat dels
    mensuals sumaria dues vegades les mateixes hores. En una carpeta plana compartida
    amb una altra font, el patró és també el que deixa fora els fitxers de l'altra.
    """
    folder = pathlib.Path(folder)
    paths = sorted(folder.glob("????-??.json" if monthly else "????.json"))
    if not paths:
        raise FileNotFoundError(f"no hi ha cap període d'ESIOS desat a {folder}")
    return [value for path in paths for value in json.loads(path.read_text())["indicator"]["values"]]


def published_instants(values):
    """Els instants, en UTC, en què hi ha algun valor."""
    return set(pd.to_datetime([value["datetime"] for value in values], format="ISO8601",
                              utc=True))


def hourly(values, anchor=None, past_last_day=False):
    """Valors d'ESIOS com a dies per hores, sumant les geografies de cada instant, i
    quantes hores va publicar la resposta cada dia local.

    La clau és `datetime`, que porta el desplaçament horari: `datetime_utc` etiqueta
    malament la primera 02:00 del canvi d'hora d'octubre. Amb `anchor`, els instants
    que va publicar l'àncora, una hora absent val zero si l'àncora la va publicar, entre
    el primer i l'últim dia publicat de la sèrie, o fins a l'últim de l'àncora amb
    `past_last_day`.

    El recompte és d'hores i no de valors: una sèrie per províncies en porta una per
    província a cada instant. Comparat amb `local_hours`, diu si la resposta té la
    forma del calendari local o si alguna hora ha caigut fora de la graella.
    """
    if not values:
        empty = pd.DatetimeIndex([], name="day")
        return (pd.DataFrame(columns=HOUR_COLUMNS, index=empty),
                pd.Series(dtype=int, index=empty))
    instants = pd.to_datetime([value["datetime"] for value in values], format="ISO8601", utc=True)
    series = pd.Series([value["value"] for value in values], index=instants).groupby(level=0).sum()

    published = set(series.index.tz_convert(MARKET_TIMEZONE).date)
    first, last = min(published), max(published)
    if anchor and past_last_day:
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
    days = series.index.tz_convert(MARKET_TIMEZONE).normalize().tz_localize(None)
    return (pd.DataFrame(list(rows.values()), index=index, columns=HOUR_COLUMNS),
            days.value_counts().reindex(index, fill_value=0))


def single_geography(values, indicator):
    """Els valors, si són d'una sola geografia. Si no, aixeca: `hourly` les sumaria."""
    names = {value.get("geo_name") for value in values}
    if len(names) > 1:
        raise ValueError(f"l'indicador {indicator} té diverses geografies: {sorted(names)}")
    return values


def _stored(indicator, directory):
    """Els valors d'un indicador del seu propi directori, amb el format que li toca."""
    return values(series_folder(indicator, directory), indicator in MONTHLY)


def load(indicator, directory=ESIOS):
    """Una sèrie admissible, en dies per hores, i les hores publicades cada dia."""
    if indicator in REVISED_FORECASTS:
        raise ValueError(f"l'indicador {indicator} és una previsió revisada, no admissible")
    if indicator not in SERIES:
        raise KeyError(f"l'indicador {indicator} no és a la llista de sèries")
    stored = _stored(indicator, directory)
    if indicator not in PROVINCES:
        single_geography(stored, indicator)
    anchor = SERIES[indicator]
    return hourly(stored, None if anchor is None else
                  published_instants(_stored(anchor, directory)))
