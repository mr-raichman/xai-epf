"""Descàrrega i lectura de les sèries de Red Eléctrica a l'API d'ESIOS.

Es desa la resposta de l'API tal com arriba, un fitxer per indicador i any, o per
mes en les sèries de `MONTHLY`, amb les mitjanes horàries que calcula el servidor.
"""
import datetime
import json
import pathlib

import pandas as pd
import requests

from epf.config.keys import api_key
from epf.config.market import MARKET_TIMEZONE
from epf.config.paths import ESIOS
from epf.data.calendar import HOUR_COLUMNS, repair_dst

URL = "https://api.esios.ree.es/indicators/"
ACCEPT = "application/json; application/vnd.esios-api-v1+json"

# Primer any que es demana, més enrere del que la font arriba a servir: la decisió
# és baixar cada font tan enrere com vagi, i una resposta buida es desa igual i no
# es torna a demanar mai més.
FIRST_YEAR = 2010

# Primer dia que la font serveix el preu, mesurat: una sol·licitud directa a l'API
# no en torna cap abans. És un fet d'ESIOS i no una tria nostra, que viu a
# `experiments.py`. Alguna altra sèrie comença unes hores abans.
ESIOS_FIRST_DAY = datetime.date(2014, 1, 1)

# Temps després de tancar un any a partir del qual el seu fitxer es dona per complet.
PUBLICATION_MARGIN = datetime.timedelta(days=2)

DEMAND = 1293
PBF_WIND = 12
PRICE = 600
PRICE_COUNTRIES = ("España", "Portugal", "Francia")

# Indicador i la seva àncora. ESIOS omet les hores que valen zero. Sense àncora,
# la sèrie no pot valer zero i una hora absent queda absent. Amb àncora, una hora
# absent val zero els dies en què l'àncora es va publicar.
SERIES = {
    # Demanda real i previsions D+1.
    DEMAND: None, 1775: None, 1777: None, 1779: None, 10358: None,
    # Generació en temps real.
    551: None, 1295: None, 10206: None, 10351: None, 10352: None,
    1294: DEMAND, 546: DEMAND, 547: DEMAND, 548: DEMAND, 549: DEMAND, 550: DEMAND,
    1296: DEMAND, 1297: DEMAND, 2078: DEMAND, 2079: DEMAND, 2344: DEMAND,
    # Saldo net de les interconnexions en temps real.
    10207: DEMAND, 10208: DEMAND, 10209: DEMAND,
    # Programa base de funcionament (PBF), per tecnologia i interconnexió.
    PBF_WIND: None,
    **{indicator: PBF_WIND for indicator in range(1, 36) if indicator != PBF_WIND},
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


def download(directory=ESIOS, key=None):
    """Baixa cada període de cada indicador que encara no té un fitxer complet."""
    session = requests.Session()
    session.headers.update({"x-api-key": key or api_key("ESIOS_KEY"), "Accept": ACCEPT})

    for indicator in sorted({*SERIES, PRICE, *REVISED_FORECASTS}):
        written = 0
        for name, first, last in periods(indicator, datetime.date.today()):
            path = pathlib.Path(directory) / str(indicator) / f"{name}.json"
            if _is_complete(path, last):
                continue
            # ESIOS interpreta les dates en hora local.
            response = session.get(URL + str(indicator), timeout=600, params={
                "start_date": f"{first}T00:00", "end_date": f"{last}T23:59",
                "time_trunc": "hour", "time_agg": "average"})
            if response.status_code != 200 or "indicator" not in _json_or_empty(response):
                print(f"no s'ha pogut baixar l'indicador {indicator} de {name}: "
                      f"HTTP {response.status_code}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".part")
            partial.write_text(response.text)
            partial.replace(path)
            written += 1
        print(f"indicador {indicator}: {written} fitxers baixats")


def _values(indicator, directory):
    """Tots els valors desats d'un indicador, de tots els períodes."""
    folder = pathlib.Path(directory) / str(indicator)
    # Només els fitxers del format de la sèrie: un fitxer anual oblidat al costat
    # dels mensuals sumaria dues vegades les mateixes hores.
    paths = sorted(folder.glob("????-??.json" if indicator in MONTHLY else "????.json"))
    if not paths:
        raise FileNotFoundError(f"no hi ha dades de l'indicador {indicator} a {folder}")
    return [value for path in paths for value in json.loads(path.read_text())["indicator"]["values"]]


def _published_days(values):
    """Dies locals amb algun valor."""
    instants = pd.to_datetime([value["datetime"] for value in values], format="ISO8601", utc=True)
    return set(instants.tz_convert(MARKET_TIMEZONE).date)


def hourly(values, anchor_days=None):
    """Valors d'ESIOS com a dies per hores, sumant les geografies de cada instant, i
    quantes hores va publicar la resposta cada dia local.

    La clau és `datetime`, que porta el desplaçament horari: `datetime_utc` etiqueta
    malament la primera 02:00 del canvi d'hora d'octubre. Amb `anchor_days`, una
    hora absent val zero els dies d'àncora entre el primer i l'últim dia publicat.

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
    zero_days = set() if anchor_days is None else {d for d in anchor_days if first <= d <= last}

    start = pd.Timestamp(first).tz_localize(MARKET_TIMEZONE)
    end = pd.Timestamp(last).tz_localize(MARKET_TIMEZONE) + pd.DateOffset(days=1)
    grid = pd.date_range(start, end, freq="h", inclusive="left")
    frame = pd.DataFrame({"day": grid.tz_localize(None).normalize(),
                          "value": series.reindex(grid.tz_convert("UTC")).to_numpy()})
    frame = frame[frame["day"].dt.date.isin(published | zero_days)]
    zero = frame["day"].dt.date.isin(zero_days)
    frame.loc[zero, "value"] = frame.loc[zero, "value"].fillna(0.0)

    rows = {day: repair_dst(hours.tolist()) for day, hours in frame.groupby("day")["value"]}
    index = pd.DatetimeIndex(list(rows), name="day")
    days = series.index.tz_convert(MARKET_TIMEZONE).normalize().tz_localize(None)
    return (pd.DataFrame(list(rows.values()), index=index, columns=HOUR_COLUMNS),
            days.value_counts().reindex(index, fill_value=0))


def _single_geography(values, indicator):
    names = {value.get("geo_name") for value in values}
    if len(names) > 1:
        raise ValueError(f"l'indicador {indicator} té diverses geografies: {sorted(names)}")
    return values


def load(indicator, directory=ESIOS):
    """Una sèrie admissible, en dies per hores, i les hores publicades cada dia."""
    if indicator in REVISED_FORECASTS:
        raise ValueError(f"l'indicador {indicator} és una previsió revisada, no admissible: "
                         "es llegeix amb load_revised_forecast")
    if indicator not in SERIES:
        raise KeyError(f"l'indicador {indicator} no és a la llista de sèries")
    values = _values(indicator, directory)
    if indicator not in PROVINCES:
        _single_geography(values, indicator)
    anchor = SERIES[indicator]
    anchor_days = None if anchor is None else _published_days(_values(anchor, directory))
    return hourly(values, anchor_days)


def load_price(country, directory=ESIOS):
    """Preu del mercat diari d'un país en EUR/MWh, i les hores publicades cada dia."""
    if country not in PRICE_COUNTRIES:
        raise KeyError(f"cap preu desat per a {country}; n'hi ha per a {PRICE_COUNTRIES}")
    return hourly([value for value in _values(PRICE, directory) if value["geo_name"] == country])


def load_revised_forecast(indicator, directory=ESIOS):
    """Una previsió revisada, que conté informació posterior al tancament de la subhasta,
    i les hores publicades cada dia."""
    if indicator not in REVISED_FORECASTS:
        raise KeyError(f"l'indicador {indicator} no és una previsió revisada")
    return hourly(_single_geography(_values(indicator, directory), indicator))
