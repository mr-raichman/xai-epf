"""El client i el lector dels fitxers d'ESIOS, l'API de Red Eléctrica."""

import collections
import datetime
import json
import pathlib
import statistics
import zoneinfo

import pandas as pd
import requests

from epf import Refusal
from epf.parameters import HOUR_COLUMNS, HOURS_PER_DAY
from epf.paths import KEYS, relative, write_whole

# La zona horària del mercat diari espanyol, en la qual es compten els dies i les hores.
MARKET_TIMEZONE = zoneinfo.ZoneInfo("Europe/Madrid")

# Hora local que falta l'últim diumenge de març i es repeteix l'últim d'octubre.
DST_HOUR = 2

URL = "https://api.esios.ree.es/indicators/"
ACCEPT = "application/json; application/vnd.esios-api-v1+json"

# Segons que una petició espera la resposta d'ESIOS.
TIMEOUT = 600

# Temps després de l'últim dia que es llegeix d'un any a partir del qual el seu fitxer
# es dona per complet.
PUBLICATION_MARGIN = datetime.timedelta(days=2)

# Com serveix ESIOS un indicador: el seu número; com n'agrega el servidor cada hora,
# `average` o `sum`, el seu `time_agg`; la geografia que se'n pren i, amb `only`, si ha
# de ser l'única del fitxer o, sense, si es tria entre les que porta; i el nom de
# l'indicador que li fa d'àncora, o `None`.
Indicator = collections.namedtuple("Indicator", "id aggregation geography only anchor")


def years(span):
    """Torna els anys que toca `span`, el primer i l'últim dia que es llegeixen."""
    return range(span[0].year, span[1].year + 1)


def periods(span):
    """Dona, de cada any de `span`, el nom del fitxer, el primer dia i l'últim.

    Dona també l'últim dia que se'n llegeix, el de l'any o el de `span`. Cada any es
    demana sencer.
    """
    for year in years(span):
        last = datetime.date(year, 12, 31)
        yield str(year), datetime.date(year, 1, 1), last, min(last, span[1])


def _is_complete(path, last):
    """Diu si `path` es va baixar prou després de l'últim dia que es llegeix, `last`."""
    if not path.exists():
        return False
    closed = (
        datetime.datetime.combine(
            last + datetime.timedelta(days=1), datetime.time(), MARKET_TIMEZONE
        )
        + PUBLICATION_MARGIN
    )
    return (
        datetime.datetime.fromtimestamp(path.stat().st_mtime, MARKET_TIMEZONE) >= closed
    )


def _json_or_empty(response):
    try:
        return response.json()
    except ValueError:
        return {}


def api_key(name, path=KEYS):
    """Torna el valor de la línia `name=<valor>` del fitxer de claus, `path`.

    Refusa un fitxer que no existeix o que no té la línia.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise Refusal(
            f"no existeix {relative(path)}; crear amb la línia {name}=<clau>"
        ) from None
    for line in lines:
        found, separator, value = line.partition("=")
        if separator and found.strip() == name and value.strip():
            return value.strip()
    raise Refusal(f"{relative(path)} no té la línia {name}=<clau>")


def session_for(key=None):
    """Torna una sessió amb la clau i la capçalera que l'API demana.

    Sense `key`, la clau és la línia `ESIOS_KEY` del fitxer de claus.
    """
    session = requests.Session()
    session.headers.update({"x-api-key": key or api_key("ESIOS_KEY"), "Accept": ACCEPT})
    return session


def missing(folder, span):
    """Torna cada any de `span` que `folder` no té complet, sense demanar res a ESIOS.

    De cada un, el nom del fitxer, el primer dia i l'últim. `span` és el primer i
    l'últim dia que es llegeixen.
    """
    return [
        (name, first, last)
        for name, first, last, read in periods(span)
        if not _is_complete(pathlib.Path(folder) / f"{name}.json", read)
    ]


def download(indicator, folder, span, session=None):
    """Baixa a `folder` cada any d'un indicador que no hi té un fitxer complet.

    Els anys són els de `span`, el primer i l'últim dia que es llegeixen. Desa la
    resposta tal com arriba, amb l'agregació de les hores que l'indicador demana. Torna
    quants fitxers ha escrit i, de cada any que ESIOS ha refusat, el camí i l'estat
    HTTP.
    """
    session = session or session_for()
    folder = pathlib.Path(folder)
    written, refused = 0, []
    for name, first, last in missing(folder, span):
        path = folder / f"{name}.json"
        # ESIOS interpreta les dates en hora local.
        response = session.get(
            URL + str(indicator.id),
            timeout=TIMEOUT,
            params={
                "start_date": f"{first}T00:00",
                "end_date": f"{last}T23:59",
                "time_trunc": "hour",
                "time_agg": indicator.aggregation,
            },
        )
        if response.status_code != 200 or "indicator" not in _json_or_empty(response):
            refused.append((path, response.status_code))
            continue
        folder.mkdir(parents=True, exist_ok=True)
        write_whole(path, response.text)
        written += 1
    return written, refused


def files(folder, span):
    """Torna els fitxers baixats d'un indicador, un per any de `span`, en ordre.

    Només els `<any>.json`, i no el `.part` que hi hagi deixat una baixada interrompuda.
    Refusa una carpeta que no en té cap.
    """
    paths = [
        path
        for path in sorted(pathlib.Path(folder).glob("????.json"))
        if int(path.stem) in years(span)
    ]
    if not paths:
        raise Refusal(
            f'no hi ha cap any baixat a {relative(folder)}/; baixar amb "uv run epf '
            f'download"'
        )
    return paths


def values(paths):
    """Torna tots els valors dels fitxers baixats d'un indicador, `paths`."""
    return [
        value
        for path in paths
        for value in json.loads(path.read_text(encoding="utf-8"))["indicator"]["values"]
    ]


def admissible(values, indicator):
    """Torna els valors de la geografia de l'indicador.

    Amb `only`, refusa uns valors que portin cap altra geografia: vol dir que ESIOS ja
    no serveix l'indicador com diu la taula, i el que se'n triés podria ser una altra
    cosa amb el mateix nom. Sense `only`, tria la geografia d'entre les que porten els
    valors, com els del preu, que en porten sis.
    """
    if indicator.only:
        others = {value["geo_name"] for value in values} - {indicator.geography}
        if others:
            raise Refusal(
                f"altres geografies a més de {indicator.geography}: "
                f"{', '.join(sorted(others))}"
            )
    return [value for value in values if value["geo_name"] == indicator.geography]


def published_instants(values):
    """Torna els instants, en UTC, en què hi ha algun valor."""
    return set(
        pd.to_datetime(
            [value["datetime"] for value in values], format="ISO8601", utc=True
        )
    )


def repair_dst(hours):
    """Torna sempre 24 hores, corregint els dies de canvi d'hora amb mitjanes.

    El dia de 23 hores rep la mitjana de les dues veïnes a l'hora que falta, i el de 25
    fusiona les dues hores repetides en la seva mitjana. Falla amb un dia de qualsevol
    altre nombre d'hores.
    """
    if len(hours) == HOURS_PER_DAY:
        return hours
    if len(hours) == HOURS_PER_DAY - 1:
        inserted = statistics.mean(hours[DST_HOUR - 1 : DST_HOUR + 1])
        return hours[:DST_HOUR] + [inserted] + hours[DST_HOUR:]
    if len(hours) == HOURS_PER_DAY + 1:
        merged = statistics.mean(hours[DST_HOUR : DST_HOUR + 2])
        return hours[:DST_HOUR] + [merged] + hours[DST_HOUR + 2 :]
    raise ValueError(f"un dia de {len(hours)} hores")


def _counted(instants, one, many):
    """Torna `3 valors fora de l'hora en punt, el primer a 2019-05-01 08:15 UTC`.

    `instants` són en UTC, i `one` i `many` el nom en singular i en plural.
    """
    first = f"{instants[0]:%Y-%m-%d %H:%M} UTC"
    if len(instants) == 1:
        return f"1 {one}, a {first}"
    return f"{len(instants)} {many}, el primer a {first}"


def hourly(values, anchor=None):
    """Torna uns valors d'ESIOS com a dies per hores.

    `anchor` són els instants que va publicar l'àncora de l'indicador: una hora absent
    que l'àncora va publicar hi val zero, també abans del primer dia publicat de
    l'indicador i passat l'últim. Sense cap valor, torna buit. Refusa un valor fora de
    l'hora en punt, que la graella deixaria fora sense dir-ho, com tres de cada quatre
    d'un indicador servit per quarts d'hora; i dos valors al mateix instant, que d'una
    sola geografia són una hora repetida i sumats la doblarien.
    """
    if not values:
        return pd.DataFrame(
            columns=HOUR_COLUMNS, index=pd.DatetimeIndex([], name="day")
        )
    # `datetime` i no `datetime_utc`, que etiqueta malament la primera 02:00 del canvi
    # d'hora d'octubre.
    instants = pd.to_datetime(
        [value["datetime"] for value in values], format="ISO8601", utc=True
    )
    off = instants[instants != instants.floor("h")]
    if len(off):
        raise Refusal(
            _counted(
                off, "valor fora de l'hora en punt", "valors fora de l'hora en punt"
            )
        )
    repeated = instants[instants.duplicated()]
    if len(repeated):
        raise Refusal(
            _counted(
                repeated, "instant amb més d'un valor", "instants amb més d'un valor"
            )
        )
    series = pd.Series([value["value"] for value in values], index=instants)

    published = set(series.index.tz_convert(MARKET_TIMEZONE).date)
    first, last = min(published), max(published)
    if anchor:
        anchored = pd.DatetimeIndex(list(anchor)).tz_convert(MARKET_TIMEZONE).date
        first, last = min(first, min(anchored)), max(last, max(anchored))

    start = pd.Timestamp(first).tz_localize(MARKET_TIMEZONE)
    end = pd.Timestamp(last).tz_localize(MARKET_TIMEZONE) + pd.DateOffset(days=1)
    grid = pd.date_range(start, end, freq="h", inclusive="left")
    zero = grid.tz_convert("UTC").isin(list(anchor or ()))
    frame = pd.DataFrame(
        {
            "day": grid.tz_localize(None).normalize(),
            "value": series.reindex(grid.tz_convert("UTC")).to_numpy(),
            "zero": zero,
        }
    )
    frame.loc[frame["zero"], "value"] = frame.loc[frame["zero"], "value"].fillna(0.0)
    zero_days = set(frame.loc[frame["zero"], "day"].dt.date)
    frame = frame[frame["day"].dt.date.isin(published | zero_days)]

    rows = {
        day: repair_dst(hours.tolist()) for day, hours in frame.groupby("day")["value"]
    }
    index = pd.DatetimeIndex(list(rows), name="day")
    return pd.DataFrame(list(rows.values()), index=index, columns=HOUR_COLUMNS)
