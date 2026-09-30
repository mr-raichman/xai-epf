"""Comprovació de coherència de la capa de dades amb la font."""
import collections
import datetime
import sys

import pandas as pd

from epf.config.market import HOURS_PER_DAY, QUARTER_HOURLY_FROM
from epf.config.split import HISTORY_START
from epf.data import esios, prices
from epf.data.calendar import local_hours
from epf.data.omie import latest_version_per_day, load, read_periods

failures = []


def check(description, condition, detail=""):
    mark = "ok  " if condition else "FALLA"
    print(f"{mark} {description}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(description)


chosen = latest_version_per_day()
frame = load()
periods = {day: len(read_periods(path)) for day, path in chosen.items()}
first, last = min(chosen), max(chosen)
expected_days = (last - first).days + 1

check("OMIE comença on comença el seu arxiu", first == prices.OMIE_FROM, f"{first}")
check("cap dia buit entre el primer i l'últim", len(chosen) == expected_days,
      f"{len(chosen)} de {expected_days}, de {first} a {last}")
check("l'índex del DataFrame no repeteix cap dia", frame.index.is_unique)
check("l'índex està ordenat", frame.index.is_monotonic_increasing)
check("cada dia té exactament 24 hores", frame.shape[1] == HOURS_PER_DAY, f"{frame.shape}")
check("cap valor absent després de reparar", not frame.isna().to_numpy().any())

quarter = {day for day, n in periods.items() if n > HOURS_PER_DAY + 1}
check("cap dia quart-horari abans del canvi de mercat",
      all(day >= QUARTER_HOURLY_FROM for day in quarter),
      f"{len(quarter)} dies" + (f", el primer {min(quarter)}" if quarter else ""))
check("tots els dies a partir del canvi són quart-horaris",
      all(n > HOURS_PER_DAY + 1 for day, n in periods.items() if day >= QUARTER_HOURLY_FROM))

repaired = {day for day, n in periods.items() if n % HOURS_PER_DAY != 0}
springs = {day for day in repaired if day.month == 3}
autumns = {day for day in repaired if day.month == 10}
check("els dies reparats són els de canvi d'hora i cap més",
      repaired == springs | autumns and all(day.weekday() == 6 for day in repaired),
      f"{len(springs)} de març, {len(autumns)} d'octubre, tots diumenges")

versions = collections.Counter(int(path.suffix[1:]) for path in chosen.values())
check("s'usa alguna versió superior a la primera", sum(versions.values()) > versions[1],
      f"{dict(sorted(versions.items()))}")

# Dia amb Espanya i Portugal desacoblats: a l'hora 14, -0.60 a Espanya i -0.58 a Portugal.
DECOUPLED_DAY, DECOUPLED_HOUR, SPANISH_PRICE = datetime.date(2024, 6, 1), 14, -0.60
check("es llegeix la columna espanyola i no la portuguesa",
      abs(frame.loc[str(DECOUPLED_DAY)].iloc[DECOUPLED_HOUR] - SPANISH_PRICE) < 1e-9,
      f"{DECOUPLED_DAY} hora {DECOUPLED_HOUR}")

joined = prices.load()
joined_days = pd.date_range(HISTORY_START, joined.index[-1], freq="D")
check("el preu unit comença on comença ESIOS", joined.index[0].date() == HISTORY_START,
      f"{joined.index[0].date()}")
check("el preu unit no té cap dia buit", joined.index.equals(joined_days),
      f"{len(joined)} de {len(joined_days)}, de {HISTORY_START} a {joined.index[-1].date()}")
check("el preu unit té 24 hores i cap valor absent",
      joined.shape[1] == HOURS_PER_DAY and not joined.isna().to_numpy().any(), f"{joined.shape}")
seam = pd.Timestamp(prices.OMIE_FROM)
check("el preu unit és continu al canvi de font",
      {seam - pd.Timedelta(days=1), seam} <= set(joined.index))

spain, spain_hours = esios.load_price(prices.SPAIN)
overlap = frame.index.intersection(spain.index)
largest = (frame.loc[overlap] - spain.loc[overlap]).abs().max().max()
check("el preu espanyol d'ESIOS coincideix amb el d'OMIE", len(overlap) > 0 and largest < 0.01,
      f"{len(overlap)} dies comuns, diferència màxima {largest:.4f}")

# Cada dia del preu d'ESIOS ha d'arribar amb les hores que té el dia local. Si en
# portés de més, alguna cauria fora de la graella i es perdria sense dir-ho, i el
# 2014 a 2017 és l'únic tram del preu sense una segona font que ho destapés.
calendar_hours = local_hours(spain.index)
short = spain.index[spain_hours != calendar_hours]
changes = spain_hours[calendar_hours != HOURS_PER_DAY].value_counts().sort_index()
check("el preu d'ESIOS publica exactament les hores de cada dia local", short.empty,
      f"{len(spain)} dies, als canvis d'hora "
      + " i ".join(f"{count} de {hours} h" for hours, count in changes.items()))

# Hores que ESIOS no publica en una àncora: el PBF de l'apagada del 28 d'abril de 2025.
KNOWN_GAPS = {esios.PBF_WIND: pd.date_range("2025-04-28 12:00", "2025-04-29 23:00", freq="h")}

for anchor in sorted({a for a in esios.SERIES.values() if a is not None}):
    series, _ = esios.load(anchor)
    # El primer i l'últim dia poden ser parcials: l'inici de la publicació i avui.
    interior = series.reindex(pd.date_range(series.index[0], series.index[-1], freq="D")[1:-1])
    days, hours = interior.isna().to_numpy().nonzero()
    absent = interior.index[days] + pd.to_timedelta(hours, unit="h")
    unexpected = absent.difference(KNOWN_GAPS.get(anchor, pd.DatetimeIndex([])))
    check(f"l'àncora {anchor} no té hores buides fora de les conegudes", unexpected.empty,
          f"{len(absent)} hores buides, {len(unexpected)} inesperades, des de {series.index[0].date()}")

print()
print("cobertura de les sèries d'ESIOS: primer dia, últim dia, dies, hores absents")
overfull = []
for indicator in sorted(esios.SERIES):
    try:
        series, published = esios.load(indicator)
    except (FileNotFoundError, ValueError) as error:
        check(f"l'indicador {indicator} es llegeix", False, str(error))
        continue
    if series.empty:
        print(f"  {indicator:>6}  sense cap valor")
        continue
    if (published > local_hours(series.index)).any():
        overfull.append(indicator)
    print(f"  {indicator:>6}  {series.index[0].date()}  {series.index[-1].date()}  "
          f"{len(series):>5}  {int(series.isna().to_numpy().sum()):>6}")

print()
check("cap sèrie publica més hores que les que té el dia local", not overfull,
      f"{len(overfull)} sèries" + (f": {overfull}" if overfull else ""))
if failures:
    print(f"{len(failures)} comprovacions fallides: " + "; ".join(failures))
    sys.exit(1)
print("la capa de dades és coherent amb el que sabem de la font")
