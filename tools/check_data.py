"""Comprovació de coherència de la capa de dades. És una execució de tempteig.

No mesura res i no imprimeix cap xifra que es pugui confondre amb un resultat
(decisió 30): només pregunta si el que surt del carregador quadra amb el que
sabem de la font. Falla amb soroll, mai en silenci.
"""
import collections
import datetime
import sys

from epf.config.market import HOURS_PER_DAY, QUARTER_HOURLY_FROM
from epf.config.split import HISTORY_START
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

check("comença on comença l'arxiu d'OMIE", first == HISTORY_START, f"{first}")
check("cap dia buit entre el primer i l'últim", len(chosen) == expected_days,
      f"{len(chosen)} de {expected_days}, de {first} a {last}")
check("l'índex del DataFrame no repeteix cap dia", frame.index.is_unique)
check("l'índex està ordenat", frame.index.is_monotonic_increasing)
check("cada dia té exactament 24 hores", frame.shape[1] == HOURS_PER_DAY, f"{frame.shape}")
check("cap valor absent després de reparar", not frame.isna().to_numpy().any())

quarter = {day for day, n in periods.items() if n > HOURS_PER_DAY + 1}
check("cap dia quart-horari abans del canvi de mercat",
      all(day >= QUARTER_HOURLY_FROM for day in quarter),
      f"{len(quarter)} dies, el primer {min(quarter)}")
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

# Un dia normal amb Espanya i Portugal desacoblats: a l'hora 14 el fitxer porta
# -0.58 per a Portugal i -0.60 per a Espanya, així que llegir la columna que no
# toca es veuria aquí (decisió 43).
DECOUPLED_DAY, DECOUPLED_HOUR, SPANISH_PRICE = datetime.date(2024, 6, 1), 14, -0.60
check("es llegeix la columna espanyola i no la portuguesa",
      abs(frame.loc[str(DECOUPLED_DAY)].iloc[DECOUPLED_HOUR] - SPANISH_PRICE) < 1e-9,
      f"{DECOUPLED_DAY} hora {DECOUPLED_HOUR}")

print()
if failures:
    print(f"{len(failures)} comprovacions fallides: " + "; ".join(failures))
    sys.exit(1)
print("la capa de dades és coherent amb el que sabem de la font")
