"""Tota comprovació de les dades, i l'única que n'hi ha.

Dos nivells, tots dos sencers i cap dels dos saltable, perquè una comprovació que
es pot saltar no és una comprovació:

`refuse_incoherent` és el contracte que qualsevol conjunt de dades ha de complir,
i és el que una execució crida abans de començar. No sap de quina font ve res.

`main` és la passada sencera, feta a mà, que hi afegeix el que només es pot
preguntar a OMIE i a ESIOS: les versions dels fitxers, el dia desacoblat, les hores
que cada àncora publica i la cobertura de cada indicador.

`report` no comprova res, informa: imprimeix les puntuacions de veïnatge més altes
amb la finestra de cada una. No hi ha cap llindar, i no n'hi pot haver: la
puntuació és distància partit per MAD, i els seus extrems en aquesta sèrie són
moviments ordinaris dins de finestres de preu gairebé pla, no moviments grans.
"""
import collections
import datetime
import sys

import numpy as np
import pandas as pd

from epf.calendar import local_hours
from epf.data import esios, omie
from epf.data.price import (ESIOS_FIRST_DAY, OMIE_FROM, SPAIN,
                            from_esios, load)
from epf.data.outliers import hampel_score, window_median_mad
from epf.market import HOURS_PER_DAY, QUARTER_HOURLY_FROM

# Veïns d'un preu: les hores adjacents de la sèrie cronològica, i la mateixa hora en
# els dies del voltant.
HOUR_HALF_WIDTH = 12
DAY_HALF_WIDTH = 7

# Quantes de les puntuacions més altes s'informen. És quantes files s'imprimeixen i
# no un llindar: no decideix res sobre cap valor.
HIGHEST = 100


def refuse_incoherent(name, frame):
    """El contracte de qualsevol conjunt de dades. Torna el marc perquè es compongui.

    Cinc condicions, i cap d'elles és de cap font en concret: un dia repetit, un dia
    fora d'ordre, un dia que falta al mig, una graella que no és d'un dia sencer o un
    valor absent. Totes cinc produeixen un número plausible més endavant en lloc de
    petar, que és per això que es comproven aquí i no es deixen al model.
    """
    def refuse(condition, detail):
        if condition:
            raise ValueError(f"el conjunt {name} no és coherent: {detail}")

    refuse(not frame.index.is_unique,
           f"{len(frame) - frame.index.nunique()} dies repetits a l'índex")
    refuse(not frame.index.is_monotonic_increasing, "l'índex no està ordenat")
    expected = pd.date_range(frame.index[0], frame.index[-1], freq="D")
    missing = expected.difference(frame.index)
    refuse(len(missing) > 0,
           f"{len(missing)} dies buits entre {frame.index[0].date()} i "
           f"{frame.index[-1].date()}, el primer {missing[0].date() if len(missing) else ''}")
    refuse(frame.shape[1] != HOURS_PER_DAY,
           f"{frame.shape[1]} columnes i no {HOURS_PER_DAY}")
    absent = int(frame.isna().to_numpy().sum())
    refuse(absent > 0, f"{absent} valors absents")
    return frame


def highest(scores):
    """Posicions de les `HIGHEST` puntuacions més altes, sense comptar les NaN."""
    flat = np.nan_to_num(scores.ravel(), nan=-np.inf)
    return np.argsort(flat)[-HIGHEST:]


def _describe(days, values, label, scores, median, mad):
    """Resum de les puntuacions més altes: com és la finestra i on cauen."""
    marks = highest(scores)
    years = pd.Series(days.repeat(HOURS_PER_DAY)[marks].year).value_counts().sort_index()
    print(f"{label}, les {HIGHEST} puntuacions més altes:")
    print(f"  MAD de la finestra, mediana {np.median(mad[marks]):.2f}, "
          f"màxim {mad[marks].max():.2f}")
    print(f"  mediana de la finestra, mediana {np.median(median[marks]):.2f}")
    print(f"  distància del valor a la mediana, mediana "
          f"{np.median(np.abs(values - median)[marks]):.2f}")
    print("  per any: " + ", ".join(f"{year} {count}" for year, count in years.items()))


def report(name, frame):
    """Imprimeix les puntuacions de veïnatge més altes i les torna, per si es dibuixen.

    Les dues vistes: contra les hores adjacents de la sèrie cronològica, i contra la
    mateixa hora dels dies del voltant.
    """
    values = frame.to_numpy().ravel()
    by_hours = hampel_score(values, HOUR_HALF_WIDTH).reshape(frame.shape)
    by_days = np.column_stack([hampel_score(frame[column].to_numpy(), DAY_HALF_WIDTH)
                               for column in frame.columns])

    hour_median, hour_mad = window_median_mad(values, HOUR_HALF_WIDTH)
    day_parts = [window_median_mad(frame[column].to_numpy(), DAY_HALF_WIDTH)
                 for column in frame.columns]
    print(f"veïnatge de {name}:")
    _describe(frame.index, values, f"hores adjacents (±{HOUR_HALF_WIDTH} h)", by_hours,
              hour_median, hour_mad)
    _describe(frame.index, values, f"mateixa hora (±{DAY_HALF_WIDTH} dies)", by_days,
              np.column_stack([part[0] for part in day_parts]).ravel(),
              np.column_stack([part[1] for part in day_parts]).ravel())
    return by_hours, by_days


# ---------------------------------------------------- la passada sencera, a mà

failures = []


def check(description, condition, detail=""):
    mark = "ok  " if condition else "FALLA"
    print(f"{mark} {description}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(description)


# Dia amb Espanya i Portugal desacoblats: a l'hora 14, -0.60 a Espanya i -0.58 a Portugal.
DECOUPLED_DAY, DECOUPLED_HOUR, SPANISH_PRICE = datetime.date(2024, 6, 1), 14, -0.60

# Hores que ESIOS no publica en una àncora: el PBF de l'apagada del 28 d'abril de 2025.
KNOWN_GAPS = {esios.PBF_WIND: pd.date_range("2025-04-28 12:00", "2025-04-29 23:00", freq="h")}


def check_price():
    """El que només es pot preguntar a les dues fonts del preu."""
    chosen = omie.latest_version_per_day()
    from_omie = omie.load()
    periods = {day: len(omie.read_periods(path)) for day, path in chosen.items()}
    first, last = min(chosen), max(chosen)
    expected_days = (last - first).days + 1

    check("OMIE comença on comença el seu arxiu", first == OMIE_FROM, f"{first}")
    check("cap dia buit entre el primer i l'últim", len(chosen) == expected_days,
          f"{len(chosen)} de {expected_days}, de {first} a {last}")

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

    check("es llegeix la columna espanyola i no la portuguesa",
          abs(from_omie.loc[str(DECOUPLED_DAY)].iloc[DECOUPLED_HOUR] - SPANISH_PRICE) < 1e-9,
          f"{DECOUPLED_DAY} hora {DECOUPLED_HOUR}")

    price = load()
    check("el preu unit comença on comença ESIOS",
          price.index[0].date() == ESIOS_FIRST_DAY, f"{price.index[0].date()}")
    seam = pd.Timestamp(OMIE_FROM)
    check("el preu unit és continu al canvi de font",
          {seam - pd.Timedelta(days=1), seam} <= set(price.index))

    spain, spain_hours = from_esios(SPAIN)
    overlap = from_omie.index.intersection(spain.index)
    largest = (from_omie.loc[overlap] - spain.loc[overlap]).abs().max().max()
    check("el preu espanyol d'ESIOS coincideix amb el d'OMIE",
          len(overlap) > 0 and largest < 0.01,
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
    return price


def check_esios_series():
    """El que només es pot preguntar als indicadors que encara no són cap conjunt."""
    for anchor in sorted({a for a in esios.SERIES.values() if a is not None}):
        series, _ = esios.load(anchor)
        # El primer i l'últim dia poden ser parcials: l'inici de la publicació i avui.
        interior = series.reindex(pd.date_range(series.index[0], series.index[-1], freq="D")[1:-1])
        days, hours = interior.isna().to_numpy().nonzero()
        absent = interior.index[days] + pd.to_timedelta(hours, unit="h")
        unexpected = absent.difference(KNOWN_GAPS.get(anchor, pd.DatetimeIndex([])))
        check(f"l'àncora {anchor} no té hores buides fora de les conegudes", unexpected.empty,
              f"{len(absent)} hores buides, {len(unexpected)} inesperades, "
              f"des de {series.index[0].date()}")

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


def main():
    price = check_price()
    try:
        refuse_incoherent("price", price)
        check("el conjunt price compleix el contracte", True, f"{price.shape}")
    except ValueError as error:
        check("el conjunt price compleix el contracte", False, str(error))
    print()
    report("price", price)
    print()
    check_esios_series()
    if failures:
        print(f"{len(failures)} comprovacions fallides: " + "; ".join(failures))
        return 1
    print("la capa de dades és coherent amb el que sabem de la font")
    return 0


if __name__ == "__main__":
    sys.exit(main())
