"""El que només es pot preguntar a OMIE i a ESIOS, passat a mà.

El contracte de qualsevol conjunt viu a `contract.py` i el crida el carregador, de
manera que un marc que arriba aquí ja l'ha passat. Això és l'altra meitat: les
versions dels fitxers d'OMIE, el dia desacoblat, les hores que publiquen les previsions
i el dia a què pertanyen, el programa a tots dos costats del canvi de format, les hores
que cada àncora publica i la cobertura de cada indicador, que són fets de la font i no
del conjunt.

Res d'aquí corre en una passada. Informa de tot el que troba en lloc de petar al
primer error, que és el seu ofici, i acaba amb 1 si alguna comprovació ha fallat.
"""
import collections
import datetime

import pandas as pd

from epf.data import esios, forecasts, omie, schedule
from epf.data.esios import QUARTER_HOURLY_FROM, local_hours
from epf.data.outliers import report
from epf.data.price import ESIOS_FIRST_DAY, NAME, OMIE_FROM, from_esios, load
from epf.parameters import DAYS_PER_WEEK, HOURS_PER_DAY


failures = []


def check(description, condition, detail=""):
    mark = "ok  " if condition else "FALLA"
    print(f"{mark} {description}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(description)


# Dia amb Espanya i Portugal desacoblats: a l'hora 14, -0.60 a Espanya i -0.58 a Portugal.
DECOUPLED_DAY, DECOUPLED_HOUR, SPANISH_PRICE = datetime.date(2024, 6, 1), 14, -0.60

# Hores que ESIOS no publica en una àncora: el PBF de l'apagada, que declara el conjunt
# del programa.
KNOWN_GAPS = {esios.PBF_WIND: schedule.BLACKOUT}

# Mínim del programa nuclear de la setmana que comença el 2025-10-01 sobre la d'abans.
# Una caiguda a la quarta part, la de servir l'energia d'un quart d'hora com si fos la de
# l'hora, dona 0,25 i cap caiguda dona 1: el llindar és a mig camí.
SEAM_FLOOR = 0.5


def attempt(description, function, *arguments):
    """El que torna `function`, o `None` amb la comprovació `description` fallida pel seu
    error: l'auditoria informa de tot el que troba, i un error que l'aturés amagaria les
    comprovacions que encara podien córrer."""
    try:
        return function(*arguments)
    except (FileNotFoundError, KeyError, ValueError) as error:
        check(description, False, str(error))
        return None


def check_price():
    """El que només es pot preguntar a les dues fonts del preu. Torna el conjunt, o
    `None` si no s'ha pogut carregar."""
    from_omie = attempt("els fitxers d'OMIE es llegeixen", omie.load)
    if from_omie is not None:
        chosen = omie.latest_version_per_day()
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
              all(n > HOURS_PER_DAY + 1 for day, n in periods.items()
                  if day >= QUARTER_HOURLY_FROM))

        repaired = {day for day, n in periods.items() if n % HOURS_PER_DAY != 0}
        springs = {day for day in repaired if day.month == 3}
        autumns = {day for day in repaired if day.month == 10}
        check("els dies reparats són els de canvi d'hora i cap més",
              repaired == springs | autumns and all(day.weekday() == 6 for day in repaired),
              f"{len(springs)} de març, {len(autumns)} d'octubre, tots diumenges")

        versions = collections.Counter(int(path.suffix[1:]) for path in chosen.values())
        check("s'usa alguna versió superior a la primera",
              sum(versions.values()) > versions[1], f"{dict(sorted(versions.items()))}")

        decoupled = from_omie.loc[str(DECOUPLED_DAY)].iloc[DECOUPLED_HOUR]
        check("es llegeix la columna espanyola i no la portuguesa",
              abs(decoupled - SPANISH_PRICE) < 1e-9, f"{DECOUPLED_DAY} hora {DECOUPLED_HOUR}")

    contract = f"el conjunt {NAME} es carrega i compleix el contracte"
    price = attempt(contract, load)
    if price is not None:
        check(contract, True, f"{price.shape}")
        check("el preu unit comença on comença ESIOS",
              price.index[0].date() == ESIOS_FIRST_DAY, f"{price.index[0].date()}")
        seam = pd.Timestamp(OMIE_FROM)
        check("el preu unit és continu al canvi de font",
              {seam - pd.Timedelta(days=1), seam} <= set(price.index))

    esios_price = attempt("els fitxers d'ESIOS del preu es llegeixen", from_esios)
    if esios_price is not None:
        spain, spain_hours = esios_price
        if from_omie is not None:
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


def check_forecasts():
    """El que només es pot preguntar a ESIOS de les previsions D+1. Torna el conjunt, buit
    si no s'ha pogut carregar."""
    # Com al preu d'ESIOS: una hora fora de la graella es perdria sense dir-ho, i
    # aquestes sèries no tenen cap segona font que ho destapés.
    for name in forecasts.INDICATORS:
        read = attempt(f"la previsió {name} es llegeix", forecasts.read, name)
        if read is None:
            continue
        frame, published = read
        short = frame.index[published != local_hours(frame.index)]
        check(f"la previsió {name} publica exactament les hores de cada dia local",
              short.empty, f"{len(frame)} dies, {len(short)} sense les seves hores")

    contract = f"el conjunt {forecasts.NAME} es carrega i compleix el contracte"
    frames = attempt(contract, forecasts.load)
    if frames is None:
        return {}
    check(contract, True, ", ".join(f"{name} {frame.shape}" for name, frame in frames.items()))

    # Una previsió D+1 és del dia que es lliura i no del dia que es publica. Si ESIOS
    # canviés què vol dir `datetime`, totes es desplaçarien un dia i res no petaria, i
    # només la demanda real ho pot dir: la previsió s'hi ha d'acostar més el mateix dia
    # que el dia abans o el dia després.
    demand = attempt(f"la demanda real, {esios.DEMAND}, es llegeix", esios.load, esios.DEMAND)
    if demand is not None:
        predicted = frames["demand"].mean(axis="columns")
        actual = demand[0].mean(axis="columns")
        errors = {shift: (predicted - actual.shift(shift, freq="D")).abs().mean()
                  for shift in (1, 0, -1)}
        check("la previsió de demanda és del dia de lliurament",
              errors[0] < min(errors[1], errors[-1]),
              f"MAE de les mitjanes diàries contra la demanda real: {errors[0]:.0f} MW el "
              f"mateix dia, {errors[1]:.0f} el dia abans i {errors[-1]:.0f} el dia després")
    return frames


def check_schedule():
    """El que només es pot preguntar a ESIOS del programa. Torna el conjunt, buit si no
    s'ha pogut carregar."""
    anchor = attempt("l'àncora del programa es llegeix", schedule.anchor_instants)
    if anchor is not None:
        # Cada hora que l'àncora no publica queda absent a tot el conjunt, de manera que
        # la regla del zero només és certa si l'àncora les publica totes menys l'apagada.
        # El primer i l'últim dia poden ser parcials: l'inici de la publicació i avui.
        frame, _ = schedule.read(schedule.ANCHOR, anchor)
        interior = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="D")[1:-1])
        days, hours = interior.isna().to_numpy().nonzero()
        absent = interior.index[days] + pd.to_timedelta(hours, unit="h")
        unexpected = absent.difference(schedule.BLACKOUT)
        check("l'àncora del programa publica totes les hores fora de l'apagada",
              unexpected.empty, f"{len(absent)} hores buides, {len(unexpected)} inesperades")

        print()
        print("sèries del programa: primer i últim dia publicat, dies publicats")
        for name, indicators in schedule.GROUPS.items():
            for indicator in indicators:
                read = attempt(f"la sèrie {indicator} del programa es llegeix", schedule.read,
                               indicator, anchor)
                if read is None:
                    continue
                published = read[1][read[1] > 0].index
                print(f"  {indicator:>3}  {name:<9}  {published[0].date()}  "
                      f"{published[-1].date()}  {len(published):>5}")
        print()

    contract = f"el conjunt {schedule.NAME} es carrega i compleix el contracte"
    frames = attempt(contract, schedule.load)
    if frames is None:
        return {}
    check(contract, True, ", ".join(f"{name} {frame.shape}" for name, frame in frames.items()))

    # Des del 2025-10-01 ESIOS serveix el programa per quarts d'hora: si el servidor en
    # fes la mitjana i no la suma, cada hora en valdria la quarta part a partir d'aquell dia.
    seam = pd.Timestamp(QUARTER_HOURLY_FROM)
    week = pd.Timedelta(days=DAYS_PER_WEEK)
    nuclear = frames["nuclear"]
    before = nuclear[(nuclear.index >= seam - week) & (nuclear.index < seam)].mean().mean()
    after = nuclear[(nuclear.index >= seam) & (nuclear.index < seam + week)].mean().mean()
    check("el programa nuclear no cau a la quarta part el 2025-10-01",
          after / before > SEAM_FLOOR,
          f"la setmana que hi comença fa {after / before:.2f} de la d'abans")
    return frames


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
        found = attempt(f"l'indicador {indicator} es llegeix", esios.load, indicator)
        if found is None:
            continue
        series, published = found
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
    if price is not None:
        print()
        report(NAME, price)
    print()
    frames = check_forecasts()
    for name, frame in frames.items():
        print()
        report(f"{forecasts.NAME} {name}", frame)
    print()
    frames = check_schedule()
    for name, frame in frames.items():
        print()
        report(f"{schedule.NAME} {name}", frame)
    print()
    check_esios_series()
    if failures:
        print(f"{len(failures)} comprovacions fallides: " + "; ".join(failures))
        return 1
    print("la capa de dades és coherent amb el que sabem de la font")
    return 0
