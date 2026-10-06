"""Hores del dia i correcció dels dies de canvi d'hora."""
import statistics

import pandas as pd

from epf.market import DST_HOUR, HOURS_PER_DAY, MARKET_TIMEZONE

HOUR_COLUMNS = [f"h{hour:02d}" for hour in range(HOURS_PER_DAY)]


def local_hours(days):
    """Hores que té realment cada dia local: 23 el del canvi de març, 25 el d'octubre."""
    starts = pd.DatetimeIndex(days).tz_localize(MARKET_TIMEZONE)
    return pd.Series((starts + pd.DateOffset(days=1) - starts) // pd.Timedelta(hours=1),
                     index=days)


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
