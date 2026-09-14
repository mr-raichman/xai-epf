"""Correcció dels dies de canvi d'hora."""
import statistics

from epf.config.market import DST_HOUR, HOURS_PER_DAY


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
