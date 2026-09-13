"""El calendari: el que passa als dies de canvi d'hora.

Separat de la lectura de fitxers perquè són dues preguntes diferents. Una és
sobre el format d'OMIE i l'altra sobre el rellotge espanyol.
"""
import statistics

from epf.config.market import DST_HOUR, HOURS_PER_DAY


def repair_dst(hours):
    """Torna sempre 24 hores, reparant els dos dies de canvi horari (decisió 37).

    L'últim diumenge de març el dia té 23 hores perquè l'hora local 02 no
    existeix, i s'hi insereix la mitjana de les dues veïnes. L'últim d'octubre en
    té 25 perquè l'hora 02 se subhasta dues vegades, i les dues es fusionen en la
    seva mitjana. És la pràctica del camp: Lipiecki, Uniejewski i Weron (2024).

    L'índex surt del calendari i no de les dades. OMIE numera els períodes
    correlativament, sense marcar quin falta o quin es repeteix, així que l'hora
    afectada és sempre la tercera del fitxer. No s'ha pogut confirmar amb els
    preus: una hora entre vint-i-quatre no mou prou cap estadístic.
    """
    if len(hours) == HOURS_PER_DAY:
        return hours
    if len(hours) == HOURS_PER_DAY - 1:
        inserted = statistics.mean(hours[DST_HOUR - 1:DST_HOUR + 1])
        return hours[:DST_HOUR] + [inserted] + hours[DST_HOUR:]
    if len(hours) == HOURS_PER_DAY + 1:
        merged = statistics.mean(hours[DST_HOUR:DST_HOUR + 2])
        return hours[:DST_HOUR] + [merged] + hours[DST_HOUR + 2:]
    raise ValueError(f"un dia de {len(hours)} hores, que no és cap cas conegut")
