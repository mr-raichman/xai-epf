"""Fets sobre el mercat, no decisions nostres sobre ell.

Aquest fitxer canvia quan canvia Espanya. Les nostres eleccions viuen a
`split.py`, que canvia quan canviem d'opinió.
"""
import datetime
import zoneinfo

MARKET_TIMEZONE = zoneinfo.ZoneInfo("Europe/Madrid")

# La subhasta del mercat diari tanca a les 12:00 CET del dia anterior al lliurament.
# Una variable només és admissible si ja era pública abans d'aquest instant
# (decisió 42): l'expectativa que tenien els agents és el que va formar el preu.
GATE_CLOSURE = datetime.time(12, 0)

# Des d'aquest dia de lliurament el mercat casa 96 productes quart-horaris en lloc
# de 24 d'horaris, en un canvi europeu coordinat. Els quatre preus de cada hora es
# promedien per mantenir 24 valors diaris (decisió 29).
QUARTER_HOURLY_FROM = datetime.date(2025, 10, 1)
QUARTERS_PER_HOUR = 4

HOURS_PER_DAY = 24

# L'hora local que desapareix l'últim diumenge de març i es repeteix l'últim
# d'octubre. OMIE numera els períodes correlativament, així que és sempre la tercera.
DST_HOUR = 2
