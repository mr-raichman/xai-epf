"""Paràmetres fixos del mercat diari espanyol."""
import datetime
import zoneinfo

MARKET_TIMEZONE = zoneinfo.ZoneInfo("Europe/Madrid")

# Tancament de la subhasta, el dia anterior al lliurament. Només són admissibles
# les variables publicades abans d'aquest instant.
GATE_CLOSURE = datetime.time(12, 0)

# Primer dia de lliurament amb 96 productes quart-horaris en lloc de 24 horaris.
QUARTER_HOURLY_FROM = datetime.date(2025, 10, 1)
QUARTERS_PER_HOUR = 4

HOURS_PER_DAY = 24

# Hora local que falta l'últim diumenge de març i es repeteix l'últim d'octubre.
DST_HOUR = 2
