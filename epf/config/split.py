"""Intervals de dades utilitzades."""
import datetime

# Primer dia de l'arxiu públic de preus d'OMIE.
HISTORY_START = datetime.date(2018, 1, 1)

# Dos anys de prova, avaluats per separat: l'horari i el quart-horari promediat.
TEST_PERIODS = {
    "hourly": (datetime.date(2024, 10, 1), datetime.date(2025, 9, 30)),
    "quarter_hourly": (datetime.date(2025, 10, 1), datetime.date(2026, 9, 30)),
}

TRAIN_END = datetime.date(2024, 9, 30)
