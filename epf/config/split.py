"""Les nostres eleccions sobre quines dades entren i quan.

Canvia quan canviem d'opinió. Els fets del mercat són a `market.py`.
"""
import datetime

# OMIE no serveix res anterior en cap dels seus directoris de preus (decisió 35).
HISTORY_START = datetime.date(2018, 1, 1)

# Dos anys de prova, reportats per separat i mai sumats en una sola xifra
# (decisió 34): el primer és horari de punta a punta i diu com de bo és el model;
# el segon és l'etapa quart-horària promediada i diu si el model va sobreviure al
# canvi de format del mercat.
TEST_PERIODS = {
    "hourly": (datetime.date(2024, 10, 1), datetime.date(2025, 9, 30)),
    "quarter_hourly": (datetime.date(2025, 10, 1), datetime.date(2026, 9, 30)),
}

TRAIN_END = datetime.date(2024, 9, 30)
