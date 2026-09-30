"""Períodes de dades que utilitza el projecte."""
import datetime

# Primer dia del preu espanyol a ESIOS, el més antic que es pot utilitzar.
HISTORY_START = datetime.date(2014, 1, 1)

# Dos anys de validació abans de la prova, escurçats per model quan les seves
# dades no hi arriben. S'hi trien els hiperparàmetres i s'hi comparen les
# famílies; els anys de prova no s'hi toquen.
VALIDATION = (datetime.date(2022, 10, 1), datetime.date(2024, 9, 30))

# Últim dia d'entrenament, el dia abans de la validació. On comença depèn de cada
# model: la llargada de la finestra de calibratge és un paràmetre que s'ajusta.
TRAIN_END = datetime.date(2022, 9, 30)

# Dos anys de prova, avaluats per separat: l'horari i el quart-horari promediat.
TEST_PERIODS = {
    "hourly": (datetime.date(2024, 10, 1), datetime.date(2025, 9, 30)),
    "quarter_hourly": (datetime.date(2025, 10, 1), datetime.date(2026, 9, 30)),
}
