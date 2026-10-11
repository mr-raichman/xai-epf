"""Els números de LEAR: les finestres de calibratge i el sostre d'iteracions."""
from epf.parameters import DAYS_PER_WEEK, WARM_UP

# Les quatre finestres de calibratge de Lago et al. (2021), en dies de calibratge:
# vuit setmanes, dotze, tres anys i quatre. Els "anys" de l'article són de 364 dies, 52
# setmanes exactes que mantenen alineat el dia de la setmana, i les seves finestres
# compten dies de preus, de manera que aquí perden l'escalfament dels retards.
EIGHT_WEEKS = 8 * DAYS_PER_WEEK - WARM_UP
TWELVE_WEEKS = 12 * DAYS_PER_WEEK - WARM_UP
WINDOWS = (EIGHT_WEEKS, TWELVE_WEEKS, 364 * 3 - WARM_UP, 364 * 4 - WARM_UP)

# El sostre d'iteracions del descens de coordenades és 10000 i no 2500: mesurat el
# 2026-10-01 sobre el 2022-09-01, a 2500 la finestra de 1085 dies deixava 13 dels
# seus 24 ajusts per hora sense convergir i la de 1449 en deixava 10, i la predicció
# d'aquell dia es movia fins a 0,566 EUR/MWh. A 50000 i a 200000 surt bit a bit
# igual que a 10000.
MAX_ITER = 10000
