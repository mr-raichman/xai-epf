"""Els números de LEAR: les finestres de calibratge i el sostre d'iteracions."""
from epf.parameters import DAYS_PER_WEEK, WARM_UP

# Finestres de calibratge de LEAR, en dies de calibratge. Els "anys" de l'article
# són de 364 dies, 52 setmanes exactes que mantenen alineat el dia de la setmana;
# les seves finestres de 1092 i 1456 dies de preus donen aquests dies un cop
# descomptat l'escalfament. None vol dir tot l'historial, que l'article no fa
# servir i que hi és per posar a prova la decisió d'arribar fins al 2014.
EIGHT_WEEKS = 8 * DAYS_PER_WEEK - WARM_UP
TWELVE_WEEKS = 12 * DAYS_PER_WEEK - WARM_UP
WINDOWS = (EIGHT_WEEKS, TWELVE_WEEKS, 364 * 3 - WARM_UP, 364 * 4 - WARM_UP, None)

# Retards que porta cada finestra curta. Amb els quatre, la matriu de disseny té
# 103 columnes i LassoLarsIC demana dues files més que columnes, que cap de les
# dues finestres curtes no té: cada una porta els que hi caben. Si es canvien les
# finestres i la parella queda incoherent, LEAR falla en comptar les files.
SHORT_LAGS = {EIGHT_WEEKS: (1,), TWELVE_WEEKS: (1, 2)}

# El sostre d'iteracions del descens de coordenades és 10000 i no 2500: mesurat el
# 2026-10-01 sobre el 2022-09-01, a 2500 la finestra de 1085 dies deixava 13 dels
# seus 24 ajusts per hora sense convergir i la de 1449 en deixava 10, i la predicció
# d'aquell dia es movia fins a 0,566 EUR/MWh. A 50000 i a 200000 surt bit a bit
# igual que a 10000.
MAX_ITER = 10000
