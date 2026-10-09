"""Els números de la xarxa.

**Cap d'aquests valors s'ha triat i cap no es pot informar.** L'article els tria tots
amb una cerca per estimador de Parzen sobre el període de validació, i aquesta és la
feina que ve després; aquí hi són perquè una xarxa es pugui construir i cronometrar.
El sostre d'aquesta simplificació: fins que la cerca hi sigui, l'error d'aquesta
família no diu res de la xarxa de l'article, només del temps que costa ajustar-la.

La finestra i les setmanes de validació sí que són de l'article: quatre anys de
calibratge, de 364 dies com els seus, i 42 setmanes apartades per aturar
l'entrenament.
"""
from epf.parameters import WARM_UP

# Quatre anys de calibratge, que és el que fa l'article. Els seus "anys" són de 364
# dies, 52 setmanes exactes que mantenen alineat el dia de la setmana, i el que es
# compta aquí són dies de calibratge un cop descomptat l'escalfament, igual que a
# LEAR.
WINDOW = 364 * 4 - WARM_UP

# Setmanes de la finestra que s'aparten per aturar l'entrenament quan deixa de
# millorar. De l'article.
VALIDATION_WEEKS = 42

# Dues capes amagades, que és l'estructura de l'article un cop descomptades l'entrada
# i la sortida. Quantes neurones té cada una és un dels hiperparàmetres que es
# cercaran.
HIDDEN = (128, 128)

# Un coeficient L1 per capa, com l'article, i no un per xarxa.
L1 = (1e-5, 1e-5)

ACTIVATION = "relu"
INITIALISER = "glorot_uniform"
DROPOUT = 0.1
BATCH_NORMALISATION = True
LEARNING_RATE = 1e-3

# Mida del lot i sostre d'èpoques. La paciència és la de l'article en recalibrar, 20
# èpoques sense millorar, i el sostre d'èpoques també, de manera que qui talla és
# l'aturada anticipada i no el sostre.
BATCH = 32
EPOCHS = 1000
PATIENCE = 20
