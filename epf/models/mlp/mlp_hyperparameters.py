"""Els números de l'MLP.

Dues capes amagades, que és l'estructura de la xarxa de l'article un cop
descomptades l'entrada i la sortida, i quatre xarxes mitjanades, que és el que fa
l'article perquè una xarxa és sensible a la llavor. El sostre d'èpoques és el de
scikit-learn: mesurat sobre sis finestres reals de 1085 i 1449 dies i dues llavors,
l'aturada anticipada salta entre l'època 33 i la 126, de manera que 200 no talla
cap ajust.
"""
WINDOW = 1085
HIDDEN = (128, 128)
MEMBERS = 4
ALPHA = 1e-4
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.1
NO_CHANGE = 10
MAX_ITER = 200
