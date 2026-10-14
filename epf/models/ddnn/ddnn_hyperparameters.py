"""Els números de les dues xarxes distribucionals i de la xarxa de punt.

**Encara no s'han cercat**: mentre no, cada família fa servir el millor conjunt de les
cerques que Marcjasz et al. (2023) van publicar per al mercat alemany, copiat aquí i
marcat. Es van triar amb les dades d'un altre mercat, de manera que **cap xifra
d'aquestes famílies no diu res fins que tinguin la seva pròpia cerca**.

Els que hi són a més són els que ningú no cerca, tots del codi de l'article: el lot, el
sostre d'èpoques i la paciència.
"""

# Mida del lot, sostre d'èpoques i paciència de l'aturada: els del codi de l'article.
BATCH = 32
EPOCHS = 1500
PATIENCE = 50

# Els números de cada família mentre no s'ha cercat: de les quatre cerques de 2048 proves
# que l'article va publicar per distribució, i per a la xarxa de punt, la de millor
# puntuació pròpia, el fitxer `COPIED_FROM` del seu repositori. Els blocs, amb els noms
# del conjunt de Lago et al. (2021), són els que va triar aquella cerca, i la xarxa no els
# mira: llegeix totes les columnes que se li donen, i per això la de punt no en porta.
# L'L1 dels paràmetres va en l'ordre de `distributions`, i no en el del codi de
# l'article, que posa el pes de la cua de Johnson SU abans de l'asimetria; la xarxa de
# punt no en té cap a la sortida, i el seu zero és una L1 que no fa res.
COPIED = {
    "ddnn_jsu": {
        "blocks": ["day_of_week", "lag1", "demand", "demand_lag7", "wind_solar",
                   "wind_solar_lag1"],
        "hidden": [243, 895], "activation": ["softplus", "elu"], "dropout": 0.0,
        "l1": [0.0, 0.0], "activity_l1": [0.0, 0.0], "parameter_l1": [0.0, 0.0, 0.0, 0.0],
        "learning_rate": 0.000281493495779889},
    "ddnn_normal": {
        "blocks": ["lag1", "lag2", "lag7", "demand", "demand_lag7", "wind_solar",
                   "wind_solar_lag1"],
        "hidden": [110, 823], "activation": ["softplus", "relu"], "dropout": 0.0,
        "l1": [0.0, 0.0], "activity_l1": [1.0076687725977794e-05, 0.0],
        "parameter_l1": [0.0, 0.0], "learning_rate": 0.00032041091802416773},
    "dnn": {
        "hidden": [979, 767], "activation": ["softplus", "relu"], "dropout": 0.0,
        "l1": [0.0, 0.0], "activity_l1": [0.0, 0.0], "parameter_l1": [0.0],
        "learning_rate": 0.0005252223827375965},
}
COPIED_FROM = {"ddnn_jsu": "FINAL_DE_selection_prob_jsu3",
               "ddnn_normal": "FINAL_DE_selection_prob_normal4",
               "dnn": "FINAL_DE_selection_prob_point3"}
COPIED_TRIALS = 2048
