"""Les capes de la xarxa de Lago et al. (2021).

Separades de la família que les fa servir perquè la xarxa distribucional de Marcjasz et
al. (2023), de dos dels mateixos autors, és aquesta mateixa xarxa amb la sortida
redefinida com una distribució. El cap i la pèrdua hi arriben com a arguments, de
manera que aquella família és un cap i una pèrdua i no una còpia de la xarxa.

Les capes són les del codi de l'article, que és el que va produir les seves xifres.
"""
import keras

# Les activacions que el codi de Lago et al. (2021) posa com una capa a part, darrere
# d'una `Dense` lineal, amb el pendent negatiu del seu `LeakyReLU`.
SEPARATE = {"LeakyReLU": lambda: keras.layers.LeakyReLU(negative_slope=0.001),
            "PReLU": keras.layers.PReLU}

# L'activació amb la qual el codi de l'article apaga neurones amb `AlphaDropout` i no
# amb `Dropout`, el que manté la mitjana i la variància que `selu` necessita.
SELU = "selu"


def build(columns, hidden, activation, dropout, batch_normalisation, l1, initialiser,
          head, loss, learning_rate, clip=None):
    """La xarxa amagada, amb el cap i la pèrdua que hi posa qui la crida.

    L'ordre de cada capa és el del codi de l'article: `Dense` amb l'activació a dins, o
    lineal i l'activació darrere si és de `SEPARATE`, després la normalització per lots
    si hi és, i el dropout al final. **La normalització va després de l'activació i no
    abans**, que és el que diu la seva implementació i el contrari del que se'n diria de
    memòria.

    `hidden`, `activation` i `l1` van per capa: la xarxa de punt hi passa la mateixa a
    totes, com el codi de Lago et al. (2021), i la distribucional en pot passar una per
    capa. `head` rep l'última capa amagada i torna la sortida. `clip` és el valor màxim
    de cada component del gradient, o cap.
    """
    inputs = keras.Input(shape=(columns,))
    tensor = inputs
    for neurons, chosen, coefficient in zip(hidden, activation, l1, strict=True):
        tensor = keras.layers.Dense(
            neurons, activation=None if chosen in SEPARATE else chosen,
            kernel_initializer=initialiser,
            kernel_regularizer=keras.regularizers.L1(coefficient))(tensor)
        if chosen in SEPARATE:
            tensor = SEPARATE[chosen]()(tensor)
        if batch_normalisation:
            tensor = keras.layers.BatchNormalization()(tensor)
        if dropout > 0:
            switch = keras.layers.AlphaDropout if chosen == SELU else keras.layers.Dropout
            tensor = switch(dropout)(tensor)
    model = keras.Model(inputs, head(tensor))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                                                  clipvalue=clip),
                  loss=loss)
    return model
