"""La xarxa de Lago et al. (2021): les capes i la partició de validació.

Separada de la família que la fa servir perquè la xarxa distribucional de Marcjasz et
al. (2023), de dos dels mateixos autors, és aquesta mateixa xarxa amb la sortida
redefinida com una distribució. El cap i la pèrdua hi arriben com a arguments, de
manera que aquella família serà un cap i una pèrdua i no una còpia de la xarxa.

El que s'ha pres de l'article són fets de l'arquitectura i no codi: quantes capes,
en quin ordre, i que la pèrdua és l'absoluta. Està escrit aquí com està escrit LEAR,
de la descripció i no de la seva implementació.
"""
import keras

from epf.parameters import DAYS_PER_WEEK


def split(count, weeks, generator):
    """Índexs d'entrenament i de validació d'una finestra de `count` dies.

    L'article aparta setmanes de la finestra de calibratge per aturar l'entrenament
    quan deixa de millorar, i a l'hora de predir les tria **barrejant tota la
    finestra** en lloc de prendre'n el tram final, de manera que la xarxa s'entrena
    també amb els dies més recents. Per triar hiperparàmetres, en canvi, l'article
    pren les últimes setmanes en ordre; aquesta funció fa la primera cosa i la cerca
    farà la segona quan hi arribi.
    """
    days = weeks * DAYS_PER_WEEK
    if days >= count:
        raise ValueError(f"la finestra de {count} dies no dóna {weeks} setmanes de "
                         "validació i prou dies per entrenar")
    order = generator.permutation(count)
    return order[days:], order[:days]


def build(columns, hidden, activation, dropout, batch_normalisation, l1, initialiser,
          head, loss, learning_rate):
    """La xarxa amagada, amb el cap i la pèrdua que hi posa qui la crida.

    L'ordre de cada capa és el de l'article: `Dense` amb l'activació a dins, després
    la normalització per lots si hi és, i el dropout al final. **La normalització va
    després de l'activació i no abans**, que és el que diu la seva implementació i el
    contrari del que se'n diria de memòria.

    `l1` és un coeficient per capa, perquè l'article en té un per capa i no un per
    xarxa. `head` rep l'última capa amagada i torna la sortida.

    Les activacions `LeakyReLU` i `PReLU` no hi caben: a l'article són una capa a part
    darrere d'una `Dense` lineal. Hi entraran amb la cerca, que és qui les pot triar;
    mentre no hi siguin, passar-les aquí dóna una xarxa que no és la que l'article
    descriu.
    """
    inputs = keras.Input(shape=(columns,))
    tensor = inputs
    for neurons, coefficient in zip(hidden, l1, strict=True):
        tensor = keras.layers.Dense(
            neurons, activation=activation, kernel_initializer=initialiser,
            kernel_regularizer=keras.regularizers.L1(coefficient))(tensor)
        if batch_normalisation:
            tensor = keras.layers.BatchNormalization()(tensor)
        if dropout > 0:
            tensor = keras.layers.Dropout(dropout)(tensor)
    model = keras.Model(inputs, head(tensor))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss)
    return model
