"""Predicció del preu diari del mercat elèctric espanyol i estudi de la seva
formació."""


class Refusal(ValueError):
    """El refús d'una ordre: les dades, els paràmetres o el que s'ha teclejat no li
    permeten continuar, i el missatge en diu el motiu.

    La línia d'ordres l'escriu com a `error: <missatge>`; qualsevol altra excepció és un
    error del codi i en mostra la traça.
    """
