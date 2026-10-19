"""Predicció del preu diari del mercat elèctric espanyol i estudi de la seva
formació."""


class Refusal(ValueError):
    """El refús d'una ordre que no pot continuar, amb el motiu al missatge.

    El causen les dades, els paràmetres o el que s'ha teclejat. La línia d'ordres
    l'escriu com a `error: <missatge>`; qualsevol altra excepció és un error del codi, i
    en mostra la traça.
    """
