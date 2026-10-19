"""Aportació a l'heurística apresa en la predicció del preu del mercat elèctric diari
espanyol mitjançant aprenentatge automàtic."""


class Refusal(ValueError):
    """El refús d'una comanda que no pot continuar, amb el motiu al missatge.

    El causen les dades, els paràmetres o el que s'ha teclejat. La línia de comandes
    l'escriu com a `error: <missatge>`; qualsevol altra excepció és un error del codi, i
    en mostra el traceback.
    """
