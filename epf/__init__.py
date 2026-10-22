"""Aportació a l'heurística apresa en la predicció del preu del mercat elèctric diari
espanyol mitjançant aprenentatge automàtic."""


class Refusal(ValueError):
    """El refús d'una comanda que no pot continuar, o que acaba amb tasques fallades.

    El motiu va al missatge, i el causen les dades, els paràmetres, el que s'ha teclejat
    o les fallades, que ja s'han mostrat. La línia de comandes l'escriu com a `error:
    <missatge>`; qualsevol altra excepció és un error del codi, i en mostra el
    traceback.
    """


class Declined(Exception):
    """L'aturada d'una comanda perquè qui la fa ha respost que no.

    La línia de comandes l'escriu com `Aturat`, com un Ctrl+C, però hi acaba com un
    refús.
    """
