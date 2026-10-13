"""Tot el que es tecleja a mà, en una sola ordre amb subordres.

`epf --help` és la llista de què es pot teclejar. Aquest és l'únic lloc que sap quines
subordres hi ha, com es diuen, quins arguments prenen i quin mòdul n'executa cada una,
de manera que afegir-ne una és una fila de `COMMANDS` i res més.

Els mòduls s'importen quan es tria la subordre i no abans. Importar el camí endavant
costa 2,1 s perquè `models/boosting/boosting.py` porta `shap`, i amb els imports a
dalt els pagarien `epf show-run`, que en costa 0,4, i fins i tot `epf --help`. Per això
a la taula hi ha el nom del mòdul i no el mòdul, i per això els arguments del camí
endavant es declaren aquí: `argparse` construeix totes les subordres abans de llegir-ne
cap, així que demanar-los a `runner.py` el tornaria a importar sempre.

`python -m epf.runner` segueix arrencant una execució, perquè és el que executa
l'embolcall de fora del projecte, i `runner.parse()` demana els arguments a
`run_arguments`, de manera que les dues maneres surten d'una sola declaració.
"""
import argparse
import collections
import importlib
import pathlib

from epf import parameters
from epf.models.dnn.dnn_hyperparameters import MEMBERS

# Una subordre: quin mòdul l'executa, què en diu `epf --help` i qui li declara els
# arguments. `arguments` és `None` quan no en pren cap, i és també el que diu que el
# seu `main` es crida sense res.
Command = collections.namedtuple("Command", "module help arguments")

# Les famílies que `epf search` cerca, i el mòdul que en fa la cerca. Com a `COMMANDS`, el
# nom del mòdul i no el mòdul: la cerca de la xarxa porta TensorFlow.
SEARCHES = {"dnn": "epf.models.dnn.search", "lightgbm": "epf.models.boosting.search",
            "elastic_net": "epf.models.linear.search"}


def pass_arguments(parser):
    """Els arguments d'una passada, els mateixos pel camí endavant i per la part B: el
    nivell, quines famílies i si es comença de zero."""
    parser.add_argument("--tier", choices=(parameters.TEST, parameters.MEASUREMENT),
                        required=True,
                        help="de prova, que no dóna cap xifra, o de mesura, que sí")
    parser.add_argument("--family", action="append",
                        help="només aquesta família; es pot repetir")
    parser.add_argument("--replace", action="store_true",
                        help="comença de zero, encara que hi hagi una execució a mig fer")
    return parser


def inputs_argument(parser):
    """El conjunt d'informació, obligatori encara que n'hi hagi un de sol, de manera que
    ni una execució ni una cerca no endevinin mai sobre quin conjunt treballen."""
    parser.add_argument("--inputs", choices=parameters.INPUTS, required=True,
                        help="conjunt d'informació: lago, el de Lago et al. (2021)")
    return parser


def run_arguments(parser):
    """Els arguments del camí endavant, els mateixos per `epf run` i per `python -m`."""
    pass_arguments(parser)
    return inputs_argument(parser)


def search_arguments(parser):
    """La família, el conjunt, el membre i les proves de la cerca d'hiperparàmetres. Cada
    membre de la xarxa és una cerca pròpia, i l'arbre no en té; el nombre de membres i el
    de proves viuen amb cada família i no aquí."""
    parser.add_argument("--family", choices=tuple(SEARCHES), required=True,
                        help="la família que es cerca")
    inputs_argument(parser)
    parser.add_argument("--member", type=int, choices=range(MEMBERS),
                        help="quin membre de la xarxa es cerca; per a la xarxa, obligatori")
    parser.add_argument("--trials", type=int,
                        help="proves que es fan; per defecte, les del seu fitxer, o la "
                             "graella sencera")
    return parser


def folder_arguments(parser):
    """El directori d'una execució, que és el que prenen les quatre que en llegeixen una."""
    parser.add_argument("folder", type=pathlib.Path, nargs="+",
                        help="directori d'una execució o d'una de les seves famílies")
    return parser


COMMANDS = {
    "download-price": Command(
        "epf.data.price", "baixa el preu de les dues fonts i diu què hi ha al disc", None),
    "download-forecasts": Command(
        "epf.data.forecasts", "baixa les previsions D+1 i diu què hi ha al disc", None),
    "audit-data": Command(
        "epf.data.audit", "el que només es pot preguntar a OMIE i a ESIOS", None),
    "figure-price": Command(
        "epf.data.figures.price.price", "tots els preus horaris, del 2014 endavant", None),
    "figure-outliers": Command(
        "epf.data.figures.price.outliers", "la distància de cada preu als seus veïns", None),
    "search": Command(
        None, "cerca els hiperparàmetres d'una família de la part A", search_arguments),
    "run": Command(
        "epf.runner", "camina el període endavant amb totes les famílies", run_arguments),
    "describe": Command(
        "epf.describe", "descriu el preu amb el protocol de Trebbien et al. (2023), la part B",
        pass_arguments),
    "describe-figures": Command(
        "epf.results.describe_figures", "torna a dibuixar les figures d'un pas de la part B",
        folder_arguments),
    "rebuild": Command(
        "epf.rebuild", "refà les mètriques i les figures d'una execució", folder_arguments),
    "attribute": Command(
        "epf.attribution.attribute", "explica una execució, dels ajustos que va desar",
        folder_arguments),
    "show-run": Command(
        "epf.results.show_run", "les famílies d'una execució en una sola taula",
        folder_arguments),
    "show-figures": Command(
        "epf.results.show_figures", "les figures que comparen les famílies d'una execució",
        folder_arguments),
}


def main():
    parser = argparse.ArgumentParser(
        prog="epf", description="Pronòstic de preu del mercat diari espanyol.")
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="ordre")
    for name, command in COMMANDS.items():
        subcommand = subcommands.add_parser(name, help=command.help,
                                            description=command.help)
        if command.arguments is not None:
            command.arguments(subcommand)

    arguments = parser.parse_args()
    command = COMMANDS[arguments.command]
    # La cerca és l'única ordre el mòdul de la qual depèn d'un argument: el de la família.
    module = importlib.import_module(SEARCHES[arguments.family] if command.module is None
                                     else command.module)
    return module.main() if command.arguments is None else module.main(arguments)
