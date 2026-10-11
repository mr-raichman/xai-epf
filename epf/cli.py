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

# Una subordre: quin mòdul l'executa, què en diu `epf --help` i qui li declara els
# arguments. `arguments` és `None` quan no en pren cap, i és també el que diu que el
# seu `main` es crida sense res.
Command = collections.namedtuple("Command", "module help arguments")


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


def run_arguments(parser):
    """Els arguments del camí endavant, els mateixos per `epf run` i per `python -m`."""
    pass_arguments(parser)
    parser.add_argument("--price-only", action="store_true", required=True,
                        help="conjunt d'informació de només preu, l'únic que hi ha")
    return parser


def search_arguments(parser):
    """Les proves que fa la cerca d'hiperparàmetres. Quantes es decideix amb el cost
    d'una mesurat, i el valor per defecte viu amb la xarxa i no aquí."""
    parser.add_argument("--trials", type=int,
                        help="proves que es fan; per defecte, les del seu fitxer")
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
    "search-dnn": Command(
        "epf.models.dnn.search", "cerca els hiperparàmetres de la xarxa",
        search_arguments),
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
    module = importlib.import_module(command.module)
    return module.main() if command.arguments is None else module.main(arguments)
