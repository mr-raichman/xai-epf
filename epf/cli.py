"""Tot el que es tecleja a mà, en una sola ordre amb subordres.

`epf --help` és la llista de què es pot teclejar. Aquest és l'únic lloc que sap quines
subordres hi ha, com es diuen, quins arguments prenen i quin mòdul n'executa cada una,
de manera que afegir-ne una és una fila de `COMMANDS` i res més.

Els mòduls s'importen quan es tria la subordre i no abans. `models/boosting/boosting.py`
porta `shap`, que costa un segon llarg, i amb els imports a dalt el pagarien `epf
audit-data` i fins i tot `epf --help`. Per això a la taula hi ha el nom del mòdul i no el
mòdul, i per això els arguments es declaren aquí: `argparse` construeix totes les
subordres abans de llegir-ne cap, així que demanar-los a cada mòdul els tornaria a
importar tots.
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
    """Els arguments d'una passada o d'una cerca: el nivell, quines famílies i si es
    comença de zero."""
    parser.add_argument("--tier", choices=(parameters.SMOKE, parameters.MEASUREMENT),
                        required=True,
                        help="assaig, que no dóna cap xifra, o mesura, que sí")
    parser.add_argument("--family", action="append",
                        help="només aquesta família; es pot repetir")
    parser.add_argument("--replace", action="store_true",
                        help="comença de zero, encara que n'hi hagi una a mig fer")
    return parser


def folder_arguments(parser):
    """El directori d'una passada, que és el que pren qui en llegeix una."""
    parser.add_argument("folder", type=pathlib.Path, nargs="+",
                        help="directori d'una passada")
    return parser


COMMANDS = {
    "download-price": Command(
        "epf.data.price", "baixa el preu de les dues fonts i diu què hi ha al disc", None),
    "download-forecasts": Command(
        "epf.data.forecasts", "baixa les previsions D+1 i diu què hi ha al disc", None),
    "download-schedule": Command(
        "epf.data.schedule", "baixa el programa base per tecnologies i diu què hi ha al disc",
        None),
    "audit-data": Command(
        "epf.data.audit", "el que només es pot preguntar a OMIE i a ESIOS", None),
    "figure-price": Command(
        "epf.data.figures.price_series", "tots els preus horaris, del 2014 endavant", None),
    "figure-outliers": Command(
        "epf.data.figures.price_outliers", "la distància de cada preu als seus veïns", None),
    "describe": Command(
        "epf.describe", "descriu el preu amb el protocol de Trebbien et al. (2023)",
        pass_arguments),
    "search": Command(
        "epf.search", "cerca els números de cada família amb la cerca a l'atzar de "
        "Trebbien et al. (2023)", pass_arguments),
    "describe-figures": Command(
        "epf.results.describe_figures", "torna a dibuixar les figures d'una passada",
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
