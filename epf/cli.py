"""Tot el que es tecleja a mà, en una sola ordre amb subordres.

`epf --help` és la llista de què es pot teclejar.

Els mòduls s'importen quan es tria la subordre i no abans.
`models/lightgbm/attribution.py` porta `shap`, que costa un segon llarg, i amb els
imports a dalt el pagarien `epf dataset` i fins i tot `epf --help`. Per això a la taula
hi ha el nom del mòdul i no el mòdul, i per això els arguments es declaren aquí:
`argparse` construeix totes les subordres abans de llegir-ne cap, així que demanar-los a
cada mòdul els tornaria a importar tots.
"""

import argparse
import collections
import importlib
import sys

import epf

# Una subordre: quina funció de quin mòdul l'executa, què en diu `epf --help` i qui li
# declara els arguments. `arguments` és `None` quan no en pren cap, i és també el que
# diu que la seva funció es crida sense res.
Command = collections.namedtuple("Command", "module function help arguments")


def run_arguments(parser):
    """Els arguments d'una passada o d'una cerca: si és un assaig, i quines famílies.
    `extend` perquè repetir l'opció n'afegeixi els valors en lloc de canviar-los."""
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="un assaig: els valors d'assaig de parameters.py en lloc dels de la "
        "mesura",
    )
    parser.add_argument(
        "--family",
        nargs="+",
        action="extend",
        metavar="NOM",
        help="només aquestes famílies, o les famílies d'aquests models",
    )
    return parser


def search_arguments(parser):
    """Els arguments d'una cerca: els d'una passada, i quines llavors."""
    run_arguments(parser)
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        action="extend",
        metavar="N",
        help="només aquestes llavors",
    )
    return parser


def figures_arguments(parser):
    """La passada de la qual es tornen a dibuixar les figures: la que es nomeni o, sense
    nom, l'última acabada del seu tipus."""
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="sense nom, l'última passada d'assaig acabada en lloc de la de mesura",
    )
    parser.add_argument(
        "run",
        metavar="PASSADA",
        nargs="?",
        help="el nom d'una passada, com 2026-10-14_01 o assaig_2026-10-14_01; "
        "sense, l'última acabada",
    )
    return parser


COMMANDS = {
    "download": Command(
        "epf.dataset.build",
        "download",
        "baixa el que encara falta de cada indicador",
        None,
    ),
    "dataset": Command(
        "epf.dataset.build",
        "main",
        "fa el conjunt de dades dels fitxers baixats, el comprova i en desa les "
        "figures i els valors més allunyats dels veïns",
        None,
    ),
    "search": Command(
        "epf.run.search",
        "main",
        "cerca els hiperparàmetres de cada família amb la cerca a l'atzar de "
        "Trebbien et al. (2023)",
        search_arguments,
    ),
    "run": Command(
        "epf.run.run",
        "main",
        "descriu el preu amb el protocol de Trebbien et al. (2023)",
        run_arguments,
    ),
    "figures": Command(
        "epf.run.figures",
        "main",
        "torna a dibuixar les figures d'una passada",
        figures_arguments,
    ),
}


def main():
    parser = argparse.ArgumentParser(prog="epf", description=epf.__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="ordre")
    for name, command in COMMANDS.items():
        subcommand = subcommands.add_parser(
            name, help=command.help, description=command.help
        )
        if command.arguments is not None:
            command.arguments(subcommand)

    arguments = parser.parse_args()
    command = COMMANDS[arguments.command]
    function = getattr(importlib.import_module(command.module), command.function)
    try:
        status = function() if command.arguments is None else function(arguments)
    except epf.Refusal as refusal:
        print(f"error: {refusal}", file=sys.stderr)
        return 1
    if not status:
        print("Acabat", file=sys.stderr)
    return status
