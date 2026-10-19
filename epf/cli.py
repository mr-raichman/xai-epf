"""Tot el que es tecleja a mà, en una sola ordre amb subordres."""

import argparse
import collections
import importlib
import importlib.metadata
import sys

import epf

# Una subordre: quina funció de quin mòdul l'executa, què en diu `epf --help` i qui li
# declara els arguments, `None` si no en pren cap i la funció es crida sense res. El
# mòdul hi va pel nom, perquè s'importi només quan es tria la subordre: `shap` costa un
# segon llarg, i el pagaria fins i tot `epf --help`.
Command = collections.namedtuple("Command", "module function help arguments")


# Els arguments es declaren aquí, i no al mòdul de cada subordre: `argparse` construeix
# totes les subordres abans de llegir-ne cap, i els importaria tots.
def run_arguments(parser):
    """Declara a `parser` els arguments d'una passada, i el torna."""
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="un assaig: els valors d'assaig de parameters.py en lloc dels de la "
        "mesura",
    )
    parser.add_argument(
        "--family",
        nargs="+",
        # Repetir l'opció n'afegeix els valors en lloc de canviar-los.
        action="extend",
        metavar="NOM",
        help="només aquestes famílies, o les famílies d'aquests models",
    )
    return parser


def search_arguments(parser):
    """Declara a `parser` els arguments d'una cerca, i el torna."""
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
    """Declara a `parser` els arguments de `epf figures`, i el torna."""
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
    """Executa la subordre i en torna l'estat de sortida, o 1 si es refusa.

    Si acaba bé, escriu `Acabat`.
    """
    # El títol del treball, la `description` de `pyproject.toml`.
    title = importlib.metadata.metadata("epf")["Summary"]
    parser = argparse.ArgumentParser(prog="epf", description=title)
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
