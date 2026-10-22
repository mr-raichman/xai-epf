"""Tot el que es tecleja a mà, en una sola comanda amb subcomandes."""

import argparse
import collections
import importlib
import importlib.metadata
import os
import shlex
import signal
import subprocess
import sys

import epf
from epf import parameters, terminal

# Una subcomanda: quina funció de quin mòdul l'executa, què en diu `epf --help` i qui li
# declara els arguments, `None` si no en pren cap i la funció es crida sense res. El
# mòdul hi va pel nom, perquè s'importi només quan es tria la subcomanda: `shap` costa
# un segon llarg, i el pagaria fins i tot `epf --help`.
Command = collections.namedtuple("Command", "module function help arguments")

# Les subcomandes que fa `epf start`, en ordre.
SEQUENCE = ("download", "dataset", "run")

# La variable d'entorn que nomena un programa, amb els seus arguments, que `epf start`
# executa amb el nom de cada subcomanda que acaba, llevat de l'última, amb la qual acaba
# la comanda. Sense, no s'executa res.
NOTIFY = "EPF_NOTIFY"


# Els arguments es declaren aquí, i no al mòdul de cada subcomanda: `argparse`
# construeix totes les subcomandes abans de llegir-ne cap, i els importaria tots.
def start_arguments(parser):
    """Declara a `parser` una opció per variant d'assaig i les respostes d'una passada.

    Les variants desen a `smoke` el nom de la seva, i se'n pot donar una o cap. Les
    respostes, `--replace` i `--continue`, desen a `answer` el nom de la seva, i se'n
    pot donar una o cap. Torna `parser`.
    """
    smoke = parser.add_mutually_exclusive_group()
    for flag, name in parameters.variants().items():
        smoke.add_argument(
            flag,
            dest="smoke",
            action="store_const",
            const=name,
            default=False,
            help=f"un assaig: els valors de {name} de parameters.py en lloc dels de la "
            "mesura",
        )
    answer = parser.add_mutually_exclusive_group()
    answer.add_argument(
        "--replace",
        dest="answer",
        action="store_const",
        const="replace",
        default=None,
        help="sense preguntar, substitueix el que hi ha en disc",
    )
    answer.add_argument(
        "--continue",
        dest="answer",
        action="store_const",
        const="continue",
        default=None,
        help="sense preguntar, continua la passada en disc si es pot i, si no, s'atura",
    )
    return parser


def run_arguments(parser):
    """Declara a `parser` els arguments d'una passada, i el torna."""
    start_arguments(parser)
    parser.add_argument(
        "--family",
        nargs="+",
        # Repetir l'opció n'afegeix els valors en lloc de canviar-los.
        action="extend",
        metavar="NOM",
        help="només aquestes famílies, o les famílies d'aquests models",
    )
    return parser


def figures_arguments(parser):
    """Declara a `parser` els arguments de `epf figures`, i el torna."""
    # Una sola opció d'assaig, i no una per variant: totes les variants escriuen la seva
    # passada al mateix lloc.
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="la passada d'assaig, a assaig_output/, en lloc de la de mesura",
    )
    return parser


COMMANDS = {
    "download": Command(
        "epf.dataset.build",
        "download",
        "baixa les dades d'ESIOS",
        None,
    ),
    "dataset": Command(
        "epf.dataset.build",
        "main",
        "crea el dataset a partir dels fitxers baixats, el comprova i fa l'estudi "
        "d'outliers",
        None,
    ),
    "run": Command(
        "epf.run.run",
        "main",
        "cerca els hiperparàmetres, entrena els models i descriu el preu amb el "
        "protocol de Trebbien et al. (2023)",
        run_arguments,
    ),
    "figures": Command(
        "epf.run.figures",
        "main",
        "torna a dibuixar les figures de la passada en disc",
        figures_arguments,
    ),
    "start": Command(
        "epf.cli",
        "start",
        "executa tot el mecanisme de manera seqüencial",
        start_arguments,
    ),
}


def start(arguments):
    """Fa cada subcomanda de `SEQUENCE`, la comanda `epf start`.

    Les fa en ordre i s'atura a la primera que es refusa. Cada una rep els seus
    arguments per defecte, amb l'opció d'assaig i la resposta donades a `epf start` si
    ella en pren. Si
    l'entorn té `NOTIFY`, n'executa el programa en acabar cada una menys l'última, i un
    programa que no es pot executar es diu.
    """
    flags = {variant: flag for flag, variant in parameters.variants().items()}
    for position, name in enumerate(SEQUENCE):
        command = COMMANDS[name]
        function = getattr(importlib.import_module(command.module), command.function)
        if command.arguments is None:
            function()
        else:
            given = [flags[arguments.smoke]] if arguments.smoke else []
            given += [f"--{arguments.answer}"] if arguments.answer else []
            function(command.arguments(argparse.ArgumentParser()).parse_args(given))
        if position < len(SEQUENCE) - 1 and NOTIFY in os.environ:
            # Res del programa no arriba al terminal, on trencaria el rellotge.
            try:
                subprocess.run(
                    [*shlex.split(os.environ[NOTIFY]), name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except OSError as error:
                print(
                    f"error: no s'ha pogut avisar del final de {name}: {error}",
                    file=sys.stderr,
                )


def _stopped(signum, frame):
    """Atura la comanda al primer Ctrl+C; els següents ja no en tallen el tancament."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise KeyboardInterrupt


def main():
    """Executa la subcomanda i torna 0, 1 si es refusa o 130 si s'atura amb Ctrl+C.

    N'escriu el temps transcorregut mentre es fa i en acabar, i després `Finalitzat`, el
    refús o `Aturat`.
    """
    # El títol del treball, la `description` de `pyproject.toml`.
    title = importlib.metadata.metadata("epf")["Summary"]
    parser = argparse.ArgumentParser(prog="epf", description=title)
    subcommands = parser.add_subparsers(
        dest="command", required=True, metavar="comanda"
    )
    for name, command in COMMANDS.items():
        subcommand = subcommands.add_parser(
            name, help=command.help, description=command.help
        )
        if command.arguments is not None:
            command.arguments(subcommand)

    arguments = parser.parse_args()
    command = COMMANDS[arguments.command]
    signal.signal(signal.SIGINT, _stopped)
    try:
        # Abans d'importar el mòdul de la subcomanda, que pot trigar uns segons.
        with terminal.Clock():
            module = importlib.import_module(command.module)
            function = getattr(module, command.function)
            if command.arguments is None:
                function()
            else:
                function(arguments)
    except epf.Refusal as refusal:
        print(f"error: {refusal}", file=sys.stderr)
        return 1
    except epf.Declined:
        print("Aturat", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Aturat", file=sys.stderr)
        return 130
    print("Finalitzat", file=sys.stderr)
    return 0
