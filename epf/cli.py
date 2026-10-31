"""Tot el que es tecleja a mà, en una sola comanda amb subcomandes."""

import argparse
import collections
import importlib
import importlib.metadata
import os
import re
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
# Una opció d'assaig amb número, `--smoke2`, que tria la variant `SMOKE_2`: argparse no
# en declara cap pel seu patró, i `_parsed` la llegeix abans.
NUMBERED_SMOKE = re.compile(r"--smoke(\d+)")


def _parsed(parser, words):
    """Torna els arguments de `words` per `parser`, amb la variant d'una `--smokeN`.

    Una `--smokeN` hi entra com `--smoke`, i desa a `smoke` la variant `SMOKE_N`. No
    se'n pot donar més d'una, ni amb `--smoke`.
    """
    numbered = [NUMBERED_SMOKE.fullmatch(word) for word in words]
    numbers = [found.group(1) for found in numbered if found]
    if len(numbers) + words.count("--smoke") > 1:
        parser.error("només es pot donar una opció d'assaig")
    arguments = parser.parse_args(
        ["--smoke" if found else word for found, word in zip(numbered, words)]
    )
    if numbers:
        arguments.smoke = f"SMOKE_{numbers[0]}"
    return arguments


def set_argument(parser):
    """Declara a `parser` l'information set, `--set`, que sempre s'ha de donar."""
    parser.add_argument(
        "--set",
        dest="set_name",
        required=True,
        metavar="NOM",
        help="l'information set d'aquest nom a information.py",
    )
    return parser


def start_arguments(parser):
    """Declara a `parser` l'information set, l'assaig i les respostes, i el torna.

    `--set` desa a `set_name` el nom de l'information set. `--smoke` desa a
    `smoke` la variant `SMOKE`, i una `--smokeN`, la `SMOKE_N` (`_parsed`). Les
    respostes, `--replace` i `--continue`, desen a `answer` el nom de la seva, i se'n
    pot donar una o cap.
    """
    set_argument(parser)
    parser.add_argument(
        "--smoke",
        action="store_const",
        const="SMOKE",
        default=False,
        help="els valors de SMOKE de parameters.py en lloc dels de la mesura",
    )
    answer = parser.add_mutually_exclusive_group()
    answer.add_argument(
        "--replace",
        dest="answer",
        action="store_const",
        const="replace",
        default=None,
        help="substitueix el que hi ha en disc",
    )
    answer.add_argument(
        "--continue",
        dest="answer",
        action="store_const",
        const="continue",
        default=None,
        help="continua la passada en disc si es pot i, si no, s'atura",
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
        help="només aquestes famílies o models",
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
    return set_argument(parser)


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
        "torna a dibuixar les figures i a escriure els números de la passada en disc",
        figures_arguments,
    ),
    "start": Command(
        "epf.cli",
        "start",
        'fa "download", "dataset" i "run", en ordre',
        start_arguments,
    ),
}


def start(arguments):
    """Fa cada subcomanda de `SEQUENCE`, la comanda `epf start`.

    Les fa en ordre i s'atura a la primera que es refusa. Cada una rep els seus
    arguments per defecte, amb l'information set, l'opció d'assaig i la resposta donades
    a `epf start` si ella en pren. Si l'entorn té `NOTIFY`, n'executa el programa en
    acabar cada una menys l'última, i un programa que no es pot executar es diu.
    """
    for position, name in enumerate(SEQUENCE):
        command = COMMANDS[name]
        function = getattr(importlib.import_module(command.module), command.function)
        if command.arguments is None:
            function()
        else:
            given = ["--set", arguments.set_name]
            given += [parameters.flag(arguments.smoke)] if arguments.smoke else []
            given += [f"--{arguments.answer}"] if arguments.answer else []
            function(_parsed(command.arguments(argparse.ArgumentParser()), given))
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

    arguments = _parsed(parser, sys.argv[1:])
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
