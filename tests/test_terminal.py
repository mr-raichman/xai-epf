import re
import sys

from epf import terminal


def _shown(capsys):
    return re.sub(r"\d+:\d\d:\d\d", "h:mm:ss", capsys.readouterr().err)


def test_without_a_terminal_the_time_comes_once_after_a_blank_line(capsys):
    # Com `epf dataset`, que no escriu res; `epf figures`, que acaba amb una línia en
    # blanc; i un bloc, que acaba amb la línia d'una família.
    with terminal.Clock():
        pass
    assert _shown(capsys) == "Temps transcorregut: h:mm:ss\n\n"
    with terminal.Clock():
        print("Passada 2026-10-18_14-30\n", file=sys.stderr)
    assert (
        _shown(capsys) == "Passada 2026-10-18_14-30\n\nTemps transcorregut: h:mm:ss\n\n"
    )
    with terminal.Clock():
        print("lightgbm: 1/2 seeds    10/20 trials", file=sys.stderr)
    assert (
        _shown(capsys)
        == "lightgbm: 1/2 seeds    10/20 trials\n\nTemps transcorregut: h:mm:ss\n\n"
    )
