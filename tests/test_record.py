"""El registre d'una passada s'escriu a JSON sense perdre res en silenci, i l'empremta del
codi que porta diu quin codi el va fer, a Windows igual que a Linux."""
import json

import pytest

from epf.contract import record


def test_a_value_json_cannot_write_is_refused_and_not_silently_dropped():
    with pytest.raises(TypeError):
        json.dumps(record.plain({"bad": object()}))


def test_the_code_fingerprint_ignores_line_endings_and_sees_any_other_edit(tmp_path):
    # Una còpia amb els finals de línia de Windows ha de donar l'empremta de la de Linux, i
    # un comentari canviat, una altra.
    package = tmp_path / "epf"
    (package / "contract").mkdir(parents=True)
    (package / "cli.py").write_bytes(b"import os\n\nprint(os.sep)\n")
    (package / "contract" / "record.py").write_bytes(b"# un comentari\nx = 1\n")
    linux = record.code(package)
    (package / "cli.py").write_bytes(b"import os\r\n\r\nprint(os.sep)\r\n")
    assert record.code(package) == linux
    (package / "contract" / "record.py").write_bytes(b"# un altre comentari\nx = 1\n")
    assert record.code(package) != linux
