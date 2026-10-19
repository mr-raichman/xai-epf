import json

import pytest

from epf.contract import record


def test_a_value_json_cannot_write_is_refused_and_not_silently_dropped():
    with pytest.raises(TypeError):
        json.dumps(record.plain({"bad": object()}))


def test_the_code_fingerprint_ignores_line_endings_and_sees_any_other_edit(tmp_path):
    package = tmp_path / "epf"
    (package / "contract").mkdir(parents=True)
    (package / "cli.py").write_bytes(b"import os\n\nprint(os.sep)\n")
    (package / "contract" / "record.py").write_bytes(b"# un comentari\nx = 1\n")
    linux = record.code(package)
    (package / "cli.py").write_bytes(b"import os\r\n\r\nprint(os.sep)\r\n")
    assert record.code(package) == linux
    (package / "contract" / "record.py").write_bytes(b"# un altre comentari\nx = 1\n")
    assert record.code(package) != linux


def test_the_environment_fingerprint_ignores_line_endings_and_sees_a_version_change(
    tmp_path,
):
    # L'entrada de LightGBM del `uv.lock` del projecte.
    entry = '[[package]]\nname = "lightgbm"\nversion = "{}"\n'
    lock = tmp_path / "uv.lock"
    lock.write_bytes(entry.format("4.7.0").encode())
    linux = record.environment(lock)
    lock.write_bytes(entry.format("4.7.0").replace("\n", "\r\n").encode())
    assert record.environment(lock) == linux
    lock.write_bytes(entry.format("4.6.0").encode())
    assert record.environment(lock) != linux
