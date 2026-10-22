import os
import subprocess
import sys

import pytest

from epf import Refusal, parameters
from epf.run import run


def test_a_run_s_values_hold_no_smoke_variant():
    """Una variant d'assaig arribava al registre de cada passada, també de mesura."""
    variants = [name for name in vars(parameters) if name.startswith("SMOKE")]
    for smoke in (False, *variants):
        assert not [
            name
            for name in parameters.chosen(smoke, run.READ)
            if name.startswith("SMOKE")
        ]


@pytest.mark.parametrize("name", ["DEVELOPMENT", "THREADS"])
def test_a_variant_changing_a_value_no_command_reads_is_refused(monkeypatch, name):
    """Una variant canviava `THREADS` als processos, i no a l'arbre ni als studies."""
    monkeypatch.setattr(parameters, "SMOKE_9", {"SEEDS": (0,), name: ()}, raising=False)
    with pytest.raises(Refusal, match=f"SMOKE_9 canvia {name}, que cap comanda"):
        parameters.chosen("SMOKE_9", run.READ)


def test_a_seed_is_the_same_in_a_fresh_interpreter_whatever_its_hash_seed():
    """El `hash` d'un text canvia d'un procés a un altre, i amb ell la seed."""
    code = "from epf.parameters import seeded; print(seeded('sampler', 'dnn', 3, 7))"
    found = {
        subprocess.run(
            [sys.executable, "-c", code],
            env=os.environ | {"PYTHONHASHSEED": str(hash_seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for hash_seed in (1, 2)
    }
    assert found == {str(parameters.seeded("sampler", "dnn", 3, 7))}


def test_another_label_part_or_project_seed_gives_another_seed(monkeypatch):
    # Una part de més que val zero també: `SeedSequence` omple amb zeros una llista
    # curta, i les dues hi donen la mateixa.
    first = parameters.seeded("fit", 3)
    others = [
        parameters.seeded("split", 3),
        parameters.seeded("fit", 4),
        parameters.seeded("fit", 3, 0),
    ]
    monkeypatch.setattr(parameters, "PROJECT_SEED", 1)
    others.append(parameters.seeded("fit", 3))
    assert first not in others and len(set(others)) == len(others)


def test_every_seed_fits_in_32_bits():
    """Keras passa la seed d'un ajust a `numpy.random.seed`, que la vol sota 2^32."""
    assert all(0 <= parameters.seeded("fit", seed) < 2**32 for seed in range(1000))
