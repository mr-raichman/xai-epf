import pytest

from epf import Refusal, parameters


def test_a_run_s_values_hold_no_smoke_variant():
    """Una variant d'assaig arribava al registre de cada passada, també de mesura."""
    for smoke in (False, *parameters.variants().values()):
        assert not [
            name for name in parameters.chosen(smoke) if name.startswith("SMOKE")
        ]


def test_a_variant_changing_a_value_no_command_reads_is_refused(monkeypatch):
    monkeypatch.setattr(
        parameters, "SMOKE_9", {"SEEDS": (0,), "DEVELOPMENT": ()}, raising=False
    )
    with pytest.raises(Refusal, match="SMOKE_9 canvia DEVELOPMENT, que cap comanda"):
        parameters.chosen("SMOKE_9")
