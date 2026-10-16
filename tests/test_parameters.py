"""Els valors d'una passada, els de la mesura o els d'un assaig."""

from epf import parameters


def test_a_run_s_values_hold_no_smoke_variant():
    # Una variant d'assaig que arribava al registre de cada passada, les de mesura
    # incloses.
    for smoke in (False, True):
        assert not [
            name for name in parameters.chosen(smoke) if name.startswith("SMOKE")
        ]
