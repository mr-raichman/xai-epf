from epf import parameters


def test_a_run_s_values_hold_no_smoke_variant():
    """Una variant d'assaig arribava al registre de cada passada, també de mesura."""
    for smoke in (False, *parameters.variants().values()):
        assert not [
            name for name in parameters.chosen(smoke) if name.startswith("SMOKE")
        ]
