from epf.run import progress


def test_the_failed_seeds_are_counted_and_the_first_named_in_the_families_order():
    order = ["lightgbm", "dnn"]
    assert progress.counted([("dnn", 1)], order) == "1 seed ha fallat, dnn seed 1"
    assert (
        progress.counted([("dnn", 0), ("lightgbm", 5), ("lightgbm", 3)], order)
        == "3 seeds han fallat, la primera lightgbm seed 3"
    )


def test_without_a_terminal_each_seed_is_a_line(capsys):
    """Una família d'una sola seed, com amb `--seed 3`, la deia en plural."""
    block = progress.Block(
        "Cerca",
        "trial",
        {
            "lightgbm": {0: (10, 0), 1: (10, 0)},
            "dnn": {3: (10, 0)},
            "floor": {0: (10, 10), 1: (10, 0)},
        },
    )
    for _ in range(10):
        block.advance("lightgbm", 0)
    for _ in range(10):
        block.advance("dnn", 3)
    block.fail("floor", 1, "<el missatge>")
    block.close()
    assert capsys.readouterr().err.splitlines() == [
        "Cerca",
        "",
        "lightgbm: 1/2 seeds    10/20 trials",
        "dnn: 1/1 seed    10/10 trials",
        "error: floor seed 1 ha fallat: <el missatge>",
    ]


def test_a_seed_finishes_at_its_last_unit_and_fails_once(capsys):
    """Les unitats que una seed ja feia quan falla poden fallar també.

    Comptada a cada una, una sola seed fallada en diria dues.
    """
    block = progress.Block("Passada", "fit", {"lightgbm": {0: (2, 0), 1: (2, 0)}})
    block.advance("lightgbm", 0)
    block.advance("lightgbm", 0)
    block.fail("lightgbm", 1, "la primera")
    block.fail("lightgbm", 1, "la segona")
    assert block.failed == [("lightgbm", 1)]
    assert block.stopped("lightgbm", 1) and not block.stopped("lightgbm", 0)
    assert capsys.readouterr().err.splitlines() == [
        "Passada",
        "",
        "lightgbm: 1/2 seeds    2/4 fits",
        "error: lightgbm seed 1 ha fallat: la primera",
    ]


def test_a_live_line_is_cut_to_one_row_of_the_terminal():
    """Una línia més ampla que el terminal deixava restes del bloc d'abans.

    Es partia en dues files, i el cursor en pujava una per línia.
    """
    line = "error: lightgbm seed 3 ha fallat: " + "x" * 100
    assert progress._cut(line, 80) == line[:76] + "..."
    assert len(progress._cut(line, 80)) == 79
    assert progress._cut("ValueError: una\nque en té dues", 80) == "ValueError: una..."
    assert progress._cut(line, 0) == line
