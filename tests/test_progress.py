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
        {"lightgbm": (2, 20), "dnn": (1, 10), "floor": (2, 20)},
        {"lightgbm": (0, 0), "dnn": (0, 0), "floor": (1, 10)},
    )
    for _ in range(10):
        block.advance("lightgbm")
    block.finish("lightgbm")
    block.finish("dnn")
    block.fail("floor", 1, "<el missatge>")
    block.close()
    assert capsys.readouterr().err.splitlines() == [
        "Cerca",
        "",
        "lightgbm: 1/2 seeds    10/20 trials",
        "dnn: 1/1 seed    0/10 trials",
        "error: floor seed 1 ha fallat: <el missatge>",
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
