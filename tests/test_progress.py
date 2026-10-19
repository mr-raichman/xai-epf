from epf.run import progress


def test_the_failed_seeds_are_counted_and_the_first_named_in_the_families_order():
    order = ["lightgbm", "dnn"]
    assert progress.counted([("dnn", 1)], order) == "1 llavor ha fallat, dnn llavor 1"
    assert (
        progress.counted([("dnn", 0), ("lightgbm", 5), ("lightgbm", 3)], order)
        == "3 llavors han fallat, la primera lightgbm llavor 3"
    )


def test_without_a_terminal_each_seed_is_a_line_and_the_time_comes_last(capsys):
    """Una família d'una sola llavor, com amb `--seed 3`, en deia `1 de 1 llavors`."""
    block = progress.Block("Cerca", {"lightgbm": 2, "dnn": 1, "floor": 2}, {"floor": 1})
    block.finish("lightgbm")
    block.finish("dnn")
    block.fail("floor", 1, "<el missatge>")
    block.close()
    lines = capsys.readouterr().err.splitlines()
    assert lines[:6] == [
        "Cerca",
        "",
        "lightgbm: 1 de 2 llavors",
        "dnn: 1 de 1 llavor",
        "error: floor llavor 1 ha fallat: <el missatge>",
        "",
    ]
    assert lines[6].startswith("Temps transcorregut: 0:00:0")
    assert lines[7:] == [""]


def test_a_live_line_is_cut_to_one_row_of_the_terminal():
    """Una línia més ampla que el terminal deixava restes del bloc d'abans.

    Es partia en dues files, i el cursor en pujava una per línia.
    """
    line = "error: lightgbm llavor 3 ha fallat: " + "x" * 100
    assert progress._cut(line, 80) == line[:76] + "..."
    assert len(progress._cut(line, 80)) == 79
    assert progress._cut("ValueError: una\nque en té dues", 80) == "ValueError: una..."
    assert progress._cut(line, 0) == line
