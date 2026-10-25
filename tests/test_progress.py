from epf.run import progress


def test_the_failed_seeds_are_counted_and_the_first_named_in_the_families_order():
    order = ["lightgbm", "dnn"]
    assert progress.counted([("dnn", 1)], order) == "1 seed ha fallat, dnn seed 1"
    assert (
        progress.counted([("dnn", 0), ("lightgbm", 5), ("lightgbm", 3)], order)
        == "3 seeds han fallat, la primera lightgbm seed 3"
    )


def test_the_block_aligns_its_families_trials_and_fits():
    """Sense la columna dels trials en blanc, els ajustos no s'alinearien."""
    trials = [50, 50, 50, 50, 12, 0, 0, 0, 0, 0]
    fits = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    block = progress.Block(
        "Passada 2026-11-02_14-30",
        {
            "dnn": {
                seed: {"trial": (50, trials[seed]), "fit": (1, fits[seed])}
                for seed in range(10)
            },
            "lightgbm": {
                seed: {"trial": (50, 50), "fit": (10, 10)} for seed in range(10)
            },
            "naive_similar_day": {seed: {"fit": (1, 1)} for seed in range(10)},
        },
    )
    block.start("dnn")
    block.start("dnn")
    assert block._lines() == [
        "dnn                 3/10 seeds    212/500 trials      3/10  fits    ··",
        "lightgbm           10/10 seeds    500/500 trials    100/100 fits",
        "naive_similar_day  10/10 seeds                       10/10  fits",
    ]


def test_a_seed_is_done_at_its_last_fit_and_its_plan_sets_its_fits(capsys):
    """Comptada als trials, una seed es donaria per feta abans dels seus ajustos.

    Una família que es cerca només per a una altra acaba als seus trials.
    """
    block = progress.Block(
        "Passada",
        {
            "lightgbm": {0: {"trial": (2, 0), "fit": (10, 0)}},
            "elastic_net": {0: {"trial": (2, 0)}},
        },
    )
    for kind in ("trial", "trial"):
        block.advance("lightgbm", 0, kind)
        block.advance("elastic_net", 0, kind)
    block.plan("lightgbm", 0, 1)
    assert block._count("lightgbm") == "0/1 seed    2/2 trials    0/1 fit"
    block.advance("lightgbm", 0, "fit")
    assert capsys.readouterr().err.splitlines() == [
        "Passada",
        "",
        "elastic_net: 1/1 seed    2/2 trials",
        "lightgbm: 1/1 seed    2/2 trials    1/1 fit",
    ]


def test_a_seed_fails_once_and_its_other_units_with_it(capsys):
    """Les unitats que una seed ja feia quan falla poden fallar també.

    Comptada a cada una, una sola seed fallada en diria dues.
    """
    block = progress.Block(
        "Passada", {"lightgbm": {0: {"fit": (2, 0)}, 1: {"fit": (2, 0)}}}
    )
    for _ in range(4):
        block.start("lightgbm")
    block.advance("lightgbm", 0, "fit")
    block.advance("lightgbm", 0, "fit")
    block.fail("lightgbm", 1, "la primera")
    block.fail("lightgbm", 1, "la segona")
    assert block.working == {"lightgbm": 0}
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
