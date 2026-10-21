import numpy as np

from epf import paths
from epf.models.lightgbm.hyperparameters import DAY_COUNTER
from epf.models.lightgbm.hyperparameters import NAME as TREE
from epf.run import tables


def _saved(where, family, seed, rank, **arrays):
    """Desa a la passada `where` un model ajustat que només porta `arrays`."""
    folder = where / family
    folder.mkdir(exist_ok=True)
    np.savez(folder / paths.SEED.format(seed=seed, rank=rank), **arrays)


def test_the_scores_follow_the_families_order_and_write_each_count_whole(tmp_path):
    # En un altre ordre que el de la taula.
    _saved(tmp_path, TREE, 1, 0, r2=0.75, mae=9.0)
    _saved(tmp_path, TREE, 0, 1, r2=0.125, mae=13.0)
    _saved(tmp_path, TREE, 0, 0, r2=0.25, mae=12.0)
    _saved(
        tmp_path,
        "ddnn_jsu",
        0,
        0,
        r2=0.5,
        mae=10.0,
        crps=4.0,
        kupiec_50=23,
        kupiec_90=20,
    )
    tables.scores(tmp_path)
    assert (tmp_path / tables.SCORES_FILE).read_text(encoding="utf-8").splitlines() == [
        "family,seed,rank,r2,mae,crps,kupiec_50,kupiec_90",
        "ddnn_jsu,0,0,0.5,10.0,4.0,23,20",
        "lightgbm,0,0,0.25,12.0,,,",
        "lightgbm,0,1,0.125,13.0,,,",
        "lightgbm,1,0,0.75,9.0,,,",
    ]


def test_the_probes_reading_is_its_counters_share_and_the_trees_best_mae(tmp_path):
    columns = np.array(["price_lag1_h00", DAY_COUNTER, "hour"])
    # Mitjanes del valor absolut 1, 2 i 2: la part del comptador és 2 de 5.
    values = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -1.0]])
    _saved(tmp_path, tables.PROBE, 0, 0, columns=columns, values=values, mae=7.0)
    _saved(tmp_path, TREE, 0, 0, mae=5.0)
    _saved(tmp_path, TREE, 0, 1, mae=6.0)
    # Una seed sense el seu arbre.
    _saved(tmp_path, tables.PROBE, 1, 0, columns=columns, values=values, mae=8.0)
    tables.date_probe(tmp_path)
    assert (tmp_path / tables.PROBE_FILE).read_text(encoding="utf-8").splitlines() == [
        "seed,counter_share,mae_tree,mae_probe",
        "0,0.4,5.0,7.0",
    ]
