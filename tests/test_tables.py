import numpy as np

from epf import paths
from epf.run import tables


def _family(where, family, fits, **fields):
    """Desa a la passada `where` una família completa, els seus models de `fits`.

    Cada model ajustat, per `(seed, configuració)`, porta el seu R² i el seu MAE; el
    registre, també `fields`.
    """
    seeds = sorted({seed for seed, _ in fits})
    paths.write_record(
        where / family / paths.FAMILY_RECORD,
        {
            "configurations": {
                str(seed): [{"rank": rank} for one, rank in fits if one == seed]
                for seed in seeds
            },
            "settings": {"SEEDS": seeds},
            **fields,
        },
    )
    for (seed, rank), (r2, mae) in fits.items():
        np.savez(
            where / family / paths.MODEL.format(seed=seed, rank=rank),
            r2=np.float64(r2),
            mae=np.float64(mae),
        )


def test_the_numbers_go_by_measure_and_family_their_keys_integers(tmp_path):
    """pandas escriu una columna d'enters amb alguna casella buida com a decimals."""
    # Desades en un altre ordre que el del fitxer.
    _family(
        tmp_path,
        "naive_previous_week",
        {(1, 0): (0.125, 16.0), (0, 0): (0.0, 20.0)},
        denominator="main",
    )
    _family(tmp_path, "floor", {(0, 0): (0.25, 30.0)})
    _family(
        tmp_path,
        "lightgbm",
        {(1, 0): (0.75, 9.0), (0, 1): (0.25, 12.0), (0, 0): (0.5, 10.0)},
    )
    tables.write(tmp_path)
    lines = (tmp_path / tables.PART_I_MAIN).read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(tables.COLUMNS)
    assert [line for line in lines if line.startswith("r2,")] == [
        "r2,lightgbm,,0,0,,,,0.5,sense unitats",
        "r2,lightgbm,,0,1,,,,0.25,sense unitats",
        "r2,lightgbm,,1,0,,,,0.75,sense unitats",
        "r2,floor,,0,0,,,,0.25,sense unitats",
        "r2,naive_previous_week,,0,0,,,,0.0,sense unitats",
        "r2,naive_previous_week,,1,0,,,,0.125,sense unitats",
    ]
    assert "rmae,lightgbm,naive_previous_week,1,0,,,,0.5625,sense unitats" in lines
    assert "mae_mean,naive_previous_week,,,0,,,,18.0,EUR/MWh" in lines
    # Una sola seed no té desviació.
    assert "mae_std,floor,,,0,,,,,EUR/MWh" in lines
