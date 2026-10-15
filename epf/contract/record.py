"""El que porta cada registre i com es desa el fitxer de cada model.

`plain` escriu un valor a JSON sense perdre què era: converteix les dates, les tuples i
les claus, i deixa que qualsevol altra cosa faci petar l'escriptura, en lloc d'escriure-la
com a text sense dir-ho. `save_model` desa un model primer a un fitxer a part, en torna a
fer les sumes de l'explicació sobre el que s'ha escrit, i només llavors li dona el seu nom.
"""
import datetime

import numpy as np

from epf.contract.explanation import read_back


def plain(value):
    """El valor tal com es pot escriure a JSON, sense perdre què era."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, datetime.date):
        return str(value)
    return value


def _days(index):
    return index.to_numpy().astype("datetime64[D]")


def save_model(path, parts, explained, described, scores, rank):
    """Desa un model, primer a un fitxer a part i després al seu nom, de manera que un
    fitxer amb nom és sempre un model sencer i, d'un model explicat, un que torna el que
    explica: les sumes de l'explicació es tornen a fer sobre el que s'ha escrit."""
    arrays = {f"days_{part}": _days(days) for part, days in parts._asdict().items()}
    arrays |= {"rank": rank, "days_explained": _days(explained),
               "predicted": described.predicted,
               "actual": described.actual, "gap": described.gap, **scores}
    for name, value in described.arrays.items():
        value = np.asarray(value)
        arrays[name] = value.astype(np.float32) if value.dtype.kind == "f" and value.ndim > 1 else value
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with open(partial, "wb") as handle:
        np.savez(handle, **arrays)
    if described.stored is not None:
        excess = read_back(partial, described.stored, described.tolerance)
        # `not ... <= 0` i no `> 0`, perquè un NaN també es refusi.
        if not excess <= 0:
            raise ValueError(f"l'explicació desada a {partial} no torna el que explica: "
                             f"se'n passa del marge en {excess:.3g}")
    partial.replace(path)
