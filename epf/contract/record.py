"""Els registres de les cerques i de les passades, i el fitxer de cada model ajustat."""

import collections
import hashlib
import json

import numpy as np

from epf import Refusal, paths
from epf.contract.explanation import check, read_back
from epf.contract.metrics import scored

# El que surt d'ajustar una família en una seed i una configuració: la predicció i el
# preu de les files de prova en el mateix ordre, el que es desa de l'explicació, el
# pitjor residu de l'explicació contra la predicció que explica, amb el marge que li
# toca, i, de les explicades, les sumes que es tornen a fer sobre el fitxer desat. Un
# banc de comparació no explica res i deixa el residu a zero, que el seu fitxer també
# desa.
Fitted = collections.namedtuple(
    "Fitted",
    "predicted actual arrays gap tolerance stored",
    defaults=(0.0, None, None),
)

# Com es diu cada empremta que no és la d'ara, la que va començar la passada.
DIFFERENT = {
    "code": "un altre codi",
    "dataset": "un altre dataset",
    "environment": "unes altres biblioteques",
}


def code(package):
    """Torna l'empremta d'un paquet, la SHA-256 de tots els seus fitxers `.py`.

    Cada fitxer hi entra en l'ordre del seu camí dins del paquet, escrit amb `/`, i amb
    aquest camí al davant. Els finals de línia de Windows hi entren com els de Linux,
    perquè una mateixa còpia doni la mateixa empremta als dos sistemes; qualsevol altre
    canvi, en un comentari també, la canvia.
    """
    digest = hashlib.sha256()
    found = {
        path.relative_to(package).as_posix(): path for path in package.rglob("*.py")
    }
    for name in sorted(found):
        digest.update(name.encode() + b"\n")
        digest.update(found[name].read_bytes().replace(b"\r\n", b"\n") + b"\n")
    return digest.hexdigest()


def environment(lock):
    """Torna l'empremta de les biblioteques, la SHA-256 de `uv.lock`.

    Els finals de línia de Windows hi entren com els de Linux.
    """
    return hashlib.sha256(lock.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fingerprints():
    """Torna les empremtes del codi, del dataset i de les biblioteques.

    La del dataset és la que desa `output/data/dataset.json`.
    """
    dataset = json.loads(paths.DATASET_RECORD.read_text(encoding="utf-8"))["dataset"]
    return {
        "code": code(paths.PACKAGE),
        "dataset": dataset,
        "environment": environment(paths.LOCK),
    }


def refuse_changed(started):
    """Refusa una tasca si el codi, el dataset o les biblioteques ja no són els d'abans.

    `started` són les empremtes amb què va començar la passada. Una edició o un
    dataset nou a mitja passada hi entraria mentre els registres diuen les d'abans.
    """
    now = fingerprints()
    then = [other for name, other in DIFFERENT.items() if now[name] != started[name]]
    if then:
        raise Refusal(f"la passada es va començar amb {' i '.join(then)}")


def _days(index):
    """Torna els dies d'un índex com a `datetime64[D]`."""
    return index.to_numpy().astype("datetime64[D]")


def fitted_files(folder):
    """Torna el fitxer de cada model ajustat d'una família, per `(seed, configuració)`.

    En ordre de seed i de configuració, del primer de la primera seed, la seva millor.
    """
    found = {
        tuple(int(part) for part in path.stem.split("_")[1:]): path
        for path in folder.glob(paths.MODEL.format(seed="*", rank="*"))
    }
    return dict(sorted(found.items()))


def save_model(path, parts, explained, fitted, rank):
    """Comprova, puntua i desa un model ajustat, `fitted`, amb la seva partició.

    Un fitxer amb nom és sempre un model ajustat sencer i, d'una família explicada, un
    que torna el que explica: l'explicació ha de tornar la predicció abans d'escriure'l,
    i les sumes es tornen a fer sobre el que s'ha escrit; un que no ho fa es refusa.
    `explained` són els dies explicats, i `None` en una família que no explica res, que
    no en desa cap ni es comprova. Les puntuacions són l'R² i el MAE. Les taules de més
    d'una dimensió es desen en `float32`. `rank` és el lloc de la configuració.
    """
    if explained is not None:
        check(fitted, rank)
    scores = scored(fitted.predicted, fitted.actual)
    arrays = {f"days_{part}": _days(days) for part, days in parts._asdict().items()}
    if explained is not None:
        arrays["days_explained"] = _days(explained)
    arrays |= {
        "rank": rank,
        # La xarxa neuronal la dona en `float32`: un sol tipus a totes les famílies.
        "predicted": np.asarray(fitted.predicted, dtype=np.float64),
        "actual": fitted.actual,
        "gap": fitted.gap,
        **scores,
    }
    for name, value in fitted.arrays.items():
        value = np.asarray(value)
        arrays[name] = (
            value.astype(np.float32)
            if value.dtype.kind == "f" and value.ndim > 1
            else value
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = paths.partial(path)
    with open(partial, "wb") as handle:
        np.savez(handle, **arrays)
    if fitted.stored is not None:
        distance, margin = read_back(partial, fitted.stored, fitted.tolerance)
        # `not ... <=` i no `>`, perquè un NaN també es refusi.
        if not distance <= margin:
            raise Refusal(
                f"l'explicació desada de la configuració {rank} no torna la predicció: "
                f"se n'allunya {distance:.2g}, més que el marge de {margin:.2g}"
            )
    partial.replace(path)
