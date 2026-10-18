"""El que porta cada registre, com s'escriu, i com es desa el fitxer de cada model.

Cada registre, el d'una cerca i el d'una passada, porta les empremtes del codi que el va
fer, del conjunt de dades sobre el qual es va fer i de les biblioteques amb què es va
fer, de manera que qualsevol pot tornar a calcular la del codi d'una còpia i comprovar
quin codi va donar una xifra. `plain` escriu un valor a JSON sense perdre què era:
converteix les dates, les tuples i les claus, i deixa que qualsevol altra cosa faci
petar l'escriptura, en lloc d'escriure-la com a text sense dir-ho. `save_model` desa un
model primer a un fitxer a part, en torna a fer les sumes de l'explicació sobre el que
s'ha escrit, i només llavors li dona el seu nom.
"""

import collections
import datetime
import hashlib
import json

import numpy as np

from epf import Refusal, paths
from epf.contract.explanation import read_back

# El que surt d'ajustar una família en una llavor i una configuració: la predicció i el
# preu de les files de prova en el mateix ordre, el que es desa de l'explicació, el
# pitjor residu de l'explicació contra la predicció que explica, amb el marge que li
# toca, de les que prediuen una distribució les puntuacions de la distribució, i, de les
# explicades, les sumes que es tornen a fer sobre el fitxer desat. Un banc de comparació
# no explica res i deixa el residu a zero, que el seu fitxer també desa.
Fitted = collections.namedtuple(
    "Fitted",
    "predicted actual arrays gap tolerance scores stored",
    defaults=(0.0, None, None, None),
)


def code(package):
    """L'empremta d'un paquet: la SHA-256 de tots els seus fitxers `.py`, en l'ordre del
    seu camí dins del paquet escrit amb `/`, cada un amb aquest camí al davant. Els
    finals de línia de Windows hi entren com els de Linux, perquè una mateixa còpia doni
    la mateixa empremta als dos sistemes; qualsevol altre canvi, en un comentari també,
    la canvia."""
    digest = hashlib.sha256()
    found = {
        path.relative_to(package).as_posix(): path for path in package.rglob("*.py")
    }
    for name in sorted(found):
        digest.update(name.encode() + b"\n")
        digest.update(found[name].read_bytes().replace(b"\r\n", b"\n") + b"\n")
    return digest.hexdigest()


def environment(lock):
    """L'empremta de les biblioteques: la SHA-256 de `uv.lock`, que en fixa la versió
    exacta, amb els finals de línia de Windows com els de Linux."""
    return hashlib.sha256(lock.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fingerprints():
    """Les empremtes que porta cada registre: la del codi del paquet, la del conjunt de
    dades, que `data/dataset.json` desa, i la de les biblioteques."""
    dataset = json.loads(paths.DATASET_RECORD.read_text(encoding="utf-8"))["dataset"]
    return {
        "code": code(paths.PACKAGE),
        "dataset": dataset,
        "environment": environment(paths.LOCK),
    }


def plain(value):
    """El valor tal com es pot escriure a JSON, sense perdre què era."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, datetime.date):
        return str(value)
    return value


def write_record(path, content):
    """Escriu un registre a JSON, sencer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    paths.write_whole(path, json.dumps(plain(content), indent=2) + "\n")


def _days(index):
    return index.to_numpy().astype("datetime64[D]")


def save_model(path, parts, explained, fitted, scores, rank):
    """Desa un model, `fitted`, primer a un fitxer a part i després al seu nom, de
    manera que un fitxer amb nom és sempre un model sencer i, d'un model explicat, un
    que torna el que explica: les sumes de l'explicació es tornen a fer sobre el que
    s'ha escrit."""
    arrays = {f"days_{part}": _days(days) for part, days in parts._asdict().items()}
    arrays |= {
        "rank": rank,
        "days_explained": _days(explained),
        "predicted": fitted.predicted,
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
    partial = path.with_suffix(".part")
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
