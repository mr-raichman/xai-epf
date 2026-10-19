"""Les famílies, pels noms que pren `--family`, i quan una passada és acabada."""

import json

from epf import Refusal, paths

# Cada model amb les seves famílies, en l'ordre en què es fan, es mostren i s'escriuen a
# tot arreu. Les xarxes són un model, `nn`, i cada una de les seves dues carpetes, `dnn`
# i `ddnn`, també.
MODELS = {
    "naive": ("naive_previous_day", "naive_previous_week", "naive_similar_day"),
    "floor": ("floor",),
    "elastic_net": ("elastic_net",),
    "lightgbm": ("date_probe", "lightgbm"),
    "nn": ("dnn", "ddnn_normal", "ddnn_jsu"),
    "dnn": ("dnn",),
    "ddnn": ("ddnn_normal", "ddnn_jsu"),
}

# Totes les famílies, de tots els models, en l'ordre de `MODELS`.
EVERY = tuple(dict.fromkeys(family for model in MODELS.values() for family in model))


def named(names, among):
    """Torna les famílies d'`among` que diu `names`, en l'ordre d'`among`.

    Totes si `names` és `None`, i cap si cap no hi és. Un model diu totes les seves
    famílies, i on un model i una família es diuen igual, el nom és el del model:
    `lightgbm` vol dir l'arbre i la seva sonda de datació. Refusa un nom que no és cap
    família ni cap model.
    """
    if names is None:
        return list(among)
    unknown = [name for name in names if name not in MODELS and name not in EVERY]
    if unknown:
        quoted = ", ".join(f'"{name}"' for name in unknown)
        verb = "és" if len(unknown) == 1 else "són"
        raise Refusal(
            f"{quoted} no {verb} cap família ni cap model; les famílies: "
            f"{', '.join(EVERY)}; els models: {', '.join(MODELS)}"
        )
    wanted = {family for name in names for family in MODELS.get(name, (name,))}
    return [family for family in among if family in wanted]


def recorded(folder):
    """Torna el pla del registre d'una família, o `None` si encara no en té.

    El pla són les configuracions de cada seed.
    """
    path = folder / paths.SEED_RECORD
    if not path.exists():
        return None
    found = json.loads(path.read_text(encoding="utf-8"))["configurations"]
    # JSON escriu les claus com a text.
    return {int(seed): planned for seed, planned in found.items()}


def missing(folder):
    """Torna les seeds a les quals falta algun fitxer, cada una amb el seu pla."""
    return {
        seed: planned
        for seed, planned in recorded(folder).items()
        if not all(
            (folder / paths.SEED.format(seed=seed, rank=one["rank"])).exists()
            for one in planned
        )
    }


def complete(folder):
    """Diu si una família té pla i tots els fitxers que el pla diu."""
    return recorded(folder) is not None and not missing(folder)


def finished(where):
    """Diu si una passada és acabada, amb tots els fitxers del pla de cada família.

    Les famílies són les de tots els models. Es llegeix dels fitxers cada vegada, i no
    es desa enlloc.
    """
    return all(complete(where / family) for family in EVERY)
