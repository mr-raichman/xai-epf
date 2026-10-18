"""Les famílies: els noms que pren `--family`, cada família pel seu i cada model, una
carpeta de `models/`, per totes les seves, i si una passada les té totes. On un model i
una família es diuen igual, el nom és el del model, de manera que `lightgbm` vol dir
l'arbre i la seva sonda de datació.

Que una passada sigui acabada es llegeix del pla de cada família i dels fitxers que hi
ha, i no es desa enlloc.
"""

import json

from epf import Refusal, paths

# Cada model amb les seves famílies. Les xarxes són un model, `nn`, i cada una de les
# seves dues carpetes, `dnn` i `ddnn`, també.
MODELS = {
    "lightgbm": ("lightgbm", "date_probe"),
    "elastic_net": ("elastic_net",),
    "nn": ("dnn", "ddnn_normal", "ddnn_jsu"),
    "dnn": ("dnn",),
    "ddnn": ("ddnn_normal", "ddnn_jsu"),
    "floor": ("floor",),
    "naive": ("naive_previous_day", "naive_previous_week", "naive_similar_day"),
}

# Totes les famílies, de tots els models.
EVERY = frozenset(family for model in MODELS.values() for family in model)


def named(names, among):
    """Torna les famílies d'`among` que diu `names`, en l'ordre d'`among`, i totes si
    `names` és `None`; cap, si cap no hi és. Refusa un nom que no és cap família ni cap
    model."""
    if names is None:
        return list(among)
    unknown = [name for name in names if name not in MODELS and name not in EVERY]
    if unknown:
        quoted = ", ".join(f'"{name}"' for name in unknown)
        verb = "és" if len(unknown) == 1 else "són"
        raise Refusal(
            f"{quoted} no {verb} cap família ni cap model; les famílies: "
            f"{', '.join(sorted(EVERY))}; els models: {', '.join(sorted(MODELS))}"
        )
    wanted = {family for name in names for family in MODELS.get(name, (name,))}
    return [family for family in among if family in wanted]


def recorded(folder):
    """Les configuracions de cada llavor que desa el registre d'una família d'una
    passada, el seu pla, o `None` si encara no en té."""
    path = folder / paths.SEED_RECORD
    if not path.exists():
        return None
    found = json.loads(path.read_text(encoding="utf-8"))["configurations"]
    # JSON escriu les claus com a text.
    return {int(seed): planned for seed, planned in found.items()}


def missing(folder):
    """Les llavors del pla d'una família a les quals falta algun fitxer, cada una amb
    les seves configuracions."""
    return {
        seed: planned
        for seed, planned in recorded(folder).items()
        if not all(
            (folder / paths.SEED.format(seed=seed, rank=one["rank"])).exists()
            for one in planned
        )
    }


def complete(folder):
    """Si una família té pla i hi té tots els fitxers que diu."""
    return recorded(folder) is not None and not missing(folder)


def finished(where):
    """Si una passada és acabada: si cada família hi té tots els fitxers del seu pla."""
    return all(complete(where / family) for family in EVERY)
