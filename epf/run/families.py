"""Les famílies, pels noms que pren `--family`, i quan una passada és acabada."""

import json

from epf import Refusal, paths

# Cada família, en l'ordre en què es fan, es mostren i s'escriuen a tot arreu, amb els
# models que la tenen i els mòduls del seu model que l'ajusten, la cerquen i la
# dibuixen, o `None` si no en té cap, que s'importen quan es fan servir. Les xarxes són
# un model, `nn`, i cada una de les seves dues carpetes, `ddnn` i `dnn`, també.
MODULES = {
    "ddnn_jsu": {
        "models": ("nn", "ddnn"),
        "train": "epf.models.nn.ddnn.train",
        "search": "epf.models.nn.search",
        "figures": "epf.models.nn.ddnn.figures",
    },
    "ddnn_normal": {
        "models": ("nn", "ddnn"),
        "train": "epf.models.nn.ddnn.train",
        "search": "epf.models.nn.search",
        "figures": "epf.models.nn.ddnn.figures",
    },
    "dnn": {
        "models": ("nn", "dnn"),
        "train": "epf.models.nn.dnn.train",
        "search": "epf.models.nn.search",
        "figures": None,
    },
    "elastic_net": {
        "models": ("elastic_net",),
        "train": "epf.models.elastic_net.train",
        "search": "epf.models.elastic_net.search",
        "figures": "epf.models.elastic_net.figures",
    },
    "lightgbm": {
        "models": ("lightgbm",),
        "train": "epf.models.lightgbm.train",
        "search": "epf.models.lightgbm.search",
        "figures": "epf.models.lightgbm.figures",
    },
    "date_probe": {
        "models": ("lightgbm",),
        "train": "epf.models.lightgbm.date_probe",
        "search": None,
        "figures": "epf.models.lightgbm.figures",
    },
    "floor": {
        "models": ("floor",),
        "train": "epf.models.floor.train",
        "search": None,
        "figures": "epf.models.floor.figures",
    },
    "naive_similar_day": {
        "models": ("naive",),
        "train": "epf.models.naive.train",
        "search": None,
        "figures": None,
    },
    "naive_previous_week": {
        "models": ("naive",),
        "train": "epf.models.naive.train",
        "search": None,
        "figures": None,
    },
    "naive_previous_day": {
        "models": ("naive",),
        "train": "epf.models.naive.train",
        "search": None,
        "figures": None,
    },
}

EVERY = tuple(MODULES)

# Les famílies de cada model, en ordre.
MODELS = {
    model: tuple(family for family in EVERY if model in MODULES[family]["models"])
    for model in dict.fromkeys(
        model for found in MODULES.values() for model in found["models"]
    )
}


def named(names):
    """Torna les famílies que diu `names`, en ordre.

    Totes si `names` és `None`, i cap si cap no hi és. Un model diu totes les seves
    famílies, i on un model i una família es diuen igual, el nom és el del model:
    `lightgbm` vol dir l'arbre i la seva sonda de datació. Refusa un nom que no és cap
    família ni cap model.
    """
    if names is None:
        return list(EVERY)
    unknown = [name for name in names if name not in MODELS and name not in EVERY]
    if unknown:
        quoted = ", ".join(f'"{name}"' for name in unknown)
        verb = "és" if len(unknown) == 1 else "són"
        raise Refusal(
            f"{quoted} no {verb} cap família ni cap model; les famílies: "
            f"{', '.join(EVERY)}; els models: {', '.join(MODELS)}"
        )
    wanted = {family for name in names for family in MODELS.get(name, (name,))}
    return [family for family in EVERY if family in wanted]


def recorded(folder):
    """Torna el pla del registre d'una família, o `None` si encara no en té.

    El pla són les configuracions de cada seed.
    """
    path = folder / paths.FAMILY_RECORD
    if not path.exists():
        return None
    found = json.loads(path.read_text(encoding="utf-8"))["configurations"]
    # JSON escriu les claus com a text.
    return {int(seed): planned for seed, planned in found.items()}


def missing(folder):
    """Torna les seeds a les quals falta algun fitxer.

    Cada una, amb les configuracions que encara no tenen el seu fitxer.
    """
    found = {}
    for seed, planned in recorded(folder).items():
        absent = [
            one
            for one in planned
            if not (folder / paths.MODEL.format(seed=seed, rank=one["rank"])).exists()
        ]
        if absent:
            found[seed] = absent
    return found


def complete(folder):
    """Diu si una família té pla per a cada seed i tots els fitxers que el pla diu.

    Un registre sense `settings` és d'un altre codi, i no ho diu.
    """
    path = folder / paths.FAMILY_RECORD
    if not path.exists():
        return False
    found = json.loads(path.read_text(encoding="utf-8"))
    if "settings" not in found:
        return False
    planned = {int(seed) for seed in found["configurations"]}
    return planned >= set(found["settings"]["SEEDS"]) and not missing(folder)


def finished(where):
    """Diu si una passada és acabada, amb tots els fitxers del pla de cada família.

    Les famílies són les de tots els models. Es llegeix dels fitxers cada vegada, i no
    es desa enlloc.
    """
    return all(complete(where / family) for family in EVERY)
