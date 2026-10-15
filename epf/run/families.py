"""Els noms que pren `--family`: cada família pel seu, i cada model, una carpeta de
`models/`, per totes les seves. On un model i una família es diuen igual, el nom és el del
model, de manera que `lightgbm` vol dir l'arbre i la seva prova de datació.
"""

# Cada model amb les seves famílies. Les xarxes són un model, `nn`, i cada una de les seves
# dues carpetes, `dnn` i `ddnn`, també.
MODELS = {
    "lightgbm": ("lightgbm", "date_probe"),
    "elastic_net": ("elastic_net",),
    "nn": ("dnn", "ddnn_normal", "ddnn_jsu"),
    "dnn": ("dnn",),
    "ddnn": ("ddnn_normal", "ddnn_jsu"),
    "floor": ("floor",),
    "naive": ("naive_previous_day", "naive_previous_week", "naive_similar_day"),
}


def named(names, among):
    """Les famílies d'`among` que diu `names`, en l'ordre d'`among`; totes si `names` és
    `None`. Refusa un nom que no és cap família ni cap model, i uns noms entre els quals no
    n'hi ha cap d'`among`."""
    if names is None:
        return list(among)
    families = {family for model in MODELS.values() for family in model}
    unknown = [name for name in names if name not in MODELS and name not in families]
    if unknown:
        raise ValueError(f"{unknown} no són cap família ni cap model; les famílies són "
                         f"{sorted(families)} i els models {sorted(MODELS)}")
    wanted = {family for name in names for family in MODELS.get(name, (name,))}
    found = [family for family in among if family in wanted]
    if not found:
        raise ValueError(f"{names} no diuen cap de {list(among)}")
    return found
