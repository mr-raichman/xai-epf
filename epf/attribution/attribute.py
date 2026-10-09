"""Explica una execució ja feta, dels ajustos que va desar.

Segona passada, a part del camí endavant: aquell ajusta, desa l'ajust i puntua, i
aquesta carrega aquells ajustos i n'extreu l'atribució. Separar-les té un preu i una
raó. El preu és que un directori d'execució pot semblar sencer i no tenir cap `.npz`
fins que això s'executa; el senyal és que el fitxer no hi és, i no cap marca que ho
digui. La raó és que explicar no és puntuar: canviar un mètode d'atribució no ha de
costar un camí endavant.

Els ajustos es llegeixen i no es refan. Un ajust refet no és el mateix ajust: fins que
`boosting.lightgbm` no va demanar `deterministic`, dos ajustos del mateix dia donaven
arbres diferents amb la mateixa predicció, i per tant SHAP diferent. Llegir-los és el
que fa que l'explicació sigui de la predicció que té al costat, i
`stored.refuse_inconsistent` és qui ho comprova: la base més les contribucions contra
el CSV, a través de les dues passades.
"""
from epf import matrix
from epf.attribution import figures, fitted, stored
from epf.data import price
from epf.models.families import families, matrices
from epf.results import run_files


def attribute(folder, tables):
    """Escriu l'atribució d'una família i la dibuixa, de l'ajust que va predir cada dia.

    Torna el nom que ha escrit, o `None` si el directori no s'explica: el registre de
    la combinació no anomena cap família, i hi ha famílies sense cap mètode
    d'atribució. Valida tot el que necessita abans de tocar el disc: la marca dels
    ajustos i que hi hagi un ajust per cada dia predit.

    El `.npz` es diu per la família, que és també el nom del pronòstic de capçalera: és
    contra aquell CSV que es comprova l'additivitat, perquè és la predicció que l'ajust
    va donar.
    """
    record = run_files.load_record(folder)
    name = record.get("family")
    family = families().get(name)
    if family is None or family.attribute is None:
        return None

    forecasts = run_files.load_forecasts(folder)
    store = fitted.folder(record["tier"], record["stage"], name)
    fitted.refuse_mismatch(store, record)
    days = forecasts[name].index
    absent = [day for day in days
              if not (store / fitted.MODEL.format(day=day.date())).exists()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} dies predits sense cap ajust desat a {store}, "
            f"el primer {absent[0].date()}")

    X, _ = tables[family.matrix]
    attributions = [family.attribute(fitted.load(store, day), X, day) for day in days]
    stored.save(folder, name, days, attributions)
    worst = stored.refuse_inconsistent(folder, name, forecasts[name])
    if worst is not None:
        print(f"  {name}: additivitat a {worst:.1e} EUR/MWh", flush=True)
    figures.draw(folder, name)
    return name


def main(arguments):
    folders = run_files.family_folders(arguments.folder)
    tables = matrices(matrix.price_only(price.load()))

    failed = []
    for folder in folders:
        print(f"{folder.name}:", flush=True)
        try:
            written = attribute(folder, tables)
        except Exception as error:
            print(f"  ha fallat: {error}", flush=True)
            failed.append(folder.name)
        else:
            print(f"  {written}" if written else "  no s'atribueix", flush=True)
    if failed:
        print(f"no s'ha explicat: {', '.join(failed)}")
        return 1
    return 0
