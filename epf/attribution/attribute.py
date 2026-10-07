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
import argparse
import pathlib
import sys

from epf import matrix
from epf.attribution import figures, fitted, stored
from epf.data import price
from epf.models.families import families, matrices
from epf.results import run_files


def explained(family):
    """Els pronòstics d'una família que tenen manera d'atribuir-se."""
    return {name: of_day for name, of_day in family.forecasts.items()
            if of_day.attribute is not None}


def attribute(folder, matrices):
    """Escriu l'atribució de cada pronòstic explicable d'una família i la dibuixa.

    Torna els noms que ha escrit. Valida tot el que necessita abans de tocar el disc:
    la marca dels ajustos i que hi hagi un ajust per cada dia predit.
    """
    record = run_files.load_record(folder)
    family = families()[record["family"]]
    forecasts = run_files.load_forecasts(folder)
    wanted = {name: of_day for name, of_day in explained(family).items()
              if name in forecasts}
    if not wanted:
        return []

    store = fitted.folder(record["tier"], record["stage"], record["family"])
    fitted.refuse_mismatch(store, record)
    days = forecasts[next(iter(wanted))].index
    absent = [day for day in days if not (store / fitted.MODEL.format(day=day.date())).exists()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} dies predits sense cap ajust desat a {store}, "
            f"el primer {absent[0].date()}")

    X, _ = matrices[family.matrix]
    written = []
    for name, of_day in sorted(wanted.items()):
        attributions = [of_day.attribute(fitted.load(store, day), X, day) for day in days]
        stored.save(folder, name, days, attributions)
        worst = stored.refuse_inconsistent(folder, name, forecasts[name])
        if worst is not None:
            print(f"  {name}: additivitat a {worst:.1e} EUR/MWh", flush=True)
        figures.draw(folder, name)
        written.append(name)
    return written


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=pathlib.Path, nargs="+",
                        help="directori d'una execució o d'una de les seves famílies")
    return parser.parse_args()


def main():
    folders = run_files.family_folders(parse().folder)
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
            print(f"  {', '.join(written)}" if written
                  else "  cap pronòstic amb manera d'atribuir-se", flush=True)
    if failed:
        print(f"no s'ha explicat: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
