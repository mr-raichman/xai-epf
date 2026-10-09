"""Figures que comparen les famílies d'una execució, dibuixades al seu directori.

Les mateixes que el camí endavant deixa dins de cada família, però amb els
pronòstics de totes alhora, que és el que es compara. Totes les carpetes han de
ser de la mateixa execució, i les figures van a la que les conté.
"""
from epf import matrix
from epf.data import price
from epf.results import combination, figures, run_files


def main(arguments):
    folders = run_files.family_folders(arguments.folder)
    parents = {folder.resolve().parent for folder in folders}
    if len(parents) != 1:
        raise SystemExit(f"les carpetes són de {len(parents)} execucions diferents: "
                         f"{sorted(str(parent) for parent in parents)}")

    actual = matrix.price_only(price.load()).actual
    into = parents.pop()
    forecasts = combination.comparison(folders)
    figures.shared(into, forecasts, actual)
    print(f"comparació de {run_files.drawn(len(forecasts))} desada a {into}")
