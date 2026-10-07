"""Figures que comparen les famílies d'una execució, dibuixades al seu directori.

Les mateixes que el camí endavant deixa dins de cada família, però amb els
pronòstics de totes alhora, que és el que es compara. Totes les carpetes han de
ser de la mateixa execució, i les figures van a la que les conté.
"""
import argparse
import pathlib

from epf import matrix
from epf.data import price
from epf.results import combination, figures, run_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=pathlib.Path, nargs="+",
                        help="directori d'una execució o d'una de les seves famílies")
    folders = run_files.family_folders(parser.parse_args().folder)

    parents = {folder.resolve().parent for folder in folders}
    if len(parents) != 1:
        raise SystemExit(f"les carpetes són de {len(parents)} execucions diferents: "
                         f"{sorted(str(parent) for parent in parents)}")

    actual = matrix.price_only(price.load()).actual
    into = parents.pop()
    figures.shared(into, combination.comparison(folders, actual), actual)
    print(f"comparació de {len(folders)} famílies desada a {into}")
