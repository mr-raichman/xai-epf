"""Figures que comparen pronòstics, dibuixades al directori que es dóna.

Les mateixes sis que el camí endavant deixa dins de cada família. El directori d'una
configuració les rep amb els pronòstics de totes les famílies i la combinació alhora,
que és el que de fet es compara; el d'una família les rep només amb els seus.

Es dibuixa al directori donat i no al seu pare. Mentre es dibuixava al pare, donar una
família sola hi escrivia la comparació de l'execució sencera amb una sola sèrie, i la
que hi havia desapareixia fins que alguna cosa la refeia.
"""
from epf import matrix
from epf.data import forecasts as day_ahead
from epf.data import price
from epf.results import combination, figures, run_files


def main(arguments):
    actual = matrix.lago(price.load(), day_ahead.load()).actual
    for into in arguments.folder:
        forecasts = combination.comparison(run_files.family_folders([into]))
        figures.shared(into, forecasts, actual)
        print(f"comparació de {run_files.drawn(len(forecasts))} desada a {into}")
