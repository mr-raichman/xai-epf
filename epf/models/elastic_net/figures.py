"""Les figures de la xarxa elàstica, dibuixades del que la passada va desar: la
importància de cada columna i de cada grup, i la dependència de cada sèrie, a l'espai
transformat."""

from epf.contract import figure


def draw(folder, seeds):
    figure.importance(folder, seeds)
    present = figure.present(seeds[0])
    if present:
        figure.dependence(folder, seeds, present)
