"""Les figures de les xarxes distribucionals, dibuixades del que la passada va desar: la
importància de cada columna i de cada grup, i la dependència de cada sèrie, paràmetre a
paràmetre."""

from epf.contract import figure


def draw(folder, seeds):
    figure.importance(folder, seeds)
    present = figure.present(seeds[0])
    if present:
        figure.dependence(folder, seeds, present)
