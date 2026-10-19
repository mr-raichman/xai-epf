"""El bloc que mostra una cerca o una passada mentre es fa."""

import os
import sys

from epf.terminal import ERASE_BELOW, UP, live

# Les fallades que el bloc mostra; la resta es compten en una línia, perquè el bloc no
# passi de la pantalla, per sobre de la qual el cursor no pot tornar.
SHOWN_FAILURES = 5

# El final d'una línia tallada.
MORE = "..."


def _cut(line, columns):
    """Torna la primera fila de `line` tallada a `columns` columnes.

    Si en treu res, acaba amb `MORE`. Amb 0 columnes, que és com respon un terminal que
    no diu la seva amplada, no la talla pel llarg.
    """
    first, newline, _ = line.partition("\n")
    # Una columna de marge: hi ha terminals que parteixen la línia que hi arriba.
    room = columns - 1 if columns else len(first) + len(MORE)
    if len(first) <= room and not newline:
        return first
    return first[: max(room - len(MORE), 0)] + MORE


def counted(failed, order):
    """Torna `2 seeds han fallat, la primera lightgbm seed 3`.

    `failed` són les seeds fallades, `(família, seed)`, i la primera és en l'ordre de
    les famílies, `order`.
    """
    family, seed = min(failed, key=lambda one: (order.index(one[0]), one[1]))
    first = f"{family} seed {seed}"
    if len(failed) == 1:
        return f"1 seed ha fallat, {first}"
    return f"{len(failed)} seeds han fallat, la primera {first}"


class Block:
    """El bloc d'una comanda, que escriu a stderr.

    `title` va a la primera línia, i `unit` és el que compta dins de cada seed, `trial`
    o `fit`. `planned` són les seeds de cada família, cada una amb les unitats que
    planifica i les que ja tenia acabades, una parella `(unitats, acabades)`. `failed`
    són les seeds fallades, `(família, seed)`, en l'ordre en què han fallat.
    """

    def __init__(self, title, unit, planned):
        self.title = title
        self.unit = unit
        self.units = {
            family: {seed: units for seed, (units, _) in seeds.items()}
            for family, seeds in planned.items()
        }
        self.done = {
            family: {seed: done for seed, (_, done) in seeds.items()}
            for family, seeds in planned.items()
        }
        self.failed = []
        self.failures = []
        self.stream = sys.stderr
        self.live = live(self.stream)
        self.shown = 0
        if self.live:
            self._draw()
        else:
            self._write(f"{title}\n\n")

    def advance(self, family, seed):
        """Compta una unitat acabada d'una seed, i la seed quan és la seva última.

        Fora d'un terminal, només es veu la seed acabada.
        """
        self.done[family][seed] += 1
        if self.live:
            self._draw()
        elif self.done[family][seed] == self.units[family][seed]:
            self._write(f"{family}: {self._count(family)}\n")

    def fail(self, family, seed, message):
        """Compta una seed fallada i en mostra el missatge de l'error.

        Només la primera vegada: les unitats de la seed que ja es feien poden fallar
        també.
        """
        if self.stopped(family, seed):
            return
        self.failed.append((family, seed))
        line = f"error: {family} seed {seed} ha fallat: {message}"
        if self.live:
            self.failures.append(line)
            self._draw()
        else:
            self._write(line + "\n")

    def stopped(self, family, seed):
        """Diu si una seed ha fallat, i ja no se n'ha de fer cap altra unitat."""
        return (family, seed) in self.failed

    def close(self):
        """Deixa l'estat final, amb les línies senceres en un terminal."""
        if self.live:
            self._draw(whole=True)

    def _count(self, family, widths=(0, 0, 0)):
        """Torna `3/10 seeds    27/100 trials` d'una família.

        `widths` són les amplades que alineen les famílies: la d'un total de seeds, la
        de la paraula `seeds` i la d'un total d'unitats.
        """
        planned, done = self.units[family], self.done[family]
        seeds = len(planned)
        done_seeds = sum(done[seed] >= units for seed, units in planned.items())
        units, done_units = sum(planned.values()), sum(done.values())
        seed_word = "seed" if seeds == 1 else "seeds"
        unit_word = self.unit if units == 1 else f"{self.unit}s"
        seed_width, word_width, unit_width = widths
        seed_count = f"{done_seeds:>{seed_width}}/{seeds:<{seed_width}}"
        unit_count = f"{done_units:>{unit_width}}/{units:<{unit_width}}"
        return f"{seed_count} {seed_word:<{word_width}}    {unit_count} {unit_word}"

    def _lines(self):
        width = max(len(family) for family in self.units)
        seeds = [len(units) for units in self.units.values()]
        widths = (
            max(len(str(count)) for count in seeds),
            max(len("seed" if count == 1 else "seeds") for count in seeds),
            max(len(str(sum(units.values()))) for units in self.units.values()),
        )
        lines = [self.title, ""]
        for family in self.units:
            line = f"{family:<{width}}  {self._count(family, widths)}"
            count = sum(failed == family for failed, _ in self.failed)
            if count:
                line += f", {count} {'ha' if count == 1 else 'han'} fallat"
            lines.append(line)
        if self.failures:
            lines.append("")
            lines += self.failures[:SHOWN_FAILURES]
            hidden = len(self.failures) - SHOWN_FAILURES
            if hidden == 1:
                lines.append("i 1 seed més ha fallat")
            elif hidden > 1:
                lines.append(f"i {hidden} seeds més han fallat")
        return lines

    def _draw(self, whole=False):
        """Torna a escriure el bloc al seu lloc, cada línia tallada a una fila.

        Amb `whole`, quan ja no s'ha de tornar a escriure, les línies van senceres.
        """
        lines = self._lines()
        if not whole:
            # Una línia que es partís en dues files faria pujar el cursor massa poc, i
            # un bloc més alt que la pantalla no es pot tornar a escriure.
            columns = os.get_terminal_size(self.stream.fileno()).columns
            lines = [_cut(line, columns) for line in lines]
        back = UP.format(self.shown) + "\r" + ERASE_BELOW if self.shown else ""
        self._write(back + "\n".join(lines) + "\n")
        self.shown = len(lines)

    def _write(self, text):
        self.stream.write(text)
        self.stream.flush()
