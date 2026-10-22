"""El bloc que mostra una passada mentre es fa."""

import os
import sys

from epf.terminal import ERASE_BELOW, UP, live

# El final d'una línia tallada.
MORE = "..."

# Cada procés que fa una unitat d'una família, a la seva línia.
WORKER = "·"

# Les unitats d'una seed, en l'ordre de les seves columnes: els trials de la seva cerca
# i els seus ajustos.
KINDS = ("trial", "fit")


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


def _word(word, total):
    """Torna `word` en singular per a una sola unitat, i en plural per a la resta."""
    return word if total == 1 else f"{word}s"


def _tally(done, total, word, digits=0):
    """Torna `3/10 seeds`, amb les xifres de cada nombre a `digits`."""
    return f"{done:>{digits}}/{total:<{digits}} {_word(word, total)}"


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


def aligned(columns):
    """Torna la línia de recomptes de cada família, alineada amb les altres, pel nom.

    `columns` dona de cada família, en ordre, el recompte de les seves seeds i el de
    cada tipus de `KINDS`, `(acabades, total, paraula)`, o `None` el d'un tipus que no
    fa: `floor  2/2 seeds    2/2 fits`.
    """
    width = max(len(family) for family in columns)
    # Cada columna que alguna família té, amb les xifres del seu total més llarg i les
    # lletres de la seva paraula més llarga, que l'alineen.
    shown = []
    for number in range(1 + len(KINDS)):
        present = [found[number] for found in columns.values() if found[number]]
        if present:
            digits = max(len(str(total)) for _, total, _ in present)
            letters = max(len(_word(word, total)) for _, total, word in present)
            shown.append((number, digits, letters))
    counts = {}
    for family, found in columns.items():
        parts = []
        for number, digits, letters in shown:
            text = _tally(*found[number], digits) if found[number] else ""
            parts.append(f"{text:<{2 * digits + 2 + letters}}")
        counts[family] = f"{family:<{width}}  {'    '.join(parts)}".rstrip()
    return counts


class Block:
    """El bloc d'una passada, que escriu a stderr.

    `title` va a la primera línia, i a sota, cada seed que falla, sencera; el bloc que
    es torna a escriure són les línies de les famílies. `planned` són les seeds de cada
    família, cada una amb una parella `(unitats, acabades)` per tipus de `KINDS`, sense
    el que la família no fa: els trials d'una família que es cerca, els ajustos d'una
    que s'ajusta. `failed` són les seeds fallades, `(família, seed)`, en l'ordre en què
    han fallat.
    """

    def __init__(self, title, planned):
        self.title = title
        self.units = {
            family: {
                seed: {kind: units for kind, (units, _) in kinds.items()}
                for seed, kinds in seeds.items()
            }
            for family, seeds in planned.items()
        }
        self.done = {
            family: {
                seed: {kind: done for kind, (_, done) in kinds.items()}
                for seed, kinds in seeds.items()
            }
            for family, seeds in planned.items()
        }
        self.working = dict.fromkeys(planned, 0)
        self.failed = []
        self.stream = sys.stderr
        self.live = live(self.stream)
        self.shown = 0
        self._write(f"{title}\n\n")
        if self.live:
            self._draw()

    def start(self, family):
        """Compta un procés que comença una unitat d'una família."""
        self.working[family] += 1
        if self.live:
            self._draw()

    def advance(self, family, seed, kind):
        """Compta una unitat acabada d'una seed, i la seed quan és la seva última.

        `kind` és el tipus de la unitat, de `KINDS`. Fora d'un terminal, només es veu la
        seed acabada.
        """
        self.working[family] -= 1
        self.done[family][seed][kind] += 1
        if self.live:
            self._draw()
        elif kind == self._last(family, seed) and self._finished(family, seed):
            self._write(f"{family}: {self._count(family)}\n")

    def plan(self, family, seed, units):
        """Posa els ajustos d'una seed quan se'n fa el pla, `units`."""
        self.units[family][seed]["fit"] = units
        if self.live:
            self._draw()

    def fail(self, family, seed, message):
        """Compta una unitat fallada i, la primera vegada, la seed i el seu missatge.

        Les unitats de la seed que ja es feien poden fallar també.
        """
        self.working[family] -= 1
        self.stop(family, seed, message)

    def stop(self, family, seed, message):
        """Compta una seed fallada, la primera vegada, amb el seu missatge."""
        if not self.stopped(family, seed):
            self.failed.append((family, seed))
            line = f"error: {family} seed {seed} ha fallat: {message}"
            if self.live:
                # A sobre de les famílies, i ja no es torna a escriure: hi pot ocupar
                # tantes files com li calguin.
                self._write(self._erased() + line + "\n")
                self.shown = 0
            else:
                self._write(line + "\n")
        if self.live:
            self._draw()

    def stopped(self, family, seed):
        """Diu si una seed ha fallat, i ja no se n'ha de fer cap altra unitat."""
        return (family, seed) in self.failed

    def close(self):
        """Deixa l'estat final, amb les línies senceres en un terminal.

        Cap procés no hi fa res: o han acabat, o un Ctrl+C els ha aturat.
        """
        self.working = dict.fromkeys(self.working, 0)
        if self.live:
            self._draw(whole=True)

    def _last(self, family, seed):
        """Torna el tipus de la unitat amb què acaba una seed.

        Són els ajustos, i els trials d'una família que es cerca només per a una altra.
        """
        return "fit" if "fit" in self.units[family][seed] else "trial"

    def _finished(self, family, seed):
        last = self._last(family, seed)
        return self.done[family][seed][last] >= self.units[family][seed][last]

    def _columns(self, family):
        """Torna el recompte de les seeds d'una família i el de cada tipus de `KINDS`.

        Cada un és `(acabades, total, paraula)`, i el d'un tipus que la família no fa,
        `None`.
        """
        seeds = self.units[family]
        found = [
            (sum(self._finished(family, seed) for seed in seeds), len(seeds), "seed")
        ]
        for kind in KINDS:
            if any(kind in units for units in seeds.values()):
                done = sum(self.done[family][seed][kind] for seed in seeds)
                total = sum(units[kind] for units in seeds.values())
                found.append((done, total, kind))
            else:
                found.append(None)
        return found

    def _count(self, family):
        """Torna `3/10 seeds    212/500 trials    3/10 fits` d'una família."""
        return "    ".join(
            _tally(*column) for column in self._columns(family) if column
        )

    def _lines(self):
        counts = aligned({family: self._columns(family) for family in self.units})
        # Les línies no fan la mateixa amplada.
        longest = max(len(count) for count in counts.values())
        # Separades de les fallades de sobre.
        lines = [""] if self.failed else []
        for family, line in counts.items():
            if self.working[family]:
                line = f"{line:<{longest}}    {WORKER * self.working[family]}"
            count = sum(failed == family for failed, _ in self.failed)
            if count:
                line += f", {count} {'ha' if count == 1 else 'han'} fallat"
            lines.append(line)
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
        self._write(self._erased() + "\n".join(lines) + "\n")
        self.shown = len(lines)

    def _erased(self):
        """Torna el que puja el cursor a l'inici del bloc escrit i l'esborra."""
        return UP.format(self.shown) + "\r" + ERASE_BELOW if self.shown else ""

    def _write(self, text):
        self.stream.write(text)
        self.stream.flush()
