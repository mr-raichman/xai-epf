"""El bloc que mostra una cerca o una passada mentre es fa."""

import os
import sys
import time

# Les fallades que el bloc mostra; la resta es compten en una línia, perquè el bloc no
# passi de la pantalla, per sobre de la qual el cursor no pot tornar.
SHOWN_FAILURES = 5

# Puja el cursor n línies, i esborra del cursor al final de la pantalla.
UP = "\x1b[{}A"
ERASE_BELOW = "\x1b[J"

# El final d'una línia tallada.
MORE = "..."


def _live(stream):
    """Diu si `stream` és un terminal que actua sobre els codis del cursor."""
    if not stream.isatty():
        return False
    if os.name != "nt":
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    # La sortida d'errors, STD_ERROR_HANDLE.
    handle = kernel32.GetStdHandle(-12)
    mode = ctypes.c_uint32()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    # ENABLE_VIRTUAL_TERMINAL_PROCESSING; zero vol dir que la consola no l'admet.
    return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))


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


def elapsed(seconds):
    """Torna una durada com `h:mm:ss`."""
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"


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

    `title` va a la primera línia, `totals` són les seeds que planifica cada família, i
    `done` les que ja tenia acabades.
    """

    def __init__(self, title, totals, done=None):
        self.title = title
        self.totals = dict(totals)
        self.done = {family: 0 for family in self.totals} | dict(done or {})
        self.failed = {family: 0 for family in self.totals}
        self.failures = []
        self.started = time.monotonic()
        self.stream = sys.stderr
        self.live = _live(self.stream)
        self.shown = 0
        if self.live:
            self._draw()
        else:
            self._write(f"{title}\n\n")

    def finish(self, family):
        """Compta una seed acabada."""
        self.done[family] += 1
        if self.live:
            self._draw()
        else:
            self._write(f"{family}: {self._count(family)}\n")

    def fail(self, family, seed, message):
        """Compta una seed fallada i en mostra el missatge de l'error."""
        self.failed[family] += 1
        line = f"error: {family} seed {seed} ha fallat: {message}"
        if self.live:
            self.failures.append(line)
            self._draw()
        else:
            self._write(line + "\n")

    def tick(self):
        """Posa al dia el temps transcorregut."""
        if self.live:
            self._draw()

    def close(self):
        """Deixa l'estat final i la línia en blanc que va abans de la de tancament."""
        if self.live:
            self._draw(whole=True)
            self._write("\n")
        else:
            self._write(f"\n{self._time()}\n\n")

    def _count(self, family, width=0):
        """Torna `3 de 10 seeds`, les acabades d'una família alineades a `width`."""
        total = self.totals[family]
        seeds = "seed" if total == 1 else "seeds"
        return f"{self.done[family]:>{width}} de {total} {seeds}"

    def _time(self):
        return f"Temps transcorregut: {elapsed(time.monotonic() - self.started)}"

    def _lines(self):
        width = max(len(family) for family in self.totals)
        lines = [self.title, ""]
        for family, total in self.totals.items():
            line = f"{family:<{width}}  {self._count(family, len(str(total)))}"
            count = self.failed[family]
            if count:
                line += f", {count} {'ha' if count == 1 else 'han'} fallat"
            lines.append(line)
        lines.append("")
        if self.failures:
            lines += self.failures[:SHOWN_FAILURES]
            hidden = len(self.failures) - SHOWN_FAILURES
            if hidden == 1:
                lines.append("i 1 seed més ha fallat")
            elif hidden > 1:
                lines.append(f"i {hidden} seeds més han fallat")
            lines.append("")
        lines.append(self._time())
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
