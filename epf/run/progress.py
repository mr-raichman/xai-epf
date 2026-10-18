"""El bloc que mostra una cerca o una passada mentre es fa: una línia per família amb
les llavors acabades, les fallades a sota i el temps transcorregut.

En un terminal el bloc es reescriu al seu lloc amb els codis del cursor; sense, o a una
consola de Windows que no admet el mode VT, s'escriu una línia per llavor acabada.
"""

import math
import os
import sys
import time

# Les fallades que el bloc mostra; la resta es compten en una línia, perquè el bloc no
# passi de la pantalla, per sobre de la qual el cursor no pot tornar.
SHOWN_FAILURES = 5

# Puja el cursor n línies, i esborra del cursor al final de la pantalla.
UP = "\x1b[{}A"
ERASE_BELOW = "\x1b[J"


def _live(stream):
    """Diu si `stream` és un terminal que actua sobre els codis del cursor."""
    if not stream.isatty():
        return False
    if os.name != "nt":
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
    mode = ctypes.c_uint32()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    # ENABLE_VIRTUAL_TERMINAL_PROCESSING; zero vol dir que la consola no l'admet.
    return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))


def elapsed(seconds):
    """Torna una durada com `h:mm:ss`."""
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"


def counted(failed, order):
    """Compta les llavors fallades, `(família, llavor)`, i en nomena la primera en
    l'ordre de les famílies, `order`."""
    family, seed = min(failed, key=lambda one: (order.index(one[0]), one[1]))
    first = f"{family} llavor {seed}"
    if len(failed) == 1:
        return f"1 llavor ha fallat, {first}"
    return f"{len(failed)} llavors han fallat, la primera {first}"


class Block:
    """El bloc d'una ordre: `title` a la primera línia, les llavors que planifica cada
    família, `totals`, i les que ja tenia acabades, `done`.

    Escriu a stderr, com tot el que mostra la línia d'ordres.
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
        """Compta una llavor acabada."""
        self.done[family] += 1
        if self.live:
            self._draw()
        else:
            self._write(f"{family}: {self._count(family)}\n")

    def fail(self, family, seed, message):
        """Compta una llavor fallada i en mostra el missatge de l'error."""
        self.failed[family] += 1
        line = f"error: {family} llavor {seed} ha fallat: {message}"
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
            self._draw()
            self._write("\n")
        else:
            self._write(f"\n{self._time()}\n\n")

    def _count(self, family, width=0):
        """Torna `3 de 10 llavors`, les acabades d'una família alineades a `width`."""
        total = self.totals[family]
        seeds = "llavor" if total == 1 else "llavors"
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
                lines.append("i 1 llavor més ha fallat")
            elif hidden > 1:
                lines.append(f"i {hidden} llavors més han fallat")
            lines.append("")
        lines.append(self._time())
        return lines

    def _draw(self):
        text = "\n".join(self._lines())
        back = UP.format(self.shown) + "\r" + ERASE_BELOW if self.shown else ""
        self._write(back + text + "\n")
        # Les files de pantalla que ocupa, on una línia més ampla que el terminal en fa
        # més d'una; un terminal que no diu la seva amplada, 0, no en parteix cap.
        # shortcut: un bloc més alt que la pantalla en deixa còpies de la part de dalt
        # a l'historial del terminal; mostrar només les fallades que hi càpiguen.
        columns = os.get_terminal_size(self.stream.fileno()).columns
        self.shown = sum(
            max(1, math.ceil(len(row) / columns)) if columns else 1
            for row in text.split("\n")
        )

    def _write(self, text):
        self.stream.write(text)
        self.stream.flush()
