"""El que s'escriu al terminal mentre es fa una comanda.

En un terminal, el temps transcorregut és l'última línia mentre la comanda es fa, i es
torna a escriure cada segon, de manera que es veu que no s'ha penjat.
"""

import os
import sys
import threading
import time

# Puja el cursor n línies, i esborra del cursor al final de la pantalla.
UP = "\x1b[{}A"
ERASE_BELOW = "\x1b[J"


def live(stream):
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


def interactive():
    """Diu si qui fa la comanda pot respondre: si stdin i stderr són un terminal."""
    return sys.stdin.isatty() and sys.stderr.isatty()


def asked(question):
    """Fa a stderr una pregunta de sí o no, i torna cert per `y` i fals per `n`.

    Llegeix la resposta d'stdin, en majúscula o en minúscula, i torna a fer la pregunta
    a qualsevol altra. Torna `None` si l'entrada s'acaba sense cap resposta.
    """
    while True:
        print(question, end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if not line:
            return None
        # El terminal ja ha escrit la resposta i el salt de línia, i el rellotge, que no
        # els ha vist, quedaria una fila amunt.
        if isinstance(sys.stderr, Clock):
            sys.stderr.echoed(line)
        answer = line.strip().lower()
        if answer in ("y", "n"):
            return answer == "y"


def elapsed(seconds):
    """Torna una durada com `h:mm:ss`."""
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"


class Clock:
    """El temps transcorregut d'una comanda, l'última línia de stderr dins d'un `with`.

    Mentre dura, `sys.stderr` és el rellotge, que treu el temps abans de cada escriptura
    i el torna a posar a sota quan una línia s'acaba. En sortir, per bé o per una
    excepció, deixa el temps final i una línia en blanc. Fora d'un terminal només
    escriu el final.
    """

    def __init__(self):
        self.stream = sys.stderr
        self.live = live(self.stream)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        # Els dos últims caràcters escrits, al principi com després d'una línia buida.
        self.tail = "\n\n"
        # Les files que el temps té per sobre de la seva, o `None` si no hi és.
        self.shown = None

    def __enter__(self):
        self.started = time.monotonic()
        sys.stderr = self
        if self.live:
            with self.lock:
                self._show()
            threading.Thread(target=self._tick, daemon=True).start()
        return self

    def __exit__(self, *exception):
        self.stopped.set()
        with self.lock:
            self._hide()
            sys.stderr = self.stream
            self._raw(self._line() + "\n\n")

    def write(self, text):
        with self.lock:
            self._hide()
            self._raw(text)
            self.tail = (self.tail + text)[-2:]
            self._show()
        return len(text)

    def flush(self):
        self.stream.flush()

    def echoed(self, text):
        """Compta `text`, que el terminal ja ha escrit, com escrit a la sortida."""
        with self.lock:
            self.tail = (self.tail + text)[-2:]
            self._show()

    def __getattr__(self, name):
        # La resta, com `isatty` i `fileno`, és la de la sortida d'errors de debò.
        return getattr(self.stream, name)

    def _line(self):
        """Torna el temps, separat per una línia en blanc del que hi ha a sobre."""
        newlines = len(self.tail) - len(self.tail.rstrip("\n"))
        seconds = time.monotonic() - self.started
        return "\n" * (2 - newlines) + f"Temps transcorregut: {elapsed(seconds)}"

    def _show(self):
        """Escriu el temps a sota, en un terminal i a l'inici d'una línia."""
        if self.live and not self.stopped.is_set() and self.tail.endswith("\n"):
            line = self._line()
            self._raw(line)
            self.shown = line.count("\n")

    def _hide(self):
        """Esborra el temps i torna el cursor on era abans d'escriure'l."""
        if self.shown is not None:
            # Amb 0, el codi per pujar el cursor en puja una.
            up = UP.format(self.shown) if self.shown else ""
            self._raw("\r" + up + ERASE_BELOW)
            self.shown = None

    def _tick(self):
        while not self.stopped.wait(1):
            with self.lock:
                if self.shown is not None:
                    self._hide()
                    self._show()

    def _raw(self, text):
        self.stream.write(text)
        self.stream.flush()
