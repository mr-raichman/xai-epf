"""Els processos que fan les tasques d'una passada, uns quants alhora.

Només el pare escriu al terminal, on el bloc no es podria tornar a escriure si hi
caigués una línia d'un procés.
"""

import concurrent.futures
import multiprocessing
import multiprocessing.connection
import os
import signal
import threading
import traceback

from epf import Refusal


class Failure(Exception):
    """La fallada d'una tasca que no és un refús, amb el traceback del seu procés.

    `line` és la classe i el missatge de l'error.
    """

    def __init__(self, line, trace):
        super().__init__(line, trace)
        self.line, self.trace = line, trace


def environment(threads):
    """Torna les variables d'entorn que limiten els fils de cada procés a `threads`.

    Entre operacions de TensorFlow, un sol fil, el valor amb què es va mesurar el límit.
    """
    return {
        "OMP_NUM_THREADS": str(threads),
        "TF_NUM_INTRAOP_THREADS": str(threads),
        "TF_NUM_INTEROP_THREADS": "1",
    }


def _silenced(log):
    """Afegeix la sortida i els errors d'aquest procés al fitxer `log`.

    Ho fa als descriptors, de manera que hi va també el que escriu una biblioteca de C.
    """
    with open(log, "a", encoding="utf-8") as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)


def _started(log):
    """Prepara un procés del pool, amb la seva sortida a `log`.

    El procés no fa cas d'un Ctrl+C, que l'atura el pare, i acaba quan el pare ja no hi
    és.
    """
    # Amb el Ctrl+C, que arriba a tots els processos, la tasca fallaria i el procés
    # passaria a la següent.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(target=_orphaned, daemon=True).start()
    _silenced(log)


def _orphaned():
    """Acaba aquest procés quan el pare ja no hi és, i ningú no en recolliria res."""
    multiprocessing.connection.wait([multiprocessing.parent_process().sentinel])
    os._exit(1)


def _task(function, *arguments):
    """Fa una tasca, i una fallada que no és un refús en surt com a `Failure`."""
    try:
        return function(*arguments)
    except Refusal:
        raise
    except Exception as error:
        trace = "".join(traceback.format_exception(error))
        raise Failure(f"{type(error).__name__}: {error}", trace) from None


def _fed(pool, running, tasks, workers, block):
    """Dona al pool tasques de `tasks` fins que n'hi ha `workers` en marxa.

    S'atura abans si `tasks` s'acaba o dona `None`. Apunta cada tasca a `running`, i la
    compta a `block` com a començada.
    """
    while len(running) < workers:
        task = next(tasks, None)
        if task is None:
            return
        label, function, arguments = task
        running[pool.submit(_task, function, *arguments)] = label
        block.start(label[0])


def done(tasks, workers, threads, log, block, on_return=None, on_failure=None):
    """Fa les tasques en `workers` processos de `threads` fils cada un.

    Cada tasca és `((família, seed, tipus, número), funció, arguments)`: el tipus,
    `trial` o `fit`, i el número del trial o el lloc de la configuració. La funció ha
    de viure en un mòdul, perquè un procés nou la importa pel seu nom. De `tasks` se'n
    llegeixen fins que n'hi ha `workers` en marxa, al començament i cada vegada que en
    torna una, i mai no n'hi ha més alhora. Una tasca `None` vol dir que encara no n'hi
    ha cap a punt, i se'n torna a llegir quan en torna una altra; si no n'hi ha cap en
    marxa, s'acaba. Torna el resultat de cada etiqueta que ha tornat. Una tasca que
    falla no atura les altres, i un Ctrl+C les acaba totes alhora, com una excepció de
    `on_return` o de `on_failure`. Cada tasca es compta a `block` quan comença; cada una
    tornada crida `on_return(etiqueta, resultat)` i s'hi compta, i cada una fallada
    crida `on_failure(etiqueta)` i hi compta la seed com a fallada; el bloc es tanca en
    acabar. La sortida dels processos s'afegeix a `log`, i cada fallada també, un refús
    pel seu missatge i qualsevol altra pel seu traceback. Deixa el límit de fils a
    l'entorn d'aquest procés.
    """
    # Abans d'engegar cap procés: cada un, amb `spawn`, és un intèrpret nou que llegeix
    # el límit en carregar NumPy, LightGBM i TensorFlow.
    os.environ.update(environment(threads))
    log.parent.mkdir(parents=True, exist_ok=True)
    tasks = iter(tasks)
    results, running = {}, {}
    with concurrent.futures.ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_started,
        initargs=(str(log),),
    ) as pool:
        try:
            _fed(pool, running, tasks, workers, block)
            while running:
                ready, _ = concurrent.futures.wait(
                    running, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in ready:
                    label = running.pop(future)
                    family, seed, kind, number = label
                    trace = None
                    try:
                        results[label] = future.result()
                    except Refusal as refusal:
                        message = str(refusal)
                    except Failure as failure:
                        message, trace = failure.line, failure.trace
                    except Exception as error:
                        # El procés mateix, com un que el sistema mata per falta de
                        # memòria.
                        message = f"{type(error).__name__}: {error}"
                        trace = "".join(traceback.format_exception(error))
                    else:
                        if on_return is not None:
                            on_return(label, results[label])
                        block.advance(family, seed, kind)
                        _fed(pool, running, tasks, workers, block)
                        continue
                    # Un refús, pel seu missatge, que el bloc talla a l'amplada del
                    # terminal; qualsevol altra fallada, pel seu traceback.
                    with open(log, "a", encoding="utf-8") as handle:
                        handle.write(
                            f"{family} seed {seed} {kind} {number} ha fallat:"
                            + (f" {message}\n" if trace is None else f"\n{trace}")
                        )
                    if on_failure is not None:
                        on_failure(label)
                    block.fail(family, seed, message)
                    # Després de comptar la fallada, que la seed ja no en doni cap més.
                    _fed(pool, running, tasks, workers, block)
        except BaseException:
            # Sense, sortir del `with` esperaria que cada procés acabés la seva tasca.
            for worker in multiprocessing.active_children():
                worker.terminate()
            block.close()
            raise
    block.close()
    return results
