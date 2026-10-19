"""Els processos que fan les tasques d'una cerca o d'una passada, uns quants alhora.

Només el pare escriu al terminal, on el bloc no es podria tornar a escriure si hi
caigués una línia d'un procés.
"""

import concurrent.futures
import multiprocessing
import os
import traceback

from epf import Refusal


class Failure(Exception):
    """La fallada d'una tasca que no és un refús, amb la traça del seu procés.

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


def _task(function, *arguments):
    """Fa una tasca, i una fallada que no és un refús en surt com a `Failure`."""
    try:
        return function(*arguments)
    except Refusal:
        raise
    except Exception as error:
        trace = "".join(traceback.format_exception(error))
        raise Failure(f"{type(error).__name__}: {error}", trace) from None


def done(tasks, workers, threads, log, block):
    """Fa les tasques en `workers` processos de `threads` fils cada un.

    Cada tasca és `((família, llavor), funció, arguments)`, i la funció ha de viure en
    un mòdul, perquè un procés nou la importa pel seu nom. Torna el resultat de cada
    etiqueta i les etiquetes de les que han fallat. Una tasca que falla no atura les
    altres. `block` mostra cada llavor acabada o fallada, i el temps cada segon, i es
    tanca en acabar. La sortida dels processos s'afegeix a `log`, i la traça de cada
    fallada que no és un refús també. Deixa el límit de fils a l'entorn d'aquest procés.
    """
    # Abans d'engegar cap procés: cada un, amb `spawn`, és un intèrpret nou que llegeix
    # el límit en carregar NumPy, LightGBM i TensorFlow.
    os.environ.update(environment(threads))
    log.parent.mkdir(parents=True, exist_ok=True)
    results, failed = {}, []
    with concurrent.futures.ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_silenced,
        initargs=(str(log),),
    ) as pool:
        running = {
            pool.submit(_task, function, *arguments): label
            for label, function, arguments in tasks
        }
        waiting = set(running)
        while waiting:
            ready, waiting = concurrent.futures.wait(
                waiting, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in ready:
                label = running[future]
                family, seed = label
                trace = None
                try:
                    results[label] = future.result()
                    block.finish(family)
                    continue
                except Refusal as refusal:
                    message = str(refusal)
                except Failure as failure:
                    message, trace = failure.line, failure.trace
                except Exception as error:
                    # El procés mateix, com un que el sistema mata per falta de memòria.
                    message = f"{type(error).__name__}: {error}"
                    trace = "".join(traceback.format_exception(error))
                failed.append(label)
                block.fail(family, seed, message)
                if trace:
                    with open(log, "a", encoding="utf-8") as handle:
                        handle.write(f"{family} llavor {seed} ha fallat:\n{trace}")
            block.tick()
    block.close()
    return results, failed
