"""Els processos que fan les tasques d'una cerca o d'una passada, uns quants alhora.

Cada procés és un intèrpret nou, engegat amb `spawn`, que llegeix el límit de fils de
l'entorn quan carrega numpy, LightGBM i TensorFlow, i el pare l'hi posa abans
d'engegar-ne cap. Amb uns altres fils un ajust dóna uns altres números, de manera que el
límit és el mateix a tots els processos, quan n'hi ha un de sol també.
"""

import concurrent.futures
import multiprocessing
import os
import traceback


def environment(threads):
    """El límit de fils de cada procés, amb els noms amb què el llegeixen les
    biblioteques. Un sol fil entre operacions de TensorFlow, el valor amb què es va
    mesurar el límit."""
    return {
        "OMP_NUM_THREADS": str(threads),
        "TF_NUM_INTRAOP_THREADS": str(threads),
        "TF_NUM_INTEROP_THREADS": "1",
    }


def done(tasks, workers, threads):
    """Cada tasca, `(etiqueta, funció, arguments)`, feta per un de `workers` processos
    de `threads` fils cada un. Torna el resultat de cada etiqueta i les etiquetes de les
    que han petat. Una tasca que peta no atura les altres, i es diu en el moment amb el
    seu error. La funció ha de viure en un mòdul, perquè un procés nou la importa pel
    seu nom."""
    os.environ.update(environment(threads))
    results, failed = {}, []
    with concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        running = {
            pool.submit(function, *arguments): label
            for label, function, arguments in tasks
        }
        for future in concurrent.futures.as_completed(running):
            label = running[future]
            try:
                results[label] = future.result()
            except Exception as error:
                failed.append(label)
                print(
                    f"{label}: ha petat\n{''.join(traceback.format_exception(error))}",
                    flush=True,
                )
    return results, failed
