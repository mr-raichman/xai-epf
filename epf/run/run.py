"""La comanda `epf run`, la passada del protocol de Trebbien et al. (2023).

Cerca i ajusta amb un sol pool: una seed d'una família s'ajusta tan bon punt la seva
cerca té tots els trials, mentre les altres seeds encara es cerquen.
"""

import datetime
import importlib
import json
import time

from epf import Refusal, parameters, paths
from epf.contract import matrix, record
from epf.contract.searches import configurations
from epf.contract.split import split, validation_parts
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT
from epf.run import families, figures, pool, progress, search

# Com diu el refús d'una passada continuada cada empremta que no és la d'ara: la que la
# va començar i aquella amb què s'ha d'acabar.
DIFFERENT = {
    "code": ("un altre codi", "aquell codi"),
    "dataset": ("un altre dataset", "aquell dataset"),
    "environment": ("unes altres biblioteques", "aquelles biblioteques"),
}

# Cada família, en l'ordre en què una passada les ajusta, amb el mòdul del seu model que
# l'ajusta, que s'importa quan es tria la família.
TRAINERS = {
    "ddnn_jsu": "epf.models.nn.ddnn.train",
    "ddnn_normal": "epf.models.nn.ddnn.train",
    "dnn": "epf.models.nn.dnn.train",
    "lightgbm": "epf.models.lightgbm.train",
    "date_probe": "epf.models.lightgbm.date_probe",
    "elastic_net": "epf.models.elastic_net.train",
    "floor": "epf.models.floor.train",
    "naive_similar_day": "epf.models.naive.train",
    "naive_previous_week": "epf.models.naive.train",
    "naive_previous_day": "epf.models.naive.train",
}


def _where(smoke, fingerprints):
    """Torna l'última passada del tipus si no és acabada, o la següent d'avui.

    Refusa continuar-ne una que van començar un altre codi, un altre dataset o unes
    altres biblioteques, perquè les seves famílies no es podrien comparar, i una
    d'assaig començada amb una altra variant que la d'ara, `smoke`, que hi barrejaria
    els valors de dues. Un registre sense una de les empremtes d'ara és d'un altre codi.
    """
    found = paths.runs(smoke)
    if not found or families.finished(found[-1]):
        return paths.next_run(smoke, datetime.date.today())
    latest = found[-1]
    differ, begun = set(), set()
    for path in sorted(latest.glob(f"*/{paths.SEED_RECORD}")):
        made = json.loads(path.read_text(encoding="utf-8"))
        differ |= {
            name for name, value in fingerprints.items() if made.get(name) != value
        }
        begun.add(made.get("smoke"))
    if differ:
        then = [DIFFERENT[name] for name in DIFFERENT if name in differ]
        raise Refusal(
            f"la passada {latest.name} es va començar amb "
            f"{' i '.join(other for other, _ in then)}; continuar amb "
            f"{' i '.join(that for _, that in then)}, o esborrar "
            f"{paths.relative(latest)}/"
        )
    variants = sorted(map(str, begun - {smoke}))
    if variants:
        flags = {name: flag for flag, name in parameters.variants().items()}
        flag = flags.get(variants[0], variants[0])
        raise Refusal(
            f'la passada {latest.name} es va començar amb "{flag}"; continuar amb '
            f'"uv run epf run {flag}", o esborrar {paths.relative(latest)}/'
        )
    return latest


def _train(module, family, folder, seed, configuration, explained_days):
    """Ajusta una configuració d'una família en una seed, una tasca de `pool`.

    Crida el `train` del mòdul del model, `module`, amb un pla d'aquesta seed i aquesta
    configuració soles. Torna quants segons ha trigat.
    """
    started = time.time()
    importlib.import_module(module).train(
        family, folder, {seed: [configuration]}, explained_days
    )
    return time.time() - started


def _ordered(by_seed):
    """Torna un diccionari de seeds, escrites com a text, en ordre de seed."""
    return dict(sorted(by_seed.items(), key=lambda item: int(item[0])))


def _planned(folder, seed, configurations):
    """Desa al registre d'una família el pla d'una seed, que ja no canvia."""
    path = folder / paths.SEED_RECORD
    found = json.loads(path.read_text(encoding="utf-8"))
    found["configurations"][str(seed)] = configurations
    found["configurations"] = _ordered(found["configurations"])
    record.write_record(path, found)


def _timed(folder, seed, rank, seconds):
    """Desa al registre d'una família els segons d'un ajust, al costat dels altres."""
    path = folder / paths.SEED_RECORD
    found = json.loads(path.read_text(encoding="utf-8"))
    ranks = found["seconds"].setdefault(str(seed), {})
    ranks[str(rank)] = round(seconds, 1)
    found["seconds"][str(seed)] = _ordered(ranks)
    found["seconds"] = _ordered(found["seconds"])
    record.write_record(path, found)


def main(arguments):
    """Fa una passada, la comanda `epf run`.

    Continua l'última del seu tipus si no és acabada. Cerca el que falta i, amb els
    mateixos processos, ajusta cada configuració planificada que encara no té el seu
    fitxer. Una seed d'una família que no es cerca rep el pla de seguida, i una d'una
    que es cerca, les millors configuracions de la seva cerca quan la té acabada; el
    pla queda al registre de la família i ja no canvia. Dibuixa les figures si cap seed
    no ha fallat. Refusa abans d'escriure la passada el que no pot fer, i acaba amb un
    refús si alguna seed ha fallat.
    """
    values = parameters.chosen(arguments.smoke)
    chosen = families.named(arguments.family, TRAINERS)
    matrices = matrix.matrices(matrix.series())
    matrix.refuse_missing_days(matrices, DEVELOPMENT, BLACKOUT_WEEK)
    # La partició i les parts de validació, només per refusar-les abans d'escriure res.
    split(values["SEEDS"][0], DEVELOPMENT, BLACKOUT_WEEK)
    validation_parts(values["SEEDS"][0], DEVELOPMENT, BLACKOUT_WEEK)
    fingerprints = record.fingerprints()
    where = _where(arguments.smoke, fingerprints)
    continued = where.exists()
    plans = {
        family: importlib.import_module(TRAINERS[family]).FAMILIES[family]
        for family in chosen
    }
    # Les configuracions que una família guarda de cada seed.
    kept = {
        family: values["BEST"] if plan["kept"] == "BEST" else plan["kept"]
        for family, plan in plans.items()
    }
    # La cerca que llegeix cada família que guarda alguna configuració, en l'ordre de
    # les cerques: la sonda de datació fa cercar l'arbre.
    searched = [
        name
        for name in search.SEARCHERS
        if any(
            kept[family] and plans[family]["searched_as"] == name for family in chosen
        )
    ]
    searching = search.Search(
        arguments.smoke,
        searched,
        values["SEEDS"],
        values["TRIALS"],
        matrices,
        fingerprints,
        where / paths.WORKERS_LOG,
    )
    where.mkdir(parents=True, exist_ok=True)
    matrix.write(matrices, where)
    # Els trials són els de la cerca, que porta cada configuració, i el calendari no
    # canvia mai.
    left_out = {"TRIALS", "DAYS_PER_WEEK", "HOURS_PER_DAY", "HOUR_COLUMNS"}
    written = {name: value for name, value in values.items() if name not in left_out}
    for family in chosen:
        if families.recorded(where / family) is None:
            record.write_record(
                where / family / paths.SEED_RECORD,
                {
                    "family": family,
                    "smoke": arguments.smoke,
                    **fingerprints,
                    "seeds": values["SEEDS"],
                    "explained_days": values["EXPLAINED_DAYS"],
                    "fixed": plans[family]["fixed"],
                    "configurations": {},
                    "seconds": {},
                    "parameters": written,
                },
            )

    def planned(family, seed):
        """Fa, desa i torna el pla d'una seed d'una família, o `None` si no pot.

        No pot mentre la cerca de la seed que la família llegeix no és acabada, ni si en
        acabar no té cap trial que puntuï.
        """
        made = None
        if kept[family]:
            made = searching.found(plans[family]["searched_as"], seed)
            if made is None or not made["ranked"]:
                return None
        configured = configurations(made, kept[family])
        _planned(where / family, seed, configured)
        return configured

    for family in chosen:
        recorded = families.recorded(where / family)
        for seed in values["SEEDS"]:
            if seed not in recorded:
                planned(family, seed)
    recorded = {family: families.recorded(where / family) for family in chosen}
    missing = {family: families.missing(where / family) for family in chosen}

    def units(family, seed):
        """Torna les unitats de cada tipus d'una seed d'una família, i les que ja té."""
        found = {}
        if family in searched:
            found["trial"] = searching.counts(family, seed)
        if family in chosen:
            configured = recorded[family].get(seed)
            if configured is None:
                # Abans del pla, les que en guardarà.
                found["fit"] = (min(kept[family], values["TRIALS"]), 0)
            else:
                absent = missing[family].get(seed, [])
                found["fit"] = (len(configured), len(configured) - len(absent))
        return found

    block = progress.Block(
        f"Passada {where.name}{', continuada' if continued else ''}",
        {
            family: {seed: units(family, seed) for seed in values["SEEDS"]}
            for family in families.EVERY
            if family in chosen or family in searched
        },
    )
    for name in searched:
        for seed in values["SEEDS"]:
            why = searching.unscored(name, seed)
            if why is not None:
                block.stop(name, seed, why)
    # Cada ajust planificat que encara no té el seu fitxer, en l'ordre de les famílies,
    # de les seeds i dels llocs; la d'una seed acabada de cercar s'hi afegeix al final.
    queue = [
        (family, seed, configuration)
        for family in chosen
        for seed, absent in missing[family].items()
        for configuration in absent
    ]

    def tasks():
        # Un generador: el pool en llegeix tasques quan té processos lliures. Primer els
        # ajustos, després els trials, i `None` mentre els trials que falten esperen
        # els que encara no han tornat.
        while True:
            while queue and block.stopped(*queue[0][:2]):
                queue.pop(0)
            if queue:
                family, seed, configuration = queue.pop(0)
                yield (
                    (family, seed, "fit", configuration["rank"]),
                    _train,
                    (
                        TRAINERS[family],
                        family,
                        where / family,
                        seed,
                        configuration,
                        values["EXPLAINED_DAYS"],
                    ),
                )
                continue
            task = searching.next(block.stopped)
            if task is None and not searching.in_flight():
                return
            yield task

    def returned(label, result):
        family, seed, kind, rank = label
        if kind == "fit":
            _timed(where / family, seed, rank, result)
            return
        if searching.returned(label, result) is None:
            return
        why = searching.unscored(family, seed)
        if why is not None:
            block.stop(family, seed, why)
            return
        # La cerca de la seed és acabada: cada família triada que la llegeix en rep el
        # pla, si encara no el té, i els seus ajustos es fan abans que cap trial.
        for other in chosen:
            if (
                kept[other]
                and plans[other]["searched_as"] == family
                and seed not in families.recorded(where / other)
            ):
                configured = planned(other, seed)
                block.plan(other, seed, len(configured))
                queue.extend((other, seed, one) for one in configured)

    def failed(label):
        if label[2] == "trial":
            searching.failed(label)

    pool.done(
        tasks(),
        values["WORKERS"],
        values["THREADS"],
        where / paths.WORKERS_LOG,
        block,
        returned,
        failed,
    )
    if block.failed:
        counted = progress.counted(block.failed, families.EVERY)
        raise Refusal(f"la passada {where.name} no és acabada: {counted}")
    figures.draw(where)
