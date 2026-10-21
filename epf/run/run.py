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
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, THREADS, WORKERS
from epf.run import families, figures, pool, progress, search

# Els valors que una passada llegeix de `parameters.chosen`, els únics que una variant
# d'assaig pot canviar; la resta s'importen on es fan servir.
READ = ("SEEDS", "TRIALS", "BEST", "EXPLAINED_DAYS")


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
        then = [record.DIFFERENT[name] for name in record.DIFFERENT if name in differ]
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


def _train(module, family, folder, seed, configuration, explained_days, fingerprints):
    """Ajusta una configuració d'una família en una seed, una tasca de `pool`.

    Crida el `train` del mòdul del model, `module`. Torna quants segons ha trigat, sense
    importar-lo. Refusa un codi, un dataset o unes biblioteques que no són els de
    `fingerprints`, les empremtes de la passada.
    """
    record.refuse_changed(fingerprints)
    train = importlib.import_module(module).train
    # Un procés l'importa a la seva primera tasca del model, i només en un assaig això
    # passa sovint en un ajust.
    started = time.time()
    train(family, folder, seed, configuration, explained_days)
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


class Run:
    """Una passada: els plans de les famílies triades, la seva cerca i la cua d'ajustos.

    Dona al pool els ajustos planificats abans que cap trial, i el pla d'una seed d'una
    família que es cerca quan la seva cerca té tots els trials.
    """

    def __init__(self, where, smoke, values, chosen, fingerprints, matrices):
        """Obre la cerca, escriu la passada a `where` i fa cada pla que ja pot.

        `smoke` és la variant d'assaig, `values` els valors de `parameters.chosen`,
        `chosen` les famílies triades, i `fingerprints` i `matrices` les de la passada.
        La cerca refusa abans que s'escrigui res de la passada. Escriu el registre de
        cada família que encara no en té, i el bloc es mostra des d'aquí.
        """
        self.where, self.values, self.chosen = where, values, chosen
        self.fingerprints = fingerprints
        continued = where.exists()
        self.plans = {}
        for family in chosen:
            module = importlib.import_module(families.MODULES[family]["train"])
            self.plans[family] = module.FAMILIES[family]
        # Les configuracions que una família guarda de cada seed.
        self.kept = {
            family: values["BEST"] if plan["kept"] == "BEST" else plan["kept"]
            for family, plan in self.plans.items()
        }
        # La cerca que llegeix cada família que guarda alguna configuració, en l'ordre
        # de les cerques: la sonda de datació fa cercar l'arbre.
        self.searched = [
            name
            for name in families.EVERY
            if families.MODULES[name]["search"]
            and any(
                self.kept[family] and self.plans[family]["searched_as"] == name
                for family in chosen
            )
        ]
        self.searching = search.Search(
            smoke,
            self.searched,
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
        written = {
            name: value for name, value in values.items() if name not in left_out
        }
        for family in chosen:
            if families.recorded(where / family) is None:
                record.write_record(
                    where / family / paths.SEED_RECORD,
                    {
                        "family": family,
                        "smoke": smoke,
                        **fingerprints,
                        "fixed": self.plans[family]["fixed"],
                        "configurations": {},
                        "seconds": {},
                        "settings": written,
                    },
                )
        for family in chosen:
            recorded = families.recorded(where / family)
            for seed in values["SEEDS"]:
                if seed not in recorded:
                    self.planned(family, seed)
        recorded = {family: families.recorded(where / family) for family in chosen}
        missing = {family: families.missing(where / family) for family in chosen}
        self.block = progress.Block(
            f"Passada {where.name}{', continuada' if continued else ''}",
            {
                family: {
                    seed: self._units(family, seed, recorded, missing)
                    for seed in values["SEEDS"]
                }
                for family in families.EVERY
                if family in chosen or family in self.searched
            },
        )
        for name in self.searched:
            for seed in values["SEEDS"]:
                why = self.searching.unscored(name, seed)
                if why is not None:
                    self.block.stop(name, seed, why)
        # Cada ajust planificat que encara no té el seu fitxer, en l'ordre de les
        # famílies, de les seeds i dels llocs; la d'una seed acabada de cercar s'hi
        # afegeix al final.
        self.queue = [
            (family, seed, configuration)
            for family in chosen
            for seed, absent in missing[family].items()
            for configuration in absent
        ]

    def planned(self, family, seed):
        """Fa, desa i torna el pla d'una seed d'una família, o `None` si no pot.

        No pot mentre la cerca de la seed que la família llegeix no és acabada, ni si en
        acabar no té cap trial que puntuï.
        """
        made = None
        if self.kept[family]:
            made = self.searching.found(self.plans[family]["searched_as"], seed)
            if made is None or not made["ranked"]:
                return None
        configured = configurations(made, self.kept[family])
        _planned(self.where / family, seed, configured)
        return configured

    def _units(self, family, seed, recorded, missing):
        """Torna les unitats de cada tipus d'una seed d'una família, i les que ja té.

        `recorded` i `missing` són el pla de cada família i els seus ajustos que encara
        no tenen el seu fitxer.
        """
        found = {}
        if family in self.searched:
            found["trial"] = self.searching.counts(family, seed)
        if family in self.chosen:
            configured = recorded[family].get(seed)
            if configured is None:
                # Abans del pla, les que en guardarà com a molt.
                found["fit"] = (min(self.kept[family], self.values["TRIALS"]), 0)
            else:
                absent = missing[family].get(seed, [])
                found["fit"] = (len(configured), len(configured) - len(absent))
        return found

    def tasks(self):
        """Dona les tasques de la passada, un generador que el pool llegeix.

        Primer els ajustos, després els trials, i `None` mentre els trials que falten
        esperen els que encara no han tornat.
        """
        while True:
            while self.queue and self.block.stopped(*self.queue[0][:2]):
                self.queue.pop(0)
            if self.queue:
                family, seed, configuration = self.queue.pop(0)
                yield (
                    (family, seed, "fit", configuration["rank"]),
                    _train,
                    (
                        families.MODULES[family]["train"],
                        family,
                        self.where / family,
                        seed,
                        configuration,
                        self.values["EXPLAINED_DAYS"],
                        self.fingerprints,
                    ),
                )
                continue
            task = self.searching.next(self.block.stopped)
            if task is None and not self.searching.in_flight():
                return
            yield task

    def returned(self, label, result):
        """Desa el que torna una tasca: d'un ajust, els segons; d'un trial, el resultat.

        Quan amb el trial la cerca de la seed ja és acabada, planifica cada família
        triada que la llegeix, i en posa els ajustos a la cua.
        """
        family, seed, kind, rank = label
        if kind == "fit":
            _timed(self.where / family, seed, rank, result)
            return
        if self.searching.returned(label, result) is None:
            return
        why = self.searching.unscored(family, seed)
        if why is not None:
            self.block.stop(family, seed, why)
            return
        # Els ajustos de la seed es fan abans que cap trial.
        for other in self.chosen:
            if (
                self.kept[other]
                and self.plans[other]["searched_as"] == family
                and seed not in families.recorded(self.where / other)
            ):
                configured = self.planned(other, seed)
                self.block.plan(other, seed, len(configured))
                self.queue.extend((other, seed, one) for one in configured)

    def failed(self, label):
        """Dona per fallat a la cerca un trial que no ha tornat cap resultat."""
        if label[2] == "trial":
            self.searching.failed(label)

    def undone(self):
        """Torna les famílies triades que no són completes, en ordre."""
        return [
            family
            for family in self.chosen
            if not families.complete(self.where / family)
        ]


def main(arguments):
    """Fa una passada, la comanda `epf run`.

    Continua l'última del seu tipus si no és acabada. Cerca el que falta i, amb els
    mateixos processos, ajusta cada configuració planificada que encara no té el seu
    fitxer. Una seed d'una família que no es cerca rep el pla de seguida, i una d'una
    que es cerca, les millors configuracions de la seva cerca quan la té acabada; el
    pla queda al registre de la família i ja no canvia. Dibuixa les figures si cap seed
    no ha fallat. Refusa abans d'escriure la passada el que no pot fer; un study
    refusat deixa a `output/searches/` els que s'han obert abans, nous i sense cap
    trial, que la passada següent reinicia en obrir-los. Acaba amb un refús si alguna
    seed ha fallat, i amb un `RuntimeError` si, sense cap fallada, a alguna família li
    falta el pla d'una seed o un fitxer.
    """
    values = parameters.chosen(arguments.smoke, READ)
    chosen = families.named(arguments.family)
    matrices = matrix.matrices(matrix.series())
    matrix.refuse_missing_days(matrices, DEVELOPMENT, BLACKOUT_WEEK)
    # La partició i les parts de validació, només per refusar-les abans d'escriure res.
    split(values["SEEDS"][0], DEVELOPMENT, BLACKOUT_WEEK)
    validation_parts(values["SEEDS"][0], DEVELOPMENT, BLACKOUT_WEEK)
    fingerprints = record.fingerprints()
    where = _where(arguments.smoke, fingerprints)
    run = Run(where, arguments.smoke, values, chosen, fingerprints, matrices)
    pool.done(
        run.tasks(),
        WORKERS,
        THREADS,
        where / paths.WORKERS_LOG,
        run.block,
        run.returned,
        run.failed,
    )
    if run.block.failed:
        counted = progress.counted(run.block.failed, families.EVERY)
        raise Refusal(f"la passada {where.name} no és acabada: {counted}")
    undone = run.undone()
    if undone:
        # Sense cap seed fallada, només un error del codi hi deixa res per fer, i en
        # mostra el traceback.
        raise RuntimeError(
            f"la passada {where.name} no és acabada i cap seed no ha fallat: "
            + (
                f"1 família no és completa, {undone[0]}"
                if len(undone) == 1
                else f"{len(undone)} famílies no són completes, la primera {undone[0]}"
            )
        )
    figures.draw(where)
