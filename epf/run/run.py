"""La comanda `epf run`, la passada del protocol de Trebbien et al. (2023).

Cerca i ajusta amb un sol pool: una seed d'una família s'ajusta tan bon punt la seva
cerca té tots els trials, mentre les altres seeds encara es cerquen.
"""

import datetime
import importlib
import json
import shutil
import sys
import time

from epf import Refusal, parameters, paths, terminal
from epf.contract import matrix, record, searches
from epf.contract.searches import configurations
from epf.contract.split import split, validation_parts
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, THREADS, WORKERS
from epf.run import families, figures, pool, progress, search

# Els valors que una passada llegeix de `parameters.chosen`, els únics que una variant
# d'assaig pot canviar; la resta s'importen on es fan servir.
READ = ("SEEDS", "TRIALS", "BEST", "EXPLAINED_DAYS")

# El que les taules i les figures de la passada llegeixen d'una família, que el seu
# registre copia de `FAMILIES`: si se'n fa el test de Ljung-Box dels errors i, de la que
# en té, la columna del comptador de dies.
COPIED = ("ljung_box_tested", "counter")


def _plans(chosen, values):
    """Torna com es planifica cada família triada, i les famílies que es cerquen.

    De cada família, la seva entrada de `FAMILIES` i quantes configuracions en guarda
    de cada seed, amb `values`, els valors de `parameters.chosen`. Les que es cerquen
    són les cerques que llegeix cada família que en guarda alguna, en l'ordre de les
    cerques: la sonda de datació fa cercar l'arbre.
    """
    plans = {
        family: importlib.import_module(families.MODULES[family]["train"]).FAMILIES[
            family
        ]
        for family in chosen
    }
    kept = {
        family: values["BEST"] if plan["kept"] == "BEST" else plan["kept"]
        for family, plan in plans.items()
    }
    searched = [
        name
        for name in families.EVERY
        if families.MODULES[name]["search"]
        and any(
            kept[family] and plans[family]["searched_as"] == name for family in chosen
        )
    ]
    return plans, kept, searched


def _otherwise(where, smoke, fingerprints):
    """Torna com es va començar una passada que no es pot continuar, o `None`.

    Ho torna en dues parts, com es va començar i què faria falta per continuar-la. No
    es pot continuar si la van començar un altre codi, un altre dataset o unes altres
    biblioteques, perquè les seves famílies no es podrien comparar, ni una d'assaig
    començada amb una altra variant que la d'ara, `smoke`, que hi barrejaria els valors
    de dues. Un registre sense una de les empremtes d'ara és d'un altre codi.
    """
    differ, begun = set(), set()
    for path in sorted(where.glob(f"*/{paths.FAMILY_RECORD}")):
        made = json.loads(path.read_text(encoding="utf-8"))
        differ |= {
            name for name, value in fingerprints.items() if made.get(name) != value
        }
        begun.add(made.get("smoke"))
    if differ:
        then = [record.DIFFERENT[name] for name in record.DIFFERENT if name in differ]
        return (
            f"es va començar amb {' i '.join(other for other, _ in then)}",
            f"continuar amb {' i '.join(that for _, that in then)}",
        )
    variants = sorted(map(str, begun - {smoke}))
    if variants:
        flags = {name: flag for flag, name in parameters.variants().items()}
        flag = flags.get(variants[0], variants[0])
        return f'es va començar amb "{flag}"', f'continuar amb "uv run epf run {flag}"'
    return None


def _counted(count, one, many):
    """Torna `1 trial` o `3 trials`, de `one` i `many`, el singular i el plural."""
    return f"{count} {one if count == 1 else many}"


def _on_disk(smoke, searched, seeds, matrices, fingerprints):
    """Torna el que una passada nova substituiria: de cada cosa, els camins i una línia.

    És tot el que hi ha a la carpeta de les passades del tipus, i cada study que la
    passada obriria, de les famílies de `searched` a les seeds de `seeds`, que es va fer
    d'una altra manera, amb el seu registre. Llegeix els studies que hi són i no en crea
    cap. `matrices` i `fingerprints` són les de la passada.
    """
    found = []
    named = paths.named_runs(smoke)
    runs = paths.runs(smoke)
    for path in sorted(runs.iterdir()) if runs.exists() else []:
        shown = paths.relative(path) + ("/" if path.is_dir() else "")
        if path not in named:
            count = ""
            if path.is_dir():
                files = sum(one.is_file() for one in path.rglob("*"))
                count = _counted(files, "fitxer", "fitxers") + ", "
            found.append(([path], f"{shown}: {count}no és cap passada"))
            continue
        recorded = len(list(path.glob(f"*/{paths.FAMILY_RECORD}")))
        models = len(list(path.glob(f"*/{paths.MODEL.format(seed='*', rank='*')}")))
        otherwise = _otherwise(path, smoke, fingerprints)
        if families.finished(path):
            state = "és acabada"
        elif otherwise is not None:
            state = otherwise[0]
        else:
            state = "no és acabada"
        found.append(
            (
                [path],
                f"{shown}: {_counted(recorded, 'família', 'famílies')}, "
                f"{_counted(models, 'model ajustat', 'models ajustats')}, {state}",
            )
        )
    for family in searched:
        module = search.searcher(family)
        X, _ = getattr(matrices, module.MATRIX)
        folder = searches.folder(smoke, family)
        for seed in seeds:
            store = folder / searches.STUDY.format(seed=seed)
            study = searches.stored(store, family, seed)
            if study is None:
                continue
            why = searches.made_otherwise(
                study, X.columns, seed, module, fingerprints["dataset"]
            )
            if why is None:
                continue
            kept = folder / searches.RECORD.format(seed=seed)
            found.append(
                (
                    [store] + ([kept] if kept.exists() else []),
                    f"{paths.relative(store)}: "
                    f"{_counted(len(study.trials), 'trial', 'trials')}, "
                    f"es va fer {why}",
                )
            )
    return found


def _replaced(found, smoke):
    """Esborra cada camí de `found`, un cop vist que tots són del tipus `smoke`.

    Abans d'esborrar-ne cap, falla si algun no hi és o no és a la carpeta de les
    passades o de les cerques del tipus: no esborra mai res de fora.
    """
    roots = (paths.runs(smoke), paths.searches(smoke))
    every = [path for held, _ in found for path in held]
    for path in every:
        if not path.exists() or not any(path.is_relative_to(root) for root in roots):
            raise ValueError(
                f"{paths.relative(path)} ja no hi és, o no és a la carpeta de les "
                "passades ni de les cerques del tipus"
            )
    for path in every:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _where(smoke, searched, seeds, matrices, fingerprints):
    """Torna on es fa la passada: una de nova d'aquest minut, o la del tipus continuada.

    Sense res a la carpeta de les passades del tipus ni cap study fet d'una altra
    manera, és una de nova. Si no, en mostra cada cosa i pregunta si se substitueix:
    `y` ho esborra i en fa una de nova; `n` continua la passada més nova de la forma
    del tipus si no és acabada i es pot continuar, i si no, refusa. Refusa també sense
    terminal o sense resposta, i llavors no esborra res.
    """
    found = _on_disk(smoke, searched, seeds, matrices, fingerprints)
    if found:
        for _, line in found:
            print(line, file=sys.stderr)
        if not terminal.interactive():
            raise Refusal(
                "sense terminal, no es pot preguntar si se substitueix el que hi ha "
                "en disc"
            )
        answer = terminal.asked("Substituir el que hi ha en disc? [y/n] ")
        if answer is None:
            raise Refusal("sense resposta, es conserva el que hi ha en disc")
        if not answer:
            return _continued(smoke, fingerprints)
        _replaced(found, smoke)
    return paths.next_run(smoke, datetime.datetime.now())


def _continued(smoke, fingerprints):
    """Torna la passada més nova de la forma del tipus, si es pot continuar.

    Refusa que no n'hi hagi cap, que sigui acabada, o que no es pugui continuar.
    """
    named = paths.named_runs(smoke)
    if not named:
        raise Refusal(
            "no hi ha cap passada per continuar i es conserva el que hi ha en disc"
        )
    latest = named[-1]
    if families.finished(latest):
        raise Refusal(f"la passada {latest.name} és acabada i es conserva")
    otherwise = _otherwise(latest, smoke, fingerprints)
    if otherwise is not None:
        raise Refusal(f"la passada {latest.name} {otherwise[0]}; {otherwise[1]}")
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
    path = folder / paths.FAMILY_RECORD
    found = json.loads(path.read_text(encoding="utf-8"))
    found["configurations"][str(seed)] = configurations
    found["configurations"] = _ordered(found["configurations"])
    paths.write_record(path, found)


def _timed(folder, seed, rank, seconds):
    """Desa al registre d'una família els segons d'un ajust, al costat dels altres."""
    path = folder / paths.FAMILY_RECORD
    found = json.loads(path.read_text(encoding="utf-8"))
    ranks = found["seconds"].setdefault(str(seed), {})
    ranks[str(rank)] = round(seconds, 1)
    found["seconds"][str(seed)] = _ordered(ranks)
    found["seconds"] = _ordered(found["seconds"])
    paths.write_record(path, found)


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
        self.plans, self.kept, self.searched = _plans(chosen, values)
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
                paths.write_record(
                    where / family / paths.FAMILY_RECORD,
                    {
                        "family": family,
                        "smoke": smoke,
                        **fingerprints,
                        "fixed": self.plans[family]["fixed"],
                        **{
                            name: self.plans[family][name]
                            for name in COPIED
                            if name in self.plans[family]
                        },
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
            f"{'Continuant passada' if continued else 'Passada'} {where.name}",
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

    Continua la del seu tipus si no és acabada, i refusa una d'acabada. Cerca el que
    falta i, amb els mateixos processos, ajusta cada configuració planificada que encara
    no té el seu fitxer. Una seed d'una família que no es cerca rep el pla de seguida, i
    una d'una que es cerca, les millors configuracions de la seva cerca quan la té
    acabada; el pla queda al registre de la família i ja no canvia. Dibuixa les figures
    si cap seed no ha fallat. Refusa abans d'escriure la passada el que no pot fer; un
    study refusat deixa a la carpeta de les cerques els que s'han obert abans, nous i
    sense cap trial, que la passada següent reinicia en obrir-los. Acaba amb un refús si
    alguna seed ha fallat, i amb un `RuntimeError` si, sense cap fallada, a alguna
    família li falta el pla d'una seed o un fitxer.
    """
    values = parameters.chosen(arguments.smoke, READ)
    chosen = families.named(arguments.family)
    matrices = matrix.matrices(matrix.series())
    matrix.refuse_missing_days(matrices, DEVELOPMENT, BLACKOUT_WEEK)
    # La partició i les parts de validació, només per refusar-les abans d'escriure res.
    split(values["SEEDS"][0], DEVELOPMENT, BLACKOUT_WEEK)
    validation_parts(values["SEEDS"][0], DEVELOPMENT, BLACKOUT_WEEK)
    fingerprints = record.fingerprints()
    searched = _plans(chosen, values)[2]
    where = _where(arguments.smoke, searched, values["SEEDS"], matrices, fingerprints)
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
