"""La cerca de Trebbien et al. (2023), dins d'una passada."""

import importlib

import optuna

from epf.contract import matrix, record, searches
from epf.contract.split import split
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT
from epf.run import families


def searcher(family):
    """Torna el `search.py` del model d'una família que es cerca."""
    return importlib.import_module(families.MODULES[family]["search"])


def _trial(family, seed, drawn, fingerprints):
    """Ajusta els valors d'un trial d'una família en una seed, una tasca de `pool`.

    Torna els números i la puntuació. Refusa un codi, un dataset o unes biblioteques
    que no són els de `fingerprints`, les empremtes amb què es va començar o continuar
    la passada.
    """
    record.refuse_changed(fingerprints)
    module = searcher(family)
    X, y = getattr(matrix.matrices(matrix.series()), module.MATRIX)
    return searches.tried(
        module, family, drawn, X, y, split(seed, DEVELOPMENT, BLACKOUT_WEEK), seed
    )


class Search:
    """La cerca d'una passada: el study de cada família que es cerca, a cada seed.

    Porta els trials demanats que encara no han tornat i el registre de cada seed que ja
    té tots els seus trials.
    """

    def __init__(self, smoke, searched, seeds, trials, matrices, fingerprints, log):
        """Obre el study de cada família de `searched` a cada seed de `seeds`.

        `trials` són els trials de cada seed, `matrices` i `fingerprints` les matrius i
        les empremtes de la passada, i `log`, el fitxer dels seus processos, on va cada
        trial que divergeix. Refusa un study fet d'una altra manera abans de desar cap
        registre, i desa el de cada seed que ja té tots els seus trials.
        """
        # Al pare, cada study que crea seria una línia d'Optuna a sobre del bloc.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.smoke, self.trials, self.fingerprints = smoke, trials, fingerprints
        self.log = log
        self.studies = {}
        for family in searched:
            module = searcher(family)
            X, _ = getattr(matrices, module.MATRIX)
            folder = searches.folder(smoke, family)
            folder.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                self.studies[family, seed] = searches.opened(
                    folder / searches.STUDY.format(seed=seed),
                    family,
                    module,
                    seed,
                    X.columns,
                    fingerprints["dataset"],
                )
        # Els trials que té cada seed, i el trial demanat per cada tasca que encara no
        # ha tornat. Un trial que divergeix hi compta, com a la cerca de Marcjasz et al.
        # (2023), on l'`optimize` d'Optuna el compta entre els seus, i no se'n demana
        # cap altre en lloc seu.
        self.done = {
            key: len(searches.completed(study)) + len(searches.diverged(study))
            for key, study in self.studies.items()
        }
        self.pending = {}
        self.records = {}
        for family, seed in self.studies:
            if self.done[family, seed] >= trials:
                self._saved(family, seed)

    def ask(self, family, seed):
        """Torna la tasca del trial següent d'una família en una seed, o `None`.

        `None` si la seed ja té demanats tots els seus trials.
        """
        asked = self.done[family, seed] + sum(
            label[:2] == (family, seed) for label in self.pending
        )
        if asked >= self.trials:
            return None
        study = self.studies[family, seed]
        optuna_trial, drawn = searches.asked(study, searcher(family), family)
        label = (family, seed, "trial", optuna_trial.number)
        self.pending[label] = optuna_trial
        return label, _trial, (family, seed, drawn, self.fingerprints)

    def in_flight(self):
        """Diu si algun trial demanat encara no ha tornat."""
        return bool(self.pending)

    def returned(self, label, result):
        """Desa al seu study el que torna un trial, els números i la puntuació.

        `result` és el que torna. Torna el registre de la seed si amb aquest trial ja
        els té tots, i si no, `None`.
        """
        family, seed, _, number = label
        numbers, score = result
        if searches.told(
            self.studies[family, seed],
            self.pending.pop(label),
            numbers,
            score,
            self.fingerprints,
        ):
            with open(self.log, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{family} seed {seed} trial {number} ha divergit: {score}\n"
                )
        self.done[family, seed] += 1
        if self.done[family, seed] == self.trials:
            return self._saved(family, seed)
        return None

    def failed(self, label):
        """Dona per fallat al seu study un trial que no ha tornat cap resultat."""
        family, seed, _, _ = label
        searches.abandoned(self.studies[family, seed], self.pending.pop(label))

    def found(self, family, seed):
        """Torna el registre d'una seed que ja té tots els seus trials, o `None`."""
        return self.records.get((family, seed))

    def unscored(self, family, seed):
        """Torna per què una seed amb tots els seus trials no en té cap que puntuï.

        Torna `None` si en té algun, o si encara no els té tots.
        """
        found = self.records.get((family, seed))
        if found is None or found["ranked"]:
            return None
        count = len(searches.diverged(self.studies[family, seed]))
        if count == 1:
            return "el seu trial ha divergit"
        return f"els seus {count} trials han divergit"

    def counts(self, family, seed):
        """Torna els trials d'una seed i els que ja té, `(trials, fets)`."""
        return self.trials, min(self.done[family, seed], self.trials)

    def _saved(self, family, seed):
        """Desa el registre d'una seed que ja té tots els seus trials, i el torna."""
        found = searches.record(self.studies[family, seed], family, seed, self.smoke)
        searches.save(found, self.smoke)
        self.records[family, seed] = found
        return found
