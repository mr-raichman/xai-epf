"""La cerca de Trebbien et al. (2023), dins d'una passada."""

import importlib

import optuna

from epf.contract import matrix, record, searches
from epf.run import families


def searcher(family):
    """Torna el `search.py` del model d'una família que es cerca."""
    return importlib.import_module(families.MODULES[family]["search"])


def _trial(family, seed, drawn, fingerprints, set_name):
    """Ajusta els valors d'un trial d'una família en una seed, una tasca de `pool`.

    Torna els números, la puntuació i on s'ha aturat l'ajust. Refusa un codi, unes
    matrius o unes biblioteques que no són els de `fingerprints`, les empremtes amb què
    es va començar o continuar la passada, les matrius les de l'information set
    `set_name`.
    """
    built = record.built_for(set_name, fingerprints)
    module = searcher(family)
    X, y = getattr(built, module.MATRIX)
    parts = matrix.partition(built, seed)
    return searches.tried(module, family, drawn, X, y, parts, seed)


class Search:
    """La cerca d'una passada: el study de cada família que es cerca, a cada seed.

    Porta els trials demanats que encara no han tornat i el registre de cada seed que ja
    té tots els seus trials.
    """

    def __init__(self, smoke, searched, seeds, trials, built, fingerprints, log):
        """Obre el study de cada família de `searched` a cada seed de `seeds`.

        Els trials de cada seed són els de la família, i com a molt `trials`, si no és
        `None`. `built` i `fingerprints` són les matrius i les empremtes de la passada,
        i `log`, el fitxer dels seus processos, on va cada trial que divergeix i cada
        cerca acabada. Els studies són els de l'information set de les matrius. Refusa
        un study fet d'una altra manera abans de desar cap registre, i desa el de cada
        seed que ja té tots els seus trials.
        """
        # Al pare, cada study que crea seria una línia d'Optuna a sobre del bloc.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.smoke, self.fingerprints = smoke, fingerprints
        self.set_name = built.information.name
        self.log = log
        self.trials, self.studies = {}, {}
        for family in searched:
            module = searcher(family)
            self.trials[family] = searches.trials(module, trials)
            X, _ = getattr(built, module.MATRIX)
            searches.folder(smoke, self.set_name, family).mkdir(
                parents=True, exist_ok=True
            )
            for seed in seeds:
                self.studies[family, seed] = searches.opened(
                    self.store(family, seed),
                    family,
                    module,
                    seed,
                    X.columns,
                    built,
                    fingerprints["matrices"],
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
        # Els trials d'una onada demanats a Optuna que encara no s'han donat. No compten
        # com a pendents: els d'una seed fallada no es donen mai, i la passada no
        # s'acabaria.
        self.queued = {key: [] for key in self.studies}
        self.records = {}
        for family, seed in self.studies:
            if self.done[family, seed] >= self.trials[family]:
                self._saved(family, seed)

    def ask(self, family, seed):
        """Torna la tasca del trial següent d'una família en una seed, o `None`.

        Una família que es cerca en onades, amb `TRIALS_PER_WAVE`, en demana cada una
        sencera quan l'anterior ha tornat sencera, i en dona els trials d'un en un.
        `None` si la seed ja té demanats tots els seus trials, o si la seva onada encara
        en té de pendents.
        """
        module, queued = searcher(family), self.queued[family, seed]
        if not queued:
            out = sum(label[:2] == (family, seed) for label in self.pending)
            left = self.trials[family] - self.done[family, seed] - out
            wave = getattr(module, "TRIALS_PER_WAVE", 1)
            if left <= 0 or (out and wave > 1):
                return None
            study = self.studies[family, seed]
            queued += [
                searches.asked(study, module, family) for _ in range(min(wave, left))
            ]
        optuna_trial, drawn = queued.pop(0)
        label = (family, seed, "trial", optuna_trial.number)
        self.pending[label] = optuna_trial
        return label, _trial, (family, seed, drawn, self.fingerprints, self.set_name)

    def store(self, family, seed):
        """Torna el fitxer del study d'una família en una seed."""
        folder = searches.folder(self.smoke, self.set_name, family)
        return folder / searches.STUDY.format(seed=seed)

    def in_flight(self):
        """Diu si algun trial demanat encara no ha tornat."""
        return bool(self.pending)

    def returned(self, label, result):
        """Desa al seu study el que torna un trial.

        `result` és el que torna, els números, la puntuació i on s'ha aturat l'ajust.
        Torna el registre de la seed si amb aquest trial ja els té tots, i ho diu al
        fitxer dels processos, i si no, `None`.
        """
        family, seed, _, number = label
        numbers, score, stop = result
        if searches.told(
            self.studies[family, seed],
            self.pending.pop(label),
            numbers,
            score,
            stop,
            self.fingerprints,
        ):
            self._logged(f"{family} seed {seed} trial {number} ha divergit: {score}")
        self.done[family, seed] += 1
        if self.done[family, seed] < self.trials[family]:
            return None
        trials = self.trials[family]
        self._logged(
            f"{family} seed {seed}: cerca acabada amb "
            + ("el seu trial" if trials == 1 else f"els seus {trials} trials")
        )
        return self._saved(family, seed)

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
        return self.trials[family], min(self.done[family, seed], self.trials[family])

    def _logged(self, line):
        """Afegeix una línia al fitxer dels processos de la passada."""
        with open(self.log, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _saved(self, family, seed):
        """Desa el registre d'una seed que ja té tots els seus trials, i el torna."""
        found = searches.record(self.studies[family, seed], family, seed, self.smoke)
        searches.save(found, self.smoke, self.set_name)
        self.records[family, seed] = found
        return found
