"""El que comparteixen les cerques de la part A: les files d'una prova i com es parteixen,
i, per a les dues xarxes, l'estimador, el pressupost i els selectors de blocs.

Una prova és un ajust, com al codi de Lago et al. (2021): amb els quatre anys anteriors
al primer dia de prova, aturat i puntuat amb la seva quarta part final en ordre de
calendari, partida per dies sencers a la taula per hora. La partició de la quarta part
és també la que fa servir la xarxa de l'article quan recalibra cada dia, aleshores
barrejant setmanes senceres. Cada família té la seva cerca a la seva carpeta, i
`epf search --family` tria quina.

Les dues xarxes es cerquen amb el mateix estimador, el de Parzen en arbre que Lago et al.
(2021) fan servir a través de hyperopt, aquí el d'Optuna tal com Optuna el dóna, amb les
vint proves a l'atzar de hyperopt abans de començar a guiar-se, i amb el pressupost de
l'article: la part A té un sol procediment de cerca. L'altra cosa en què hyperopt i
Optuna difereixen, quantes de les proves fetes compten com a bones, Optuna ja no la deixa
triar. Cada estudi es desa a cada prova, i tornar-lo a obrir el reprèn.
"""
import numpy as np
import optuna
import pandas as pd

from epf.matrix import calibration
from epf.parameters import DAYS_PER_WEEK, FOUR_YEARS, SEED, TEST_PERIODS

# El primer dia de prova. Les cerques s'ajusten amb els quatre anys d'abans i no en
# toquen cap.
FIRST_TEST_DAY = pd.Timestamp(min(first for first, _ in TEST_PERIODS.values()))

# La part dels dies que s'aparta per aturar l'entrenament i puntuar la prova: el
# `percentage_val` del codi de Lago et al. (2021).
VALIDATION_SHARE = 0.25

# Les proves a l'atzar abans que l'estimador comenci a guiar-se per les anteriors: les
# del `tpe.suggest` de hyperopt que va fer servir Lago et al. (2021), on Optuna en fa 10.
STARTUP = 20

# El pressupost de cada cerca guiada, les proves de Lago et al. (2021).
TRIALS = 1500


def window(X, y):
    """Les files de totes les proves: les dels quatre anys anteriors al primer dia de
    prova, una per dia a la matriu diària i 24 a la taula per hora."""
    return calibration(X, y, FIRST_TEST_DAY, FOUR_YEARS)


def split(count, generator=None):
    """Els dies d'entrenament i els de validació d'una finestra de `count` dies, com els
    parteix el codi de Lago et al. (2021).

    La validació és la `VALIDATION_SHARE` final dels dies. Amb `generator`, abans s'hi
    barregen setmanes senceres, de set dies a comptar del primer, perquè dos dies veïns
    no quedin un a cada banda: és el que fa el codi en recalibrar, i la xarxa s'entrena
    també amb els dies més recents. Sense, l'ordre és el del calendari, que és el de les
    cerques. El tall és a un nombre de dies i no de setmanes, de manera que una setmana
    pot quedar partida entre les dues bandes, com al codi.
    """
    order = np.arange(count)
    if generator is not None:
        starts = generator.permutation(order[::DAYS_PER_WEEK])
        order = np.concatenate([order[start:start + DAYS_PER_WEEK] for start in starts])
    validation = int(VALIDATION_SHARE * count)
    return order[:count - validation], order[count - validation:]


def parts(X):
    """Les files d'entrenament i les de validació d'una prova sobre la taula per hora, com
    a màscares, partides per dies sencers en ordre de calendari: les 24 hores d'un dia
    van totes a la mateixa banda."""
    days = X.index.unique()
    train, validate = split(len(days))
    return X.index.isin(days[train]), X.index.isin(days[validate])


def completed(searching):
    """Les proves acabades d'un estudi."""
    return [one for one in searching.trials
            if one.state == optuna.trial.TrialState.COMPLETE]


def through_grid(searching, objective, points, trials=None):
    """Els punts d'una graella de `points` que l'estudi encara no ha provat, o menys si
    `trials` ho diu.

    Una graella ja sencera no en fa cap: el mostrejador d'Optuna en tornaria a provar un
    a l'atzar.
    """
    remaining = points - len(completed(searching))
    searching.optimize(objective, n_trials=remaining if trials is None
                       else min(trials, remaining))


def guided(store, name):
    """L'estudi de la cerca `name`, desat a `store`, amb un estimador que no repeteix el
    que l'estudi ja ha provat ni el que proven les altres cerques.

    La llavor de l'estimador surt del nom de l'estudi i de les proves que ja porta. Del
    nom, perquè dues cerques amb la mateixa llavor proposarien les mateixes proves a
    l'atzar: quatre membres de la xarxa serien la mateixa cerca quatre vegades. De les
    proves, perquè cada crida en fa un de nou, i amb la mateixa llavor tornaria a
    proposar les del començament: una cerca represa repetiria les primeres i no en faria
    de noves. El nom entra a la llavor pels seus bytes, que no canvien d'un procés a un
    altre com el `hash` d'un text.
    """
    found = optuna.create_study(study_name=name, direction="minimize", load_if_exists=True,
                                storage=f"sqlite:///{store}")
    seed = np.random.SeedSequence([SEED, *name.encode(), len(found.trials)])
    found.sampler = optuna.samplers.TPESampler(n_startup_trials=STARTUP,
                                               seed=int(seed.generate_state(1)[0]))
    return found


def switched_on(trial, names):
    """Els blocs que una prova engega, d'entre `names`, demanats un per un a Optuna.

    Una prova que no en deixa cap es poda, perquè donaria a la xarxa una entrada buida.
    """
    chosen = [name for name in names
              if trial.suggest_categorical(f"use_{name}", (False, True))]
    if not chosen:
        raise optuna.TrialPruned("cap bloc d'entrada engegat")
    return chosen
