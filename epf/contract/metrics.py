"""Les puntuacions de tots els models, i les mètriques de la Part I que se'n fan."""

import math

import numpy as np
from scipy import stats

from epf import Refusal
from epf.contract.explanation import PRICE_UNITS, UNITLESS
from epf.parameters import DAYS_PER_WEEK, DEVELOPMENT, HOURS_PER_DAY

# Les mètriques principals de la Part I.
MAIN = ("r2", "mae", "rmae")

# Les unitats de cada mètrica, i de cada número que se'n fa.
UNITS = {
    "r2": UNITLESS,
    "mae": PRICE_UNITS,
    "rmae": UNITLESS,
    "rmse": PRICE_UNITS,
    "mse": "(EUR/MWh)²",
    "mdae": PRICE_UNITS,
    "mean_error": PRICE_UNITS,
    "wape": UNITLESS,
    "corr": UNITLESS,
    "mase": UNITLESS,
    "rmsse": UNITLESS,
    "wmae": UNITLESS,
    "deviation_from_best": UNITLESS,
    "deviation_from_best_mean": UNITLESS,
    # El logaritme del determinant d'una matriu de 24 per 24 en (EUR/MWh)².
    "cov_e": "log (EUR/MWh)⁴⁸",
    "corr_f": UNITLESS,
    "mhd": "hores",
    "mpd": PRICE_UNITS,
    "rmae_year": UNITLESS,
    "rmae_decile": UNITLESS,
    "price_decile_top": PRICE_UNITS,
    "ljung_box_q": UNITLESS,
    "ljung_box_p": UNITLESS,
    "ljung_box_days": "dies",
    "autocorrelation": UNITLESS,
    "dm_day": UNITLESS,
    "dm_hour": UNITLESS,
    "error_correlation": UNITLESS,
    "t_test": UNITLESS,
    "t_statistic": UNITLESS,
    "t_left": UNITLESS,
    "t_rope": UNITLESS,
    "t_right": UNITLESS,
}

# Retards del test de Ljung-Box, dues vegades el període setmanal, com aconsellen
# Hyndman i Athanasopoulos (2021) per a una sèrie estacional.
LAGS = 2 * DAYS_PER_WEEK

# Els límits de les deu parts del preu.
DECILES = np.linspace(0, 1, 11)

# La meitat de la regió d'equivalència de la t bayesiana, en part del MAE mitjà del
# parell sobre les seeds: l'1 % que Benavoli et al. (2017) prenen en l'exactitud.
ROPE = 0.01


def _fail_on_absent(actual):
    """Falla si el preu real té cap valor absent, i diu quants.

    Un NaN faria tornar NaN la mesura sencera, o la faria saltar l'hora i informar-la
    com si fos neta, segons com es mitjanés.
    """
    absent = int(np.isnan(np.asarray(actual)).sum())
    # Només un error del codi hi arriba, i n'hi ha prou amb el recompte: el dataset no
    # té cap preu absent dins del seu tram, i `matrix.series` retalla cada sèrie al seu.
    if absent:
        raise ValueError(
            "el preu real té 1 valor absent"
            if absent == 1
            else f"el preu real té {absent} valors absents"
        )


def mae(predicted, actual):
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def scored(predicted, actual):
    """Torna l'R² i el MAE de totes les files juntes.

    Falla amb un preu real absent, i refusa una predicció que no és un número finit: un
    sol NaN tornaria NaN totes dues puntuacions sense dir-ho.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    _fail_on_absent(actual)
    wrong = int((~np.isfinite(predicted)).sum())
    if wrong:
        raise Refusal(
            "la predicció té 1 valor no finit"
            if wrong == 1
            else f"la predicció té {wrong} valors no finits"
        )
    error = predicted - actual
    spread = actual - np.mean(actual)
    return {
        "r2": float(1 - np.sum(error**2) / np.sum(spread**2)),
        "mae": mae(predicted, actual),
    }


def sample_spread(values):
    """Torna la desviació estàndard mostral d'uns valors, la que divideix per n - 1.

    D'un sol valor no n'hi ha, i torna NaN sense cap avís.
    """
    if len(values) < 2:
        return math.nan
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def part_i_main(scores, denominator, complete):
    """Torna les files de les mètriques principals de la Part I, en l'ordre del fitxer.

    `scores` dona l'R² i el MAE de cada model ajustat, `{"r2": ..., "mae": ...}`, per
    família, en l'ordre de les famílies, i per `(seed, configuració)`; `denominator`,
    la família que divideix l'rMAE, o `None` si la passada no en té; i `complete`, les
    famílies completes. L'rMAE d'un model és el seu MAE sobre el de la millor
    configuració de `denominator` a la mateixa seed, i sense aquest no en té. Cada
    família completa es resumeix per la mitjana i la desviació mostral de cada mètrica
    sobre la millor configuració de cada seed, i una mètrica a la qual falta alguna
    seed no es resumeix. Cada fila és un diccionari amb les claus que s'hi apliquen.
    """
    base = scores.get(denominator, {})
    found = {name: {} for name in MAIN}
    for family, fits in scores.items():
        for name in found:
            found[name][family] = {}
        for (seed, rank), scored in fits.items():
            found["r2"][family][seed, rank] = scored["r2"]
            found["mae"][family][seed, rank] = scored["mae"]
            if (seed, 0) in base:
                found["rmae"][family][seed, rank] = scored["mae"] / base[seed, 0]["mae"]
    rows = []
    for name, by_family in found.items():
        for family, values in by_family.items():
            rows += [
                {
                    "measure": name,
                    "family": family,
                    **({"other_family": denominator} if name == "rmae" else {}),
                    "seed": seed,
                    "rank": rank,
                    "value": value,
                    "unit": UNITS[name],
                }
                for (seed, rank), value in values.items()
            ]
    for name, by_family in found.items():
        for family in by_family:
            seeds = sorted({seed for seed, _ in scores[family]})
            if family not in complete or any(
                (seed, 0) not in by_family[family] for seed in seeds
            ):
                continue
            best = [by_family[family][seed, 0] for seed in seeds]
            for summary, value in (
                ("mean", float(np.mean(best))),
                ("std", sample_spread(best)),
            ):
                rows.append(
                    {
                        "measure": f"{name}_{summary}",
                        "family": family,
                        **({"other_family": denominator} if name == "rmae" else {}),
                        "rank": 0,
                        "value": value,
                        "unit": UNITS[name],
                    }
                )
    return rows


def daily_errors(seeds):
    """Torna l'error mitjà de cada dia, del primer al darrer dia de prova.

    `seeds` dona, en ordre de seed, els dies de prova i els errors `(dies, 24)` de la
    millor configuració d'una família. Cada dia pren l'error de la primera seed que el
    prova, i un dia que cap no prova val `NaN`.
    """
    found = {}
    for days, errors in seeds:
        for day, mean in zip(days, errors.mean(axis=1), strict=True):
            found.setdefault(day, mean)
    span = np.arange(min(found), max(found) + np.timedelta64(1, "D"))
    return np.array([found.get(day, np.nan) for day in span])


def ljung_box(series, lags=LAGS):
    """Torna la Q de Ljung-Box, el seu valor p i les autocorrelacions d'una sèrie.

    Com `Box.test` de R, amb els dies buits passats a `acf`; sense cap dia buit és
    l'estimador de sempre. El valor p és el d'una khi quadrat de `lags` graus, sense
    restar-ne cap, perquè els errors són de fora de la mostra.
    """
    x = np.asarray(series, dtype=float) - np.nanmean(series)
    present = int((~np.isnan(x)).sum())
    covariances = []
    for lag in range(lags + 1):
        products = x[lag:] * x[: len(x) - lag]
        pairs = ~np.isnan(products)
        # Entre els parells més el retard, com `acf` de R, que sense cap dia buit és la
        # llargada de la sèrie.
        covariances.append(products[pairs].sum() / (pairs.sum() + lag))
    correlations = np.clip(np.array(covariances[1:]) / covariances[0], -1, 1)
    statistic = (
        present
        * (present + 2)
        * np.sum(correlations**2 / (present - np.arange(1, lags + 1)))
    )
    return statistic, stats.chi2.sf(statistic, lags), correlations


def _spearman_by_day(actual, predicted):
    """Torna la correlació de Spearman de cada dia entre el preu i la predicció.

    Amb els rangs mitjans dels empats, com `scipy.stats.spearmanr`.
    """
    ranks = [stats.rankdata(values, axis=1) for values in (actual, predicted)]
    ranks = [one - one.mean(axis=1, keepdims=True) for one in ranks]
    return (ranks[0] * ranks[1]).sum(axis=1) / np.sqrt(
        (ranks[0] ** 2).sum(axis=1) * (ranks[1] ** 2).sum(axis=1)
    )


def _extremes(actual, predicted):
    """Torna, per dia, la distància en hores i en preu dels extrems previstos als reals.

    Per al mínim i el màxim, sumats: les hores, de la del previst a la més propera de
    les del real, que pot ser a més d'una hora; i el preu, el real a l'hora del previst
    menys el real a la seva, en valor absolut. Un previst empatat pren la primera hora.
    """
    hours = np.arange(actual.shape[1])
    rows = np.arange(len(actual))
    distance = price = 0
    for real, forecast in (
        (actual.min(axis=1), predicted.argmin(axis=1)),
        (actual.max(axis=1), predicted.argmax(axis=1)),
    ):
        tied = actual == real[:, None]
        distance = distance + np.where(
            tied, np.abs(hours - forecast[:, None]), np.inf
        ).min(axis=1)
        price = price + np.abs(real - actual[rows, forecast])
    return distance, price


def one_seed(days, actual, predicted):
    """Torna les mètriques secundàries d'una família en una seed, fetes d'ella sola.

    `days` són els dies de prova, i `actual` i `predicted` el preu i la predicció de
    cada un, `(dies, 24)`. L'error és el preu menys la predicció. El WMAE fa la mitjana
    de les setmanes de dimarts a dilluns des de `DEVELOPMENT[0]`. El Cov-e val NaN si la
    matriu dels errors no té determinant positiu.
    """
    error = actual - predicted
    absolute = np.abs(error)
    weeks = (days - np.datetime64(DEVELOPMENT[0], "D")).astype(int) // DAYS_PER_WEEK
    sign, logdet = np.linalg.slogdet(error.T @ error / len(error))
    distance, price = _extremes(actual, predicted)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mse": float(np.mean(error**2)),
        "mdae": float(np.median(absolute)),
        "mean_error": float(np.mean(error)),
        "wape": float(absolute.sum() / np.abs(actual).sum()),
        # Hewamalage, Ackermann i Bergmeir (2022).
        "corr": float(np.corrcoef(actual.ravel(), predicted.ravel())[0, 1]),
        # Uniejewski et al. (2016).
        "wmae": float(
            np.mean(
                [
                    absolute[weeks == week].mean() / actual[weeks == week].mean()
                    for week in np.unique(weeks)
                ]
            )
        ),
        # Les quatre de Maciejowska, Lipiecki i Uniejewski (2026).
        "cov_e": float(logdet) if sign > 0 else math.nan,
        "corr_f": float(np.mean(_spearman_by_day(actual, predicted))),
        "mhd": float(np.mean(distance)),
        "mpd": float(np.mean(price)),
    }


def _r2(actual, predicted):
    """Torna l'R² de les files, sobre la seva pròpia mitjana."""
    spread = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - np.sum((actual - predicted) ** 2) / spread)


def _mae(fit):
    """Torna el MAE de la millor configuració d'una seed, de `part_i_secondary`."""
    return float(np.abs(fit["actual"] - fit["predicted"]).mean())


def _years(days):
    """Torna l'any de cada dia."""
    return days.astype("datetime64[Y]").astype(int) + 1970


def _binned(error, base, bins, keys):
    """Torna l'rMAE de cada clau de `keys`: el MAE de les files de `bins` iguals.

    `error` i `base` són els errors de la família i de la còpia a les mateixes files.
    Una clau sense cap fila val NaN.
    """
    return {
        key: float(np.abs(error[bins == key]).mean() / np.abs(base[bins == key]).mean())
        if np.any(bins == key)
        else math.nan
        for key in keys
    }


def deciles(best):
    """Torna els límits de les deu parts del preu dels dies de prova de totes les seeds.

    `best` és com a `part_i_secondary`; cada dia i hora hi compta una vegada.
    """
    days, prices = [], []
    for fits in best.values():
        for fit in fits.values():
            days.append(fit["days"])
            prices.append(fit["actual"])
    _, first = np.unique(np.concatenate(days), return_index=True)
    return np.quantile(np.concatenate(prices)[first], DECILES)


def part_i_secondary(best, roles):
    """Torna les files de les mètriques secundàries de la Part I.

    `best` dona, per família, en l'ordre de les famílies, i per seed, la millor
    configuració: els dies de prova, `days`, el preu i la predicció de cada un,
    `actual` i `predicted`, `(dies, 24)`, i de la que en té, el MAE i el MSE de la
    còpia a l'entrenament, `training_mae` i `training_mse`. `roles` és el registre de
    cada família. L'rMAE per hora, per any i per decil del preu té, a més de cada seed,
    la mitjana de les seeds que tenen files a la casella, i una casella sense cap fila
    val NaN. Cada fila és un diccionari amb les claus que s'hi apliquen. Falla si una
    família i la còpia que la divideix no tenen els mateixos dies de prova a una seed.
    """
    main = next(
        (family for family in best if roles[family].get("denominator") == "main"), None
    )
    secondary = [
        family for family in best if roles[family].get("denominator") == "secondary"
    ]
    pool = [
        family
        for family in best
        if roles[family].get("interpreted") or roles[family].get("benchmark")
    ]
    edges = deciles(best)
    rows = []

    def add(measure, family, value, **keys):
        rows.append(
            {"measure": measure, "family": family, **keys, "value": value}
            | {"unit": UNITS[measure]}
        )

    for family, fits in best.items():
        years = sorted(
            {int(year) for fit in fits.values() for year in _years(fit["days"])}
        )
        # Cada tall de l'rMAE: la clau on va la casella i les caselles.
        cuts = {
            "rmae": ("hour", range(HOURS_PER_DAY)),
            "rmae_year": ("subject", years),
            "rmae_decile": ("subject", range(1, len(DECILES))),
        }
        found = {measure: [] for measure in cuts}
        deviations = []
        for seed, fit in fits.items():
            seeded = {"seed": seed, "rank": 0}
            error = fit["actual"] - fit["predicted"]
            for name, value in one_seed(
                fit["days"], fit["actual"], fit["predicted"]
            ).items():
                add(name, family, value, **seeded)
            for other in secondary:
                if seed in best[other]:
                    ratio = _mae(fit) / _mae(best[other][seed])
                    add("rmae", family, ratio, other_family=other, **seeded)
            present = [other for other in pool if seed in best[other]]
            if present:
                # Ziel i Weron (2018).
                deviations.append(
                    _mae(fit) / min(_mae(best[other][seed]) for other in present) - 1
                )
                add("deviation_from_best", family, deviations[-1], **seeded)
            for hour in range(HOURS_PER_DAY):
                hourly = fit["actual"][:, hour], fit["predicted"][:, hour]
                add(
                    "mae",
                    family,
                    float(np.abs(error[:, hour]).mean()),
                    hour=hour,
                    **seeded,
                )
                add("r2", family, _r2(*hourly), hour=hour, **seeded)
            if main is None or seed not in best[main]:
                continue
            copy = best[main][seed]
            if not np.array_equal(fit["days"], copy["days"]):
                raise ValueError(
                    f"{family} i {main} no tenen els mateixos dies de prova a la "
                    f"seed {seed}"
                )
            # Hyndman i Koehler (2006), i l'RMSSE de Hyndman i Athanasopoulos (2021).
            scaled = {
                "mase": _mae(fit) / copy["training_mae"],
                "rmsse": float(np.sqrt(np.mean(error**2) / copy["training_mse"])),
            }
            for name, value in scaled.items():
                add(name, family, value, other_family=main, **seeded)
            bins = {
                "rmae": np.broadcast_to(np.arange(HOURS_PER_DAY), error.shape),
                "rmae_year": np.broadcast_to(_years(fit["days"])[:, None], error.shape),
                "rmae_decile": np.clip(
                    np.searchsorted(edges, fit["actual"], side="right"),
                    1,
                    len(DECILES) - 1,
                ),
            }
            base = copy["actual"] - copy["predicted"]
            for measure, (key, keys) in cuts.items():
                found[measure].append(_binned(error, base, bins[measure], keys))
                for one, value in found[measure][-1].items():
                    shown = {key: one if key == "hour" else str(one)}
                    add(measure, family, value, other_family=main, **seeded, **shown)
        if deviations:
            add("deviation_from_best_mean", family, float(np.mean(deviations)), rank=0)
        for measure, (key, keys) in cuts.items():
            for one in keys if found[measure] else ():
                values = [
                    by_seed[one]
                    for by_seed in found[measure]
                    if not math.isnan(by_seed[one])
                ]
                mean = float(np.mean(values)) if values else math.nan
                shown = {key: one if key == "hour" else str(one)}
                add(measure, family, mean, other_family=main, rank=0, **shown)
        if roles[family].get("ljung_box_tested"):
            errors = {
                seed: (fit["days"], fit["actual"] - fit["predicted"])
                for seed, fit in fits.items()
            }
            # La sèrie cosida de totes les seeds, i la de cada una.
            series = {None: daily_errors(errors.values())} | {
                seed: daily_errors([one]) for seed, one in errors.items()
            }
            for seed, one in series.items():
                statistic, value, correlations = ljung_box(one)
                keys = {"rank": 0} | ({} if seed is None else {"seed": seed})
                add("ljung_box_q", family, float(statistic), **keys)
                add("ljung_box_p", family, float(value), **keys)
                add("ljung_box_days", family, int((~np.isnan(one)).sum()), **keys)
                if seed is None:
                    for lag, correlation in enumerate(correlations, start=1):
                        add(
                            "autocorrelation",
                            family,
                            float(correlation),
                            subject=str(lag),
                            **keys,
                        )
    for decile, top in enumerate(edges[1:], start=1):
        add("price_decile_top", None, float(top), subject=str(decile))
    return rows


def diebold_mariano(first, second):
    """Torna el valor p de Diebold-Mariano que `second` és més precís que `first`.

    `first` i `second` són els errors de les dues prediccions, `(dies, 24)`. Com el
    `DM` de Lago et al. (2021): la pèrdua absoluta, la variància de les diferències
    sense corregir i una sola cua. Torna el del test multivariant, sobre la mitjana de
    cada dia, i els de l'univariant a cada hora.
    """
    differences = np.abs(first) - np.abs(second)

    def tested(found):
        statistic = found.mean(axis=0) / np.sqrt(found.var(axis=0) / len(found))
        return 1 - stats.norm.cdf(statistic)

    return float(tested(differences.mean(axis=1))), tested(differences)


def corrected_t_test(differences, ratio, rope):
    """Torna la t corregida d'unes diferències de MAE, i la seva forma bayesiana.

    `differences` són les de cada seed, `ratio` les setmanes de prova sobre les
    d'entrenament i de validació, i `rope` la meitat de la regió d'equivalència. La
    variància és la mostral per 1/n més `ratio` (Nadeau i Bengio, 2003), i la posterior,
    una t de n - 1 graus centrada en la mitjana amb aquesta escala (Benavoli et al.,
    2017). Torna l'estadístic, el valor p de dues cues i la probabilitat de la posterior
    per sota de -`rope`, entre -`rope` i `rope` i per sobre de `rope`; unes diferències
    totes iguals no tenen dispersió, i ho tornen tot NaN.
    """
    n = len(differences)
    scale = math.sqrt((1 / n + ratio) * np.var(differences, ddof=1))
    if scale == 0:
        return dict.fromkeys(
            ("t_test", "t_statistic", "t_left", "t_rope", "t_right"), math.nan
        )
    mean = float(np.mean(differences))
    statistic = mean / scale
    posterior = stats.t(n - 1, loc=mean, scale=scale)
    return {
        "t_test": float(2 * stats.t.sf(abs(statistic), n - 1)),
        "t_statistic": statistic,
        "t_left": float(posterior.cdf(-rope)),
        "t_rope": float(posterior.cdf(rope) - posterior.cdf(-rope)),
        "t_right": float(posterior.sf(rope)),
    }


def _weeks(days):
    """Torna quantes setmanes toquen uns dies, les que reparteix la partició."""
    start = np.datetime64(DEVELOPMENT[0], "D")
    return len(np.unique((days - start).astype(int) // DAYS_PER_WEEK))


def part_i_tests(best):
    """Torna les files dels tests entre famílies, mètriques secundàries de la Part I.

    `best` és com a `part_i_secondary`, cada seed també amb els dies d'entrenament i de
    validació, `days_training` i `days_validation`. A cada seed i parell ordenat, els
    valors p de Diebold-Mariano que la segona família és més precisa que la primera,
    del dia i de cada hora; a cada seed i parell, una vegada en l'ordre de les
    famílies, la correlació de Pearson dels seus errors mitjans de cada dia; i, amb més
    d'una seed, la t corregida de cada parell sobre el MAE de la primera menys el de la
    segona, amb la regió d'equivalència de `ROPE`. Falla si les seeds no tenen les
    mateixes setmanes a cada part.
    """
    rows = []

    def add(measure, family, other, value, **keys):
        rows.append(
            {"measure": measure, "family": family, "other_family": other}
            | keys
            | {"rank": 0, "value": value, "unit": UNITS[measure]}
        )

    errors = {
        family: {seed: fit["actual"] - fit["predicted"] for seed, fit in fits.items()}
        for family, fits in best.items()
    }
    for family, fits in errors.items():
        for other, others in errors.items():
            if other == family:
                continue
            for seed in fits.keys() & others.keys():
                day, hours = diebold_mariano(fits[seed], others[seed])
                add("dm_day", family, other, day, seed=seed)
                for hour, value in enumerate(hours):
                    add("dm_hour", family, other, float(value), seed=seed, hour=hour)
    pairs = [
        (family, other)
        for number, family in enumerate(errors)
        for other in list(errors)[number + 1 :]
    ]
    for family, other in pairs:
        for seed in errors[family].keys() & errors[other].keys():
            daily = [errors[name][seed].mean(axis=1) for name in (family, other)]
            correlation = float(np.corrcoef(*daily)[0, 1])
            add("error_correlation", family, other, correlation, seed=seed)
    ratios = {
        _weeks(fit["days"])
        / (_weeks(fit["days_training"]) + _weeks(fit["days_validation"]))
        for fits in best.values()
        for fit in fits.values()
    }
    if len(ratios) > 1:
        raise ValueError(f"les seeds tenen {len(ratios)} proporcions de setmanes")
    (ratio,) = ratios
    for family, other in pairs:
        seeds = sorted(errors[family].keys() & errors[other].keys())
        if len(seeds) < 2:
            continue
        maes = np.array(
            [
                [np.abs(errors[name][seed]).mean() for seed in seeds]
                for name in (family, other)
            ]
        )
        found = corrected_t_test(maes[0] - maes[1], ratio, ROPE * maes.mean())
        for name, value in found.items():
            add(name, family, other, value)
    return rows
