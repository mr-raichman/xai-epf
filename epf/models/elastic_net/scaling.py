"""Els dos escalats robustos de Lago et al. (2021): el de la mediana, i l'asinh a sobre.

La mediana i l'escala de cada columna es prenen de les files d'ajust, i mai de les que
es prediuen.
"""

import numpy as np

from epf import Refusal

# La mediana de les desviacions absolutes dividida per aquesta constant estima la
# desviació típica d'una normal; és el que fa `statsmodels.robust.mad`, i el que
# la referència utilitza encara que l'article només digui "MAD".
NORMAL_CONSISTENCY = 0.6744897501960817


class Median:
    """El valor centrat en la mediana de cada columna i dividit per la seva MAD.

    Una columna sense MAD es refusa, on el codi de l'article divideix per zero sense
    dir res. Qui declara `std_where_flat` hi posa en canvi la desviació típica: és el
    cas d'una previsió solar que val zero més de la meitat dels dies a una hora del
    vespre i que l'altra meitat arriba a milers de MW. Una columna que no es mou gens
    no té escala de cap mena, i qui la vulgui ajustar l'ha de treure abans. `names` són
    els noms de les columnes, que diuen quina es refusa.
    """

    def __init__(self, values, names, std_where_flat=False):
        self.median = np.median(values, axis=0)
        scale = np.median(np.abs(values - self.median), axis=0) / NORMAL_CONSISTENCY
        if std_where_flat:
            scale = np.where(scale > 0, scale, np.std(values, axis=0))
        flat = [name for name, spread in zip(names, scale, strict=True) if not spread]
        if len(flat) == 1:
            raise Refusal(f"1 columna no té dispersió a les files d'ajust, {flat[0]}")
        if flat:
            raise Refusal(
                f"{len(flat)} columnes no tenen dispersió a les files d'ajust, la "
                f"primera {flat[0]}"
            )
        self.scale = scale

    def forward(self, values):
        return (values - self.median) / self.scale

    def back(self, values):
        return values * self.scale + self.median


class Scaler(Median):
    """Transformació de variància estabilitzada: asinh del valor centrat i escalat."""

    def forward(self, values):
        return np.arcsinh(super().forward(values))

    def back(self, values):
        return super().back(np.sinh(values))
