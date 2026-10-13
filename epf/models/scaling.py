"""Els dos escalats robustos de Lago et al. (2021): el de la mediana i, a sobre, el que
estabilitza la variància, que és el dels dos models lineals.

El valor centrat en la mediana de cada columna i dividit per la seva escala, presos de
les files d'ajust i mai de les que es prediuen; el segon en pren a més l'asinh. LEAR el
pren de Lago et al. (2021) i la xarxa elàstica el pren de LEAR, i la xarxa de l'article
pot triar qualsevol dels dos entre els sis del seu codi, i per això viuen sobre les tres
famílies.
"""
import numpy as np

# La mediana de les desviacions absolutes dividida per aquesta constant estima la
# desviació típica d'una normal; és el que fa `statsmodels.robust.mad`, i el que
# la referència utilitza encara que l'article només digui "MAD".
NORMAL_CONSISTENCY = 0.6744897501960817


class Median:
    """El valor centrat en la mediana de cada columna i dividit per la seva MAD.

    Una columna sense MAD es refusa, que és el que fa LEAR, on el codi de l'article
    divideix per zero sense dir res. Qui declara `std_where_flat` hi posa en canvi la
    desviació típica: és el cas d'una previsió solar que val zero més de la meitat dels
    dies a una hora del vespre i que l'altra meitat arriba a milers de MW. Una columna
    que no es mou gens no té escala de cap mena, i qui la vulgui ajustar l'ha de treure
    abans.
    """

    def __init__(self, values, std_where_flat=False):
        self.median = np.median(values, axis=0)
        scale = np.median(np.abs(values - self.median), axis=0) / NORMAL_CONSISTENCY
        if std_where_flat:
            scale = np.where(scale > 0, scale, np.std(values, axis=0))
        if not scale.all():
            raise ValueError("alguna columna de les files d'ajust no té dispersió")
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
