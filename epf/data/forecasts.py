"""El conjunt de les previsions D+1 de Red Eléctrica: demanda, eòlica i fotovoltaica.

Tres indicadors d'ESIOS, cadascun a la seva carpeta dins de `paths.FORECASTS`: un
fitxer d'ESIOS només porta el període al nom, i tres `2019.json` en una carpeta plana
xocarien. No són a `esios.SERIES`, com tampoc no hi és el preu: la taula és dels
indicadors que cap conjunt no ha fet seus, i els dos llaços que la recorren, el de
l'auditoria i el de refrescar-los a mà, els buscarien a la carpeta d'on han marxat.

Executar el mòdul fa existir el conjunt: el baixa sencer i diu què queda a disc.
`load` passa el contracte de `contract.py` a cada sèrie abans de tornar-la. El que
només es pot preguntar a ESIOS és a `audit.py`.
"""
import pathlib

from epf.data import esios
from epf.data.contract import refuse_incoherent
from epf.paths import FORECASTS

# El nom del conjunt, que el contracte porta quan refusa una de les seves sèries.
NAME = "forecasts"

# Cada previsió, pel nom amb què la coneix el codi. La d'eòlica més fotovoltaica que
# pren Lago et al. (2021) no hi és: ESIOS la publica com a 10358 i és exactament la
# suma de les dues d'aquí.
INDICATORS = {"demand": 1775, "wind": 1777, "solar": 1779}


def _folder(indicator, directory):
    return pathlib.Path(directory) / str(indicator)


def read(name, directory=FORECASTS):
    """Una previsió en MW tal com la publica ESIOS, en dies per hores, i les hores
    publicades cada dia. Encara sense el contracte, que és de `load`."""
    indicator = INDICATORS[name]
    stored = esios.values(_folder(indicator, directory))
    return esios.hourly(esios.single_geography(stored, indicator))


def load(directory=FORECASTS):
    """Les tres previsions en MW, pel seu nom, amb un dia de lliurament per fila i una
    hora per columna.

    Cada una ha passat el contracte amb el seu nom, de manera que una refusa diu quina:
    qui carrega és qui comprova.
    """
    return {name: refuse_incoherent(f"{NAME} {name}", read(name, directory)[0])
            for name in INDICATORS}


def download(directory=FORECASTS):
    """Baixa les tres previsions, cadascuna a la seva carpeta."""
    session = esios.session_for()
    for indicator in INDICATORS.values():
        esios.download(indicator, _folder(indicator, directory), session)


def main():
    download()
    for name, frame in load().items():
        print(f"{name}: {len(frame)} dies, de {frame.index[0].date()} "
              f"a {frame.index[-1].date()}")
