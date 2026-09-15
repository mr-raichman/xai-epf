"""Actualització de totes les fonts de dades."""
from epf.data import esios, omie


def main():
    print("OMIE")
    omie.download()
    print("ESIOS")
    esios.download()
