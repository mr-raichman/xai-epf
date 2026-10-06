"""Actualització de totes les fonts de dades."""
from epf.data import esios, omie


def main():
    print("OMIE", flush=True)
    omie.download()
    print("ESIOS", flush=True)
    esios.download()
