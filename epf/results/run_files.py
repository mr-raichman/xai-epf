"""La crida a git amb què una passada i una cerca diuen quin commit les fa.

El commit nomena la passada, perquè és el que fixa tots els seus números.
"""
import subprocess

from epf import paths


def git(*arguments):
    """La sortida d'una ordre de git sobre el repositori. Una passada es diu pel commit
    que la fa."""
    return subprocess.run(("git", *arguments), cwd=paths.ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
