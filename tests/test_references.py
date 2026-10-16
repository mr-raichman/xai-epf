"""Cada font que cita el codi, amb la seva entrada a `references.bib`."""
import re
import unicodedata

import pytest

from epf import paths

REFERENCES = paths.ROOT / "references.bib"

# Una cita per autor i any: el primer cognom, els altres darrere de comes i d'una «i», o
# «et al.», i l'any entre parèntesis. Qualsevol altra forma no es veu.
SURNAME = r"[A-ZÀ-Ý][a-zà-ÿ]+"
CITATION = re.compile(rf"\b({SURNAME})(?:, {SURNAME})*(?: i {SURNAME})?(?: et al\.)?"
                      r" \((\d{4}[a-z]?)\)")


def keys(text):
    """La clau de cada cita del text: el primer cognom, sense accents, i l'any."""
    found = set()
    for surname, year in CITATION.findall(text):
        plain = unicodedata.normalize("NFKD", surname).encode("ascii", "ignore").decode()
        found.add(plain.lower() + year)
    return found


@pytest.mark.parametrize("text, key", [
    ("Bergmeir, Hyndman i Koo (2018)", "bergmeir2018"),
    ("Nowotarski i Weron (2018)", "nowotarski2018"),
    ("Lago et al. (2021)", "lago2021"),
    ("el test de cobertura incondicional de Kupiec (1995) a cada hora", "kupiec1995"),
    ("La partició de Trebbien et al. (2023): les setmanes", "trebbien2023"),
])
def test_a_citation_gives_its_first_author_and_year(text, key):
    assert keys(text) == {key}


def test_every_citation_in_the_code_has_its_entry():
    cited = set()
    for path in sorted((paths.ROOT / "epf").rglob("*.py")):
        cited |= keys(path.read_text(encoding="utf-8"))
    listed = set(re.findall(r"@\w+\{([^,\s]+),", REFERENCES.read_text(encoding="utf-8")))
    assert cited, "no s'ha trobat cap cita al codi"
    assert not cited - listed, f"cites sense entrada a {REFERENCES.name}: {sorted(cited - listed)}"
