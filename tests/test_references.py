import re
import unicodedata

import pytest

from epf import paths

REFERENCES = paths.ROOT / "references.bib"

# Una cita per autor i any: el primer cognom, els altres darrere de comes i d'una «i», o
# «et al.», i l'any entre parèntesis o, dins d'un parèntesi, darrere d'una coma. Un
# cognom pot dur un apòstrof darrere la inicial, O'Sullivan, que sense es llegiria com
# un altre cognom. Qualsevol altra forma no es veu.
SURNAME = r"[A-ZÀ-Ý](?:'[A-ZÀ-Ý])?[a-zà-ÿ]+"
AUTHORS = rf"({SURNAME})(?:, {SURNAME})*(?: i {SURNAME})?(?: et al\.)?"
CITATIONS = (
    re.compile(rf"\b{AUTHORS} \((\d{{4}}[a-z]?)\)"),
    re.compile(rf"\({AUTHORS}, (\d{{4}}[a-z]?)\)"),
)

# Un salt de línia dins d'un comentari o d'un docstring, amb el sagnat i el `#` de la
# línia següent, perquè una cita partida entre dues línies es llegeixi sencera.
LINE_BREAK = re.compile(r"\s*\n\s*(?:#\s*)?")


def keys(text):
    """Torna la clau de cada cita del text, el primer cognom sense accents i l'any."""
    flat = LINE_BREAK.sub(" ", text)
    found = set()
    for citation in CITATIONS:
        for surname, year in citation.findall(flat):
            plain = (
                unicodedata.normalize("NFKD", surname)
                .encode("ascii", "ignore")
                .decode()
            )
            found.add(plain.lower() + year)
    return found


@pytest.mark.parametrize(
    "text, key",
    [
        (
            "el test que Bergmeir, Hyndman i Koo (2018) fan a la validació creuada",
            "bergmeir2018",
        ),
        (
            "la fracció d'encerts (Nowotarski i Weron,\n    2018). Es fa hora a hora",
            "nowotarski2018",
        ),
        (
            "la de Trebbien et al.\n    (2023): la mitjana del valor absolut",
            "trebbien2023",
        ),
        (
            "al codi de Marcjasz et al.\n# (2023): `FLOOR + STRETCH * softplus`",
            "marcjasz2023",
        ),
        (
            (
                "com aconsellen Hyndman i\n# Athanasopoulos (2021) per a una sèrie "
                "estacional"
            ),
            "hyndman2021",
        ),
        (
            "el test de cobertura incondicional de Kupiec (1995) a cada hora",
            "kupiec1995",
        ),
    ],
)
def test_a_citation_gives_its_first_author_and_year(text, key):
    assert keys(text) == {key}


def test_every_citation_in_the_code_has_its_entry():
    cited = set()
    for path in sorted((paths.ROOT / "epf").rglob("*.py")):
        cited |= keys(path.read_text(encoding="utf-8"))
    listed = set(
        re.findall(r"@\w+\{([^,\s]+),", REFERENCES.read_text(encoding="utf-8"))
    )
    assert cited, "no s'ha trobat cap cita al codi"
    assert not cited - listed, (
        f"cites sense entrada a {REFERENCES.name}: {sorted(cited - listed)}"
    )
