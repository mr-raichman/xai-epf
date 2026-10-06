"""Claus d'API, llegides del fitxer `api-keys` de l'arrel."""
from epf.paths import KEYS


def api_key(name, path=KEYS):
    """Valor de la línia `name=valor` del fitxer de claus."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"no existeix {path}; ha de contenir la línia {name}=...") from None
    for line in lines:
        found, separator, value = line.partition("=")
        if separator and found.strip() == name and value.strip():
            return value.strip()
    raise KeyError(f"falta la línia {name}=... a {path}")
