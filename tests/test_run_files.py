"""El registre d'una passada s'escriu a JSON sense perdre res en silenci."""
import json

import pytest

from epf.results import run_files


def test_a_value_json_cannot_write_is_refused_and_not_silently_dropped():
    with pytest.raises(TypeError):
        json.dumps(run_files.plain({"bad": object()}))
