"""Noms que una configuració no pot prendre."""
import pytest

from epf.config.experiments import CONFIGURATIONS, MEASUREMENT, TEST, refuse_reserved


def test_a_configuration_cannot_be_called_like_a_tier():
    # Una configuració dita «test» donaria `runs/measurement/<etapa>/<commit>_test/`,
    # que ja va existir un cop i es llegeix com una execució d'assaig.
    for name in (MEASUREMENT, TEST):
        with pytest.raises(ValueError):
            refuse_reserved(["base", name])


def test_the_configurations_in_force_pass_it():
    refuse_reserved(CONFIGURATIONS)
