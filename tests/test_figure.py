import numpy as np

from epf.contract import figure


def test_a_model_with_no_contribution_is_left_out_of_the_importance(tmp_path):
    """Una xarxa amb tots els pesos d'entrada a zero ho prediu tot igual."""
    groups = np.array(["price_lag1", "price_lag1", "calendar"])
    model = {
        "values": np.array([[1.0, -1.0, 2.0]]),
        "groups": groups,
        "columns": np.array(["x", "y", "z"]),
    }
    constant = model | {"values": np.zeros((1, 3))}
    figure.importance(tmp_path, [model, constant], figure.OTHER_COLOUR)
    figure.importance(tmp_path, [constant], figure.OTHER_COLOUR)
    assert {path.stem for path in tmp_path.iterdir()} >= {"importance", "groups"}


def test_a_number_is_written_with_the_memorias_separators():
    # Xifres que les figures escriuen: la demanda en MW, una diferència amb signe, el
    # menys que matplotlib ja hi posa, una R², un pendent minúscul i un nom de mes, que
    # no és cap nombre.
    assert figure.catalan("30000") == "30.000"
    assert figure.catalan("1234567.5") == "1.234.567,5"
    assert figure.catalan("-12.3") == "\N{MINUS SIGN}12,3"
    assert figure.catalan("\N{MINUS SIGN}150") == "\N{MINUS SIGN}150"
    assert figure.catalan("0.87") == "0,87"
    assert figure.catalan("999") == "999"
    assert figure.catalan("-3.98e-05") == "\N{MINUS SIGN}3,98e\N{MINUS SIGN}05"
    assert figure.catalan("gener") == "gener"
