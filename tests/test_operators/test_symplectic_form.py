import numpy as np

from symmer.operators import symplectic_to_string


def test_symplectic_to_string():

    I_term = np.array([0, 0])
    assert symplectic_to_string(I_term) == "I", "identity term not identified"

    IXYZ_term = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    assert (
        symplectic_to_string(IXYZ_term) == "IXYZ"
    ), "Pauliword not correctly translated"
