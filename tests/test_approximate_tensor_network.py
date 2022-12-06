import pytest
import numpy as np
from symmer.approximate import MPOApproximator

@pytest.fixture
def symp_matrix_1():
    return np.array([
        [0,0,0,0,0,0],
        [1,1,1,0,0,0],
        [1,1,1,1,1,1],
        [0,0,0,1,1,1]
    ])
@pytest.fixture
def symp_matrix_2():
    return np.array([
        [0,1,0,1,0,1],
        [1,0,1,0,1,0],
        [1,1,0,0,1,1],
        [0,0,1,1,0,0]
    ])
@pytest.fixture
def pauli_list_1():
    return ['III', 'XXX', 'YYY', 'ZZZ']
@pytest.fixture
def pauli_list_2():
    return ['ZXZ', 'XZX', 'XYZ', 'ZIX']
@pytest.fixture
def coeff_vec_1():
    return np.random.random(4)
@pytest.fixture
def coeff_vec_2():
    return np.random.random(4)


############################################
# Testing different initialization methods #
############################################

def test_from_list(
        pauli_list_1,
        coeff_vec_1,
        symp_matrix_1
        ):
    MPOApproximator(pauli_list_1, coeff_vec_1)
    assert(True)

