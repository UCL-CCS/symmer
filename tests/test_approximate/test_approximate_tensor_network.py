import pytest
import numpy as np
from symmer.approximate import MPOOp, find_groundstate_quimb
from symmer.operators import PauliwordOp, QuantumState
from symmer.utils import exact_gs_energy

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
        ):
    MPO = MPOOp(pauli_list_1, coeff_vec_1)
    matrix_MPO = MPO.to_matrix

    WordOp = PauliwordOp.from_list(pauli_list_1, coeff_vec_1)
    matrix_WordOp = WordOp.to_sparse_matrix.toarray()

    assert(np.allclose(matrix_MPO, matrix_WordOp))

def test_from_dictionary(
        pauli_list_1,
        coeff_vec_1):
    pauli_dict = dict(zip(pauli_list_1, coeff_vec_1))
    MPO = MPOOp.from_dictionary(pauli_dict)
    matrix_MPO = MPO.to_matrix

    WordOp = PauliwordOp.from_list(pauli_list_1, coeff_vec_1)
    matrix_WordOp = WordOp.to_sparse_matrix.toarray()

    assert(np.allclose(matrix_MPO, matrix_WordOp))

############################################
# Testing QUIMB dmrg sovler #
############################################

# def test_find_groundsate_quimb(
#         pauli_list_1,
#         coeff_vec_1
#         ):
#     MPO = MPOOp(pauli_list_1, coeff_vec_1)

#     mpostate = find_groundstate_quimb(MPO)

#     assert(type(mpostate) == QuantumState)
