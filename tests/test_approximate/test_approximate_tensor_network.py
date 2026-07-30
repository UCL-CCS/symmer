import pytest
import numpy as np

pytest.importorskip("quimb")

from symmer.approximate import MPOOp, find_groundstate_quimb
from symmer.operators import PauliwordOp, QuantumState


@pytest.fixture
def pauli_list_1():
    return ['III', 'XXX', 'YYY', 'ZZZ']


@pytest.fixture
def coeff_vec_1():
    return np.array([0.1, 0.2, 0.3, 0.4])


def test_from_list(pauli_list_1, coeff_vec_1):
    """Small MPO ↔ PauliwordOp matrix equivalence (requires quimb)."""
    MPO = MPOOp(pauli_list_1, coeff_vec_1)
    matrix_MPO = MPO.to_matrix

    WordOp = PauliwordOp.from_list(pauli_list_1, coeff_vec_1)
    matrix_WordOp = WordOp.to_sparse_matrix.toarray()

    assert np.allclose(matrix_MPO, matrix_WordOp)


def test_from_dictionary(pauli_list_1, coeff_vec_1):
    pauli_dict = dict(zip(pauli_list_1, coeff_vec_1))
    MPO = MPOOp.from_dictionary(pauli_dict)
    matrix_MPO = MPO.to_matrix

    WordOp = PauliwordOp.from_list(pauli_list_1, coeff_vec_1)
    matrix_WordOp = WordOp.to_sparse_matrix.toarray()

    assert np.allclose(matrix_MPO, matrix_WordOp)


@pytest.mark.skip(reason="DMRG ground-state path not yet numerically stable in CI")
def test_find_groundstate_quimb(pauli_list_1, coeff_vec_1):
    MPO = MPOOp(pauli_list_1, coeff_vec_1)
    mpostate = find_groundstate_quimb(MPO)
    assert type(mpostate) == QuantumState
