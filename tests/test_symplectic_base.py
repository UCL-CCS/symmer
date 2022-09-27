import pytest
import numpy as np
from symmer.symplectic import PauliwordOp

####################################################################
# Assertion errors arising from poorly defined symplectic matrices #
####################################################################

def test_bad_symplectic_matrix_entry_type():
    symp_matrix = [
        [0.,1.,0.,1.,0.,1.]
    ]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, [0])

def test_symplectic_matrix_non_binary_entries():
    symp_matrix = [
        [0,1,0,1,0,'1']
    ]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, [0])

def test_incompatible_length_of_symp_matrix_and_coeff_vec():
    symp_matrix = [
        [0,1,0,1,0,1],
        [1,0,1,0,1,0],
    ]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, [0])

############################################
# Testing different initialization methods #
############################################

def test_empty():
    assert PauliwordOp.empty(3) == PauliwordOp([[0]*6], [0])

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
    
def test_from_list(
        pauli_list_1, 
        symp_matrix_1, 
        coeff_vec_1
    ):
    assert (
        PauliwordOp.from_list(pauli_list_1, coeff_vec_1) ==
        PauliwordOp(symp_matrix_1, coeff_vec_1)
    )

def test_from_dictionary(
        pauli_list_1, 
        symp_matrix_1, 
        coeff_vec_1
    ):
    pauli_dict = dict(zip(pauli_list_1, coeff_vec_1))
    assert (
        PauliwordOp.from_dictionary(pauli_dict) ==
        PauliwordOp(symp_matrix_1, coeff_vec_1)
    )

def test_to_ditionary(
    pauli_list_1, 
    symp_matrix_1, 
    coeff_vec_1
    ):
    pauli_dict = dict(zip(pauli_list_1, coeff_vec_1))
    assert PauliwordOp.from_dictionary(
        pauli_dict
    ).to_dictionary == pauli_dict

##################################################
# Testing algebraic manipulation of PauliwordOps #
##################################################

def test_Y_count(
        symp_matrix_1, 
        coeff_vec_1
    ):
    P = PauliwordOp(symp_matrix_1, coeff_vec_1)
    assert np.all(P.Y_count == np.array([0,0,3,0]))
    
def test_cleanup_zeros(symp_matrix_1):
    P = PauliwordOp.random(3,10)
    P.coeff_vec[:] = 0
    assert P.cleanup().n_terms == 0

def test_addition():
    P = PauliwordOp.random(3, 10)
    assert P + P == P * 2

def test_subtraction():
    P = PauliwordOp.random(3, 10)
    assert (P-P).n_terms == 0

def test_termwise_commutatvity(
        pauli_list_1, pauli_list_2
    ):
    P1 = PauliwordOp.from_list(pauli_list_1, np.ones(4))
    P2 = PauliwordOp.from_list(pauli_list_2, np.ones(4))
    assert(
        np.all(P1.commutes_termwise(P2) == np.array([
            [True , True , True , True ],
            [True , False, True , False],
            [False, False, True , True ],
            [False, True , True , False]
        ]))
    )

def test_adjacency_matrix(
    pauli_list_2
    ):
    P = PauliwordOp.from_list(pauli_list_2, np.ones(4))
    assert(
        np.all(P.adjacency_matrix == np.array([
            [True , False, True , False],
            [False, True , True , False],
            [True , True , True , True ],
            [False, False, True , True ]
        ]))
    )

@pytest.mark.parametrize(
    "P1_dict,P2_dict,P1P2_dict", 
    [
        ({'X':1},{'Y':1},{'Z':+1j}), 
        ({'Z':1},{'X':1},{'Y':+1j}), 
        ({'Y':1},{'Z':1},{'X':+1j}), 
        ({'Y':1},{'X':1},{'Z':-1j}), 
        ({'X':1},{'Z':1},{'Y':-1j}), 
        ({'Z':1},{'Y':1},{'X':-1j}),
    ]
)
def test_single_qubit_multiplication(
        P1_dict, P2_dict, P1P2_dict
    ):
    P1   = PauliwordOp.from_dictionary(P1_dict)
    P2   = PauliwordOp.from_dictionary(P2_dict)
    P1P2 = PauliwordOp.from_dictionary(P1P2_dict)
    assert P1 * P2 == P1P2

def test_multiplication():
    P1 = PauliwordOp.random(3, 10)
    P2 = PauliwordOp.random(3, 10)
    assert (P1 * P2).to_openfermion == P1.to_openfermion * P2.to_openfermion