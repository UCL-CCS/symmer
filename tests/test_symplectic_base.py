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
def three_qubit_symp_matrix():
    return [
        [0,0,0,0,0,0],
        [1,1,1,0,0,0],
        [1,1,1,1,1,1],
        [0,0,0,1,1,1]
    ]
@pytest.fixture  
def three_qubit_pauli_list():
    return ['III', 'XXX', 'YYY', 'ZZZ']
@pytest.fixture
def coeff_vec():
    return np.random.random(4)
    
def test_from_list(
        three_qubit_pauli_list, 
        three_qubit_symp_matrix, 
        coeff_vec
    ):
    assert (
        PauliwordOp.from_list(three_qubit_pauli_list, coeff_vec) ==
        PauliwordOp(three_qubit_symp_matrix, coeff_vec)
    )

def test_from_dictionary(
        three_qubit_pauli_list, 
        three_qubit_symp_matrix, 
        coeff_vec
    ):
    pauli_dict = dict(zip(three_qubit_pauli_list, coeff_vec))
    assert (
        PauliwordOp.from_dictionary(pauli_dict) ==
        PauliwordOp(three_qubit_symp_matrix, coeff_vec)
    )

##################################################
# Testing algebraic manipulation of PauliwordOps #
##################################################

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