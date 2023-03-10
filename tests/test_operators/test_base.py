import pytest
import numpy as np
from symmer.operators import PauliwordOp
from functools import reduce

P_matrices ={
    'X': np.array([[0, 1],
                   [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j],
                   [+1j, 0]], dtype=complex),
    'Z': np.array([[1, 0],
                   [0, -1]], dtype=complex),
    'I': np.array([[1, 0],
                   [0, 1]], dtype=complex),

}

####################################################################
# Assertion errors arising from poorly defined symplectic matrices #
####################################################################

def test_init_symplectic_float_type():
    """
    if input symplectic matrix is not a boolean or (0,1) ints  need error to be raised
    This checks for floats
    """
    coeff = [1]
    symp_matrix = [
        [0.,1.,0.,1.,0.,1.]
    ]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, coeff)

def test_init_symplectic_nonbinary_ints_type():
    """
    if input symplectic matrix is ints but not 0,1 check code raises error
    """
    coeff = [1]
    symp_matrix = [
        [0,1,2,3,4,5]
    ]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, coeff)

def test_init_symplectic_str_type():
    """
    if input symplectic matrix is string check error raised
    """
    coeff = [1]
    symp_matrix = [
        ['0','1','1','0','1','1']
    ]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, coeff)

def test_incompatible_length_of_symp_matrix_and_coeff_vec():
    """
    For a symplectic matrix of 2 rows check that if only 1 coeff defined an error will be thrown.
    """
    symp_matrix = [
        [0,1,0,1,0,1],
        [1,0,1,0,1,0],
    ]
    coeff = [1]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, coeff)

def test_init_symplectic_incorrect_dimension():
    """
    if input symplectic matrix is not correct dimension throw an error
    """
    coeff = [1]

    # error here symplectic matrix wrong dimensions
    symp_matrix = [
        [[[0,1,1,0,1,1]]]
    ]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, coeff)

def test_init_symplectic_2D_but_odd_columns():
    """
    if input symplectic matrix is 2D, but has odd number of columns throw an error (columns must alway be even)
    """
    coeff = [1,1]

    # error here... number of columns must be even (not odd... in this case 3)
    symp_matrix = [
        [0,0,1],
        [1, 0, 1]
    ]
    with pytest.raises(AssertionError):
        PauliwordOp(symp_matrix, coeff)

def test_init_symplectic_int_coeff():
    """
    if input symplectic matrix is 2D, but has odd number of columns throw an error (columns must alway be even)
    """
    # error here, should be list or array
    coeff = 1

    symp_matrix = [
        [0,0,1,1]
    ]
    with pytest.raises(TypeError):
        PauliwordOp(symp_matrix, coeff)

############################################
# Testing empty PauliwordOp                #
############################################

def test_empty():
    n_qubits = 3
    P_empty = PauliwordOp.empty(n_qubits)
    assert P_empty == PauliwordOp([[0]*6], [0])
    assert P_empty.n_terms==1
    assert np.array_equal(P_empty.coeff_vec, np.array([0]))
    assert P_empty.n_qubits == n_qubits

def test_empty_cleanup():
    n_qubits = 3

    P_empty = PauliwordOp.empty(n_qubits)
    P_empty = P_empty.cleanup()
    assert P_empty.n_qubits == n_qubits
    assert P_empty.symp_matrix.shape == (0, 2*n_qubits)

############################################
# Testing random PauliwordOp               #
############################################

def test_PauliwordOp_random_diag():

    diagonal = True

    complex_coeffs = True
    n_qubits=3
    n_terms=4
    P_random = PauliwordOp.random(n_qubits=n_qubits,
                                  n_terms=n_terms,
                                  diagonal=diagonal,
                                  complex_coeffs=complex_coeffs)
    assert np.array_equal(P_random.X_block,
                          np.zeros_like(P_random.X_block).astype(bool))

def test_PauliwordOp_random_complex():

    ## fixed
    diagonal = False
    n_qubits = 4
    n_terms = 30
    ##

    complex_coeffs = True
    P_random_complex = PauliwordOp.random(n_qubits=n_qubits,
                                  n_terms=n_terms,
                                  diagonal=diagonal,
                                  complex_coeffs=complex_coeffs)

    assert P_random_complex.coeff_vec.dtype == np.complex128
    assert np.sum(np.abs(P_random_complex.coeff_vec.imag))>0

    complex_coeffs = False
    P_random_real = PauliwordOp.random(n_qubits=n_qubits,
                                  n_terms=n_terms,
                                  diagonal=diagonal,
                                  complex_coeffs=complex_coeffs)

    assert np.array_equal(P_random_real.coeff_vec.imag,
                          np.zeros((P_random_real.n_terms)))

def test_PauliwordOp_haar():

    ## fixed
    n_qubits = 3
    ##
    P_haar_random = PauliwordOp.haar_random(n_qubits)
    assert P_haar_random.n_qubits == n_qubits

    # check unitary
    mat = P_haar_random.to_sparse_matrix.toarray()
    assert np.allclose(np.eye(mat.shape[0]), mat.dot(mat.T.conj())), 'haar random operator not unitary'

############################################
# Testing different initialization methods #
############################################

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
    # real coeffs
    return np.random.random(4)

@pytest.fixture
def coeff_vec_2():
    # complex coeffs
    return np.random.random(4) + 1j*np.random.random(4)
    
def test_from_list(
        pauli_list_1, 
        symp_matrix_1, 
        coeff_vec_1
    ):
    assert (
        PauliwordOp.from_list(pauli_list_1, coeff_vec_1) ==
        PauliwordOp(symp_matrix_1, coeff_vec_1)
    )

def test_from_list_incorrect_str():
    """
    raise error if lower case pauli operators used
    """
    with pytest.raises(AssertionError):
        PauliwordOp.from_list(['ixi', 'zzi'], [0,1])

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

def test_to_dictionary(
    pauli_list_1, 
    symp_matrix_1, 
    coeff_vec_1
    ):
    pauli_dict = dict(zip(pauli_list_1, coeff_vec_1))
    assert PauliwordOp.from_dictionary(
        pauli_dict
    ).to_dictionary == pauli_dict


def test_from_matrix(
        pauli_list_1,
        symp_matrix_1,
        coeff_vec_1
    ):
    n_qubits = len(pauli_list_1[0])

    # build matrix
    mat = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for index, p_op in enumerate(pauli_list_1):
        coeff = coeff_vec_1[index]
        p_op_mat = reduce(np.kron, [P_matrices[sig] for sig in p_op])
        mat+= coeff*p_op_mat
    # generate Pop from matrix
    PauliOp_from_matrix = PauliwordOp.from_matrix(mat)

    pauli_dict = dict(zip(pauli_list_1, coeff_vec_1))
    PauliOp_from_dict = PauliwordOp.from_dictionary(pauli_dict)
    assert PauliOp_from_matrix == PauliOp_from_dict

def test_from_matrix_to_matrix():
    n_qubits = 3

    mat = np.random.random((2**n_qubits,2**n_qubits)) + 1j*np.random.random((2**n_qubits,2**n_qubits))
    PauliOp_from_matrix = PauliwordOp.from_matrix(mat)
    PauliOp_to_matrix = PauliOp_from_matrix.to_sparse_matrix.toarray()
    assert np.allclose(PauliOp_to_matrix, mat)

##################################################
# Testing algebraic manipulation of PauliwordOps #
##################################################

def test_Y_count(
        symp_matrix_1, 
        coeff_vec_1
    ):
    P = PauliwordOp(symp_matrix_1, coeff_vec_1)
    assert np.all(P.Y_count == np.array([0,0,3,0]))

def test_getitem(
        pauli_list_2, 
        coeff_vec_2
    ):
    P = PauliwordOp.from_list(pauli_list_2, coeff_vec_2)
    assert all(
        [P[i] == PauliwordOp.from_list([pauli_list_2[i]], [coeff_vec_2[i]]) 
        for i in range(-4,4)]
    )

def test_iter(
        pauli_list_2, 
        coeff_vec_2
    ):
    P = PauliwordOp.from_list(pauli_list_2, coeff_vec_2)
    assert all(
        [Pi==PauliwordOp.from_list([pauli_list_2[i]], [coeff_vec_2[i]]) 
        for i, Pi in enumerate(P)]
    )
        
def test_cleanup_zeros(symp_matrix_1):
    P = PauliwordOp.random(3,10)
    P.coeff_vec[:] = 0
    assert P.cleanup().n_terms == 0

def test_cleanup():
    P = PauliwordOp.from_list(['XXX', 'YYY', 'XXX', 'YYY'], [1,1,-1,1])
    assert P == PauliwordOp.from_list(['YYY'], [2])

def test_addition():
    P = PauliwordOp.random(3, 10)
    assert P + P == P * 2

def test_subtraction():
    P = PauliwordOp.random(3, 10)
    assert (P-P).n_terms == 0

def test_termwise_commutatvity(
        pauli_list_1, pauli_list_2
    ):
    P1 = PauliwordOp.from_list(pauli_list_1)
    P2 = PauliwordOp.from_list(pauli_list_2)
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
    P = PauliwordOp.from_list(pauli_list_2)
    assert(
        np.all(P.adjacency_matrix == np.array([
            [True , False, True , False],
            [False, True , True , False],
            [True , True , True , True ],
            [False, False, True , True ]
        ]))
    )

@pytest.mark.parametrize(
    "P_list,is_noncon", 
    [
        (['XZ', 'ZX', 'ZI', 'IZ'],False), 
        (['XZ', 'ZX', 'XX', 'YY'],True),
    ]
)
def test_is_noncontextual(P_list, is_noncon):
    P = PauliwordOp.from_list(P_list)
    assert P.is_noncontextual == is_noncon

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

def test_multiplication_1():
    """ Tests multiplication and the OpenFermion conversion
    """
    P1 = PauliwordOp.random(3, 10)
    P2 = PauliwordOp.random(3, 10)
    assert (P1 * P2).to_openfermion == P1.to_openfermion * P2.to_openfermion

def test_multiplication_2():
    """ Tests multiplication and the Qiskit conversion
    """
    P1 = PauliwordOp.random(3, 10)
    P2 = PauliwordOp.random(3, 10)
    assert (P1 * P2).to_qiskit == P1.to_qiskit @ P2.to_qiskit

def test_to_sparse_matrix_1():
    """ Tests multiplication and the Qiskit conversion
    """
    P1 = PauliwordOp.random(3, 10)
    P2 = PauliwordOp.random(3, 10)
    assert np.allclose(
        (P1*P2).to_sparse_matrix.toarray(), 
        P1.to_sparse_matrix.toarray() @ P2.to_sparse_matrix.toarray()
    )

@pytest.mark.parametrize(
    "P_dict,P_array", 
    [
        ({'X':1}, np.array([[0,1],[1,0]])), 
        ({'Y':1}, np.array([[0,-1j],[1j,0]])), 
        ({'Z':1}, np.array([[1,0],[0,-1]])),
        ({'XY':1}, np.array([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]])),
        ({'ZY':1}, np.array([[0,-1j,0,0],[1j,0,0,0],[0,0,0,1j],[0,0,-1j,0]])),
        ({'II':1, 'IX':1, 'XI':1, 'XX':1}, np.ones([4,4]))
    ]
)
def test_to_sparse_matrix_2(
        P_dict, P_array
    ):
    P = PauliwordOp.from_dictionary(P_dict)
    assert np.all(P.to_sparse_matrix.toarray() == P_array)