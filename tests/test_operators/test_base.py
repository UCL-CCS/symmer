import pytest
import numpy as np
from symmer.operators import PauliwordOp, QuantumState
from qiskit.quantum_info import SparsePauliOp
from openfermion import QubitOperator
from scipy.sparse import rand, csr_matrix
from symmer.utils import matrix_allclose
from symmer.operators.utils import check_adjmat_noncontextual

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


def test_from_matrix_projector_dense():

    for n_qubits in range(1,5):
        mat = np.random.random((2**n_qubits,2**n_qubits)) + 1j*np.random.random((2**n_qubits,2**n_qubits))
        PauliOp_from_matrix = PauliwordOp.from_matrix(mat,
                                                      strategy='projector')
        assert np.allclose(PauliOp_from_matrix.to_sparse_matrix.toarray(),
                           mat)


def test_from_matrix_full_basis_dense():

    for n_qubits in range(1,5):
        mat = np.random.random((2**n_qubits,2**n_qubits)) + 1j*np.random.random((2**n_qubits,2**n_qubits))
        PauliOp_from_matrix = PauliwordOp.from_matrix(mat,
                                                      strategy='full_basis')

        assert np.allclose(PauliOp_from_matrix.to_sparse_matrix.toarray(),
                           mat)


def test_from_matrix_defined_basis_dense():


    for n_qubits in range(2,6):
        n_terms = (4**n_qubits)//2
        op_basis = PauliwordOp.random(n_qubits, n_terms)
        random_mat = sum(op.multiply_by_constant(np.random.uniform(0,10)) for op in op_basis).to_sparse_matrix.toarray()
        PauliOp_from_matrix = PauliwordOp.from_matrix(random_mat,
                                                      strategy='full_basis',
                                                      operator_basis=op_basis)
        assert np.allclose(random_mat,
                           PauliOp_from_matrix.to_sparse_matrix.toarray())


    op_basis = PauliwordOp.from_dictionary({'XX':1,
                                         'ZZ':2,
                                         'YY':2,
                                         'YI':-1,
                                         'YY':2,
                                         'ZX':2})

    mat = np.array([[ 2.+0.j,  0.+0.j,  0.+1.j, -1.+0.j],
                   [ 0.+0.j, -2.+0.j,  3.+0.j,  0.+1.j],
                   [ 0.-1.j,  3.+0.j, -2.+0.j,  0.+0.j],
                   [-1.+0.j,  0.-1.j,  0.+0.j,  2.+0.j]])
    PauliOp_from_matrix = PauliwordOp.from_matrix(mat, strategy='full_basis', operator_basis=op_basis)
    assert np.allclose(PauliOp_from_matrix.to_sparse_matrix.toarray(), mat)


def test_from_matrix_projector_sparse():
    density = 0.8
    for n_qubits in range(1,5):
        dim = 2 ** n_qubits
        mat = rand(dim, dim,
                  density=density,
                  format='csr',
                   dtype=complex)
        PauliOp_from_matrix = PauliwordOp.from_matrix(mat,
                                                      strategy='projector')
        assert np.allclose(PauliOp_from_matrix.to_sparse_matrix.toarray(),
                           mat.toarray())


def test_from_matrix_full_basis_sparse():

    density = 0.8
    for n_qubits in range(1,5):
        dim = 2 ** n_qubits
        mat = rand(dim, dim,
                          density=density,
                          format='csr',
                   dtype=complex)
        PauliOp_from_matrix = PauliwordOp.from_matrix(mat,
                                                      strategy='full_basis')

        assert np.allclose(PauliOp_from_matrix.to_sparse_matrix.toarray(),
                           mat.toarray())


def test_from_matrix_defined_basis_sparse():

    op_basis = PauliwordOp.from_dictionary({
         'II': 1,
         'IZ': 1,
         'ZI': 1,
         'ZZ': 1,
         'IX': 1,
         'IY': 1,
         'ZX': 1,
         'ZY': 1,
         'XI': 1,
         'XZ': 1,
         'YI': 1,
         'YZ': 1,
         'XX': 1,
         'XY': 1,
         'YX': 1,
         'YY': 1 })

    mat = np.array([[1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, -1j, 0],
                    [1, 0, 0, 0]])
    sparse_mat = csr_matrix(mat)
    PauliOp_from_matrix = PauliwordOp.from_matrix(sparse_mat,
                                                  strategy='full_basis',
                                                  operator_basis=op_basis)
    assert np.allclose(PauliOp_from_matrix.to_sparse_matrix.toarray(), mat)


def test_from_matrix_incomplete_op_basis():
    """
    Test to see if warning thrown if supplied basis is not enough.
    Returns:

    """
    op_basis = PauliwordOp.from_dictionary({'XX': 1})
    mat = np.array([[2. + 0.j, 0. + 0.j, 0. + 1.j, -1. + 0.j],
                    [0. + 0.j, -2. + 0.j, 3. + 0.j, 0. + 1.j],
                    [0. - 1.j, 3. + 0.j, -2. + 0.j, 0. + 0.j],
                    [-1. + 0.j, 0. - 1.j, 0. + 0.j, 2. + 0.j]])

    with pytest.warns(UserWarning):
        PauliwordOp.from_matrix(mat, operator_basis=op_basis)


def test_from_matrix_incomplete_op_basis_sparse():
    """
    Test to see if warning thrown if supplied basis is not enough.
    Returns:

    """
    op_basis = PauliwordOp.from_dictionary({'XX': 1})

    dim = 2**op_basis.n_qubits
    mat = rand(dim, dim,
               density=0.5,
               format='csr',
               dtype=complex)

    with pytest.warns(UserWarning):
        PauliwordOp.from_matrix(mat, operator_basis=op_basis)


def test_from_matrix_to_matrix():
    n_qubits = 3

    mat = np.random.random((2**n_qubits,2**n_qubits)) + 1j*np.random.random((2**n_qubits,2**n_qubits))
    PauliOp_from_matrix = PauliwordOp.from_matrix(mat)
    PauliOp_to_matrix = PauliOp_from_matrix.to_sparse_matrix.toarray()
    assert np.allclose(PauliOp_to_matrix, mat)


def test_from_matrix_projector_incorrect_input():
    n_q = 1
    mat = [[1, 0],
           [0, -1]]
    
    with pytest.raises(ValueError):
        PauliwordOp._from_matrix_projector(mat,
                                           n_qubits=n_q)



def test_from_openfermion():
    expected = {'XX': 0.5,
                'YY': 0.5+2j,
                'IZ': -0.5,
                'XZ': -0.5-3j}
    of_operator = QubitOperator()
    for p, coeff in expected.items():
        p_str = ' '.join([f'{sig}{ind}' for ind, sig in enumerate(p) if sig!= 'I'])
        of_operator += QubitOperator(p_str, coeff)

    Pop = PauliwordOp.from_openfermion(of_operator)
    assert PauliwordOp.from_dictionary(expected) == Pop
    assert Pop.n_qubits == 2
    assert expected == Pop.to_dictionary


def test_from_openfermion_qubit_specified():
    expected = {'XX': 0.5,
                'YY': 0.5+2j,
                'IZ': -0.5,
                'XZ': -0.5-3j}
    of_operator = QubitOperator()
    for p, coeff in expected.items():
        p_str = ' '.join([f'{sig}{ind}' for ind, sig in enumerate(p) if sig!= 'I'])
        of_operator += QubitOperator(p_str, coeff)

    three_q  = {'XXI': 0.5,
                'YYI': 0.5+2j,
                'IZI': -0.5,
                'XZI': -0.5-3j}
    Pop = PauliwordOp.from_openfermion(of_operator, n_qubits=3)
    assert Pop.n_qubits == 3
    assert Pop == PauliwordOp.from_dictionary(three_q)


def test_to_openfermion():
    expected = {'XX': 0.5,
                'YY': 0.5+2j,
                'IZ': -0.5,
                'XZ': -0.5-3j}
    of_operator = QubitOperator()
    for p, coeff in expected.items():
        p_str = ' '.join([f'{sig}{ind}' for ind, sig in enumerate(p) if sig!= 'I'])
        of_operator += QubitOperator(p_str, coeff)

    Pop = PauliwordOp.from_dictionary(expected)
    assert Pop.to_openfermion == of_operator


def test_from_qiskit():
    expected = {'XX': 0.5,
                'YY': 0.5+2j,
                'IZ': -0.5,
                'XZ': -0.5-3j}
    Pkeys, coeffs = zip(*expected.items())
    qiskit_op = SparsePauliOp(Pkeys, coeffs=coeffs)
    Pop = PauliwordOp.from_qiskit(qiskit_op)
    assert Pop.to_dictionary == expected
    assert Pop.n_qubits == 2
    assert PauliwordOp.from_dictionary(expected) == Pop


def test_to_qiskit():
    expected = {'XX': 0.5,
                'YY': 0.5+2j,
                'IZ': -0.5,
                'XZ': -0.5-3j}
    Pkeys, coeffs = zip(*expected.items())
    qiskit_op = SparsePauliOp(Pkeys, coeffs=coeffs)
    Pop = PauliwordOp.from_dictionary(expected)
    assert Pop.to_qiskit == qiskit_op
    assert Pop.n_qubits == 2


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
        (['XX', 'YY', 'ZZ', 'II'],True),
        (['II', 'ZZ', 'ZX', 'ZY', 'XZ', 'YZ', 'XX', 'XY', 'YX', 'YY'], False),
        (['III','IIZ','ZII','IXZ','IYZ','YYZ'], False),
        (['IZI', 'ZII','IIY','ZZY','XXZ','XYZ','YXZ','YYZ','XXX','XYX','YXX','YYX'], True)
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

def test_is_noncontextual_generators():
    """
    noncontextual test that breaks if only 2n generators are used rather than 3n generators
    Returns:

    """
    Hnc = PauliwordOp.from_dictionary({'IIIZX': (0.04228614428142647-0j),
                                        'IIIZY': (-0.30109670698419544-0j),
                                        'IIZZX': (0.04228614428142647-0j),
                                        'IIZZY': (-0.30109670698419544-0j),
                                        'IZIZX': (0.04228614428142647-0j),
                                        'IZIZY': (-0.30109670698419544-0j),
                                        'IZZZX': (0.04228614428142647-0j),
                                        'IZZZY': (-0.30109670698419544-0j),
                                        'ZIIZX': (0.04228614428142647-0j),
                                        'ZIIZY': (-0.30109670698419544-0j),
                                        'ZIZZX': (0.04228614428142647-0j),
                                        'ZIZZY': (-0.30109670698419544-0j),
                                        'ZZIZX': (0.04228614428142647-0j),
                                        'ZZIZY': (-0.30109670698419544-0j),
                                        'ZZZZX': (0.04228614428142647-0j),
                                        'ZZZZY': (-0.30109670698419544-0j),
                                        'IIIXI': (-1.6377047626147634-0j),
                                        'IIIYI': (-0.8887783867443338-0j),
                                        'IIZXI': (-1.6377047626147634-0j),
                                        'IIZYI': (-0.8887783867443338-0j),
                                        'IZIXI': (-1.6377047626147634-0j),
                                        'IZIYI': (-0.8887783867443338-0j),
                                        'IZZXI': (-1.6377047626147634-0j),
                                        'IZZYI': (-0.8887783867443338-0j),
                                        'ZIIXI': (-1.6377047626147634-0j),
                                        'ZIIYI': (-0.8887783867443338-0j),
                                        'ZIZXI': (-1.6377047626147634-0j),
                                        'ZIZYI': (-0.8887783867443338-0j),
                                        'ZZIXI': (-1.6377047626147634-0j),
                                        'ZZIYI': (-0.8887783867443338-0j),
                                        'ZZZXI': (-1.6377047626147634-0j),
                                        'ZZZYI': (-0.8887783867443338-0j)})

    ## note this commented out method is gives incorrect answer
    # assert check_adjmat_noncontextual(Hnc.generators.adjacency_matrix), 'noncontexutal operator is being correctly defined as noncontextual'
    assert Hnc.is_noncontextual, 'noncontexutal operator is being incorrectly being defined as contextual'

def test_is_noncontextual_anticommuting_H():
    """
    noncontextual test that breaks if only 2n generators are used rather than 3n generators
    Returns:

    """
    Hnc = PauliwordOp.from_dictionary({
             'ZZZI': (1.2532436410975218-0j),
             'IIXI': (0.8935108507410493-0j),
             'ZIYI': (-1.1362909076230914+0j),
             'IXZI': (-0.05373661687140326+0j),
             'ZYZI': (-1.0012312990477774+0j),
             'XXYI': (-0.045809456087963205+0j),
             'YXYZ': (0.21569499626612557-0j),
             'YXYX': (-0.5806963175396661+0j),
             'YXYY': (0.3218493853030614-0j)})

    assert Hnc.is_noncontextual, 'noncontexutal operator is being correctly defined as noncontextual'

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
    assert ((P1.to_qiskit.dot(P2.to_qiskit)).simplify() - (P1*P2).to_qiskit).simplify() == SparsePauliOp(['III'],
                                                                                                          coeffs=[0.+0.j])

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

def test_to_sparse_matrix_large_operator():
    """ Tests multiplication and the Qiskit conversion
    """
    H = PauliwordOp.from_dictionary({'ZIIIIIIIZIXXXIII': (-1.333664871035997-0.6347579982999967j),
                                     'IIIIIYIIXYZZIXXI': (0.6121055433989232+2.0175827791182313j),
                                     'IIIXIZZIIZIIXIZI': (-0.5187971729475656+1.2184045529704965j),
                                     'ZIIXYYZZIYYXXIZY': (0.6788676757886678+1.867085666718753j),
                                     'IZXIYIXYXIIIZZIX': (-1.0665060328185856-0.5702647494844407j),
                                     'ZIXXIIIZIIIIZIXX': (0.17268863171166954-0.07117422292367692j),
                                     'IIYXIIYIIIXIIZXI': (0.03704770372393225-0.21589376964746243j),
                                     'IYIZXXIXZXXZIIII': (0.29998428856285453-0.9742733999161437j),
                                     'YXIXIIZXZIIIIIYX': (0.3421035543407282-0.20273712913326358j),
                                     'XXXYIIIIXIIXIXIZ': (1.1502457768722+1.3148268876228302j)})
    mat_sparse = H.to_sparse_matrix

    basis = H.copy()
    basis.coeff_vec = np.ones_like(basis.coeff_vec)
    out = PauliwordOp.from_matrix(mat_sparse,
                                  strategy='full_basis',
                                  operator_basis=basis,
                                  disable_loading_bar=True)
    assert H == out, 'to_sparse_matrix of large Pauli operator is failing'


def test_QuantumState_overlap():
    for n_q in range(2,5):
        random_ket_1 = QuantumState.haar_random(n_q, vec_type='ket')
        random_ket_2 = QuantumState.haar_random(n_q, vec_type='ket')

        ket_1 = random_ket_1.to_sparse_matrix.toarray()
        ket_2 = random_ket_2.to_sparse_matrix.toarray()

        assert np.isclose(ket_2.conj().T @ ket_1,
                          random_ket_2.dagger * random_ket_1)

        assert np.isclose(ket_1.conj().T @ ket_2,
                          random_ket_1.dagger * random_ket_2)

        assert np.isclose(random_ket_1.dagger * random_ket_2,
                          (random_ket_2.dagger * random_ket_1).conj())

def test_pauliwordop_hash():
    XI = PauliwordOp.from_dictionary({'XI':1})
    XI_copy = PauliwordOp.from_dictionary({'XI': 1})
    assert hash(XI) == hash(XI_copy)

    YI = PauliwordOp.from_dictionary({'YI': 1})
    assert hash(YI) != hash(XI_copy)

    # different coeff means different hash!
    XI_3 = PauliwordOp.from_dictionary({'XI': 3})
    assert hash(XI) != hash(XI_3)