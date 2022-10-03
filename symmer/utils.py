from functools import reduce
import numpy as np
import scipy as sp
from typing import Tuple
from qiskit import QuantumCircuit
import pyzx as zx
import openfermion as of

def symplectic_to_string(symp_vec) -> str:
    """
    Returns string form of symplectic vector defined as (X | Z)

    Args:
        symp_vec (array): symplectic Pauliword array

    Returns:
        Pword_string (str): String version of symplectic array

    """
    n_qubits = len(symp_vec) // 2

    X_block = symp_vec[:n_qubits]
    Z_block = symp_vec[n_qubits:]

    Y_loc = np.bitwise_and(X_block, Z_block).astype(bool)
    X_loc = np.bitwise_xor(Y_loc, X_block).astype(bool)
    Z_loc = np.bitwise_xor(Y_loc, Z_block).astype(bool)

    char_aray = np.array(list('I' * n_qubits), dtype=str)

    char_aray[Y_loc] = 'Y'
    char_aray[X_loc] = 'X'
    char_aray[Z_loc] = 'Z'

    Pword_string = ''.join(char_aray)

    return Pword_string

def string_to_symplectic(pauli_str, n_qubits):
    """
    """
    assert(len(pauli_str) == n_qubits), 'Number of qubits is incompatible with pauli string'
    assert (set(pauli_str).issubset({'I', 'X', 'Y', 'Z'})), 'pauliword must only contain X,Y,Z,I terms'

    char_aray = np.array(list(pauli_str), dtype=str)
    X_loc = (char_aray == 'X')
    Z_loc = (char_aray == 'Z')
    Y_loc = (char_aray == 'Y')

    symp_vec = np.zeros(2*n_qubits, dtype=int)
    symp_vec[:n_qubits] += X_loc
    symp_vec[n_qubits:] += Z_loc
    symp_vec[:n_qubits] += Y_loc
    symp_vec[n_qubits:] += Y_loc

    return symp_vec

def count1_in_int_bitstring(i):
    """
    Count number of "1" bits in integer i to be thought of in binary representation

    https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer#109025
    https://web.archive.org/web/20151229003112/http://blogs.msdn.com/b/jeuge/archive/2005/06/08/hakmem-bit-count.aspx
    """
    i = i - ((i >> 1) & 0x55555555)  # add pairs of bits
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)  # quads
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24

def symplectic_to_sparse_matrix(symp_vec, coeff) -> sp.sparse.csr_matrix:
    """
    Returns (2**n x 2**n) matrix of paulioperator kronector product together
     defined from symplectic vector defined as (X | Z)

    This follows because tensor products of Pauli operators are one-sparse: they each have only
    one nonzero entry in each row and column

    Args:
        symp_vec (array): symplectic Pauliword array

    Returns:
        sparse_matrix (csr_matrix): sparse matrix of Pauliword

    """
    n_qubits = len(symp_vec) // 2

    X_block = symp_vec[:n_qubits]
    Z_block = symp_vec[n_qubits:]

    Y_number = sum(np.bitwise_and(X_block, Z_block).astype(int))
    global_phase = (-1j) ** Y_number

    # reverse order to match bitstring int valu of each bit in binary: [..., 32, 16, 8, 4, 2, 1]
    if n_qubits > 64:
        # numpy cannot handle ints over int64s (2**64) therefore use python objects
        binary_int_array = 1 << np.arange(n_qubits - 1, -1, -1).astype(object)
    else:
        binary_int_array = 1 << np.arange(n_qubits - 1, -1, -1)

    x_int = X_block @ binary_int_array
    z_int = Z_block @ binary_int_array

    dimension = 2**n_qubits

    row_ind = np.arange(dimension)
    col_ind = np.bitwise_xor(row_ind, x_int)

    row_inds_and_Zint = np.bitwise_and(row_ind, z_int)
    vals = global_phase * (-1) ** (count1_in_int_bitstring(row_inds_and_Zint)%2)

    sparse_matrix = sp.sparse.csr_matrix(
        (vals, (row_ind, col_ind)),
        shape=(dimension, dimension),
        dtype=complex
            )

    return coeff*sparse_matrix

def symplectic_cleanup(
        symp_matrix:    np.array, 
        coeff_vec:      np.array, 
        zero_threshold: float = None
    ) -> Tuple[np.array, np.array]:
    """ Remove duplicated rows of symplectic matrix terms, whilst summing
    the corresponding coefficients of the deleted rows in coeff_vec
    """
    # order lexicographically using a fast void view implementation...
    # this scales to large numbers of qubits more favourably than np.lexsort
    symp_matrix_view = np.ascontiguousarray(symp_matrix).view(
        np.dtype((np.void, symp_matrix.dtype.itemsize * symp_matrix.shape[1]))
    )
    re_order_indices = np.argsort(symp_matrix_view.ravel())
    # sort the symplectic matrix and vector of coefficients accordingly
    sorted_terms = symp_matrix[re_order_indices]
    sorted_coeff = coeff_vec[re_order_indices]
    # unique terms are those with non-zero entries in the adjacent row difference array
    diff_adjacent = np.diff(sorted_terms, axis=0)
    mask_unique_terms = np.append(True, np.any(diff_adjacent, axis=1))
    reduced_symp_matrix = sorted_terms[mask_unique_terms]
    # mask the term indices such that those which are skipped are summed under np.reduceat
    summing_indices = np.arange(symp_matrix.shape[0])[mask_unique_terms]
    reduced_coeff_vec = np.add.reduceat(sorted_coeff, summing_indices, axis=0)
    # if a zero threshold is specified terms with sufficiently small coefficient will be dropped
    if zero_threshold is not None:
        mask_nonzero = abs(reduced_coeff_vec)>zero_threshold
        reduced_symp_matrix = reduced_symp_matrix[mask_nonzero]
        reduced_coeff_vec = reduced_coeff_vec[mask_nonzero]

    return reduced_symp_matrix, reduced_coeff_vec

def random_symplectic_matrix(n_qubits,n_terms, diagonal=False):
    """ Generates a random binary matrix of dimension (n_terms) x (2*n_qubits)
    Specifying diagonal=True will set the left hand side (X_block) to all zeros
    """
    if diagonal:
        Z_block = np.random.randint(0,2,(n_terms, n_qubits))
        return np.hstack([np.zeros_like(Z_block), Z_block])
    else:
        return np.random.randint(0,2,(n_terms, 2*n_qubits)) 

def norm(vector: np.array) -> float:
    """
    Returns:
        l2-norm of input vector
    """
    return np.sqrt(np.dot(vector, vector.conjugate()))

def lp_norm(vector: np.array, p:int=2) -> float:
    """
    Returns:
        lp-norm of vector
    """
    return np.power(np.sum(np.power(np.abs(vector), p)), 1/p)

def ZX_calculus_reduction(qc: QuantumCircuit) -> QuantumCircuit:
    """ Simplify the circuit via ZX calculus using PyZX... 
    Only works on parametrized circuits!

    Returns:
        simplified_qc (QuantumCircuit): the reduced circuit via ZX calculus
    """
    # to perform ZX-calculus optimization
    qc_qasm = qc.qasm()
    qc_pyzx = zx.Circuit.from_qasm(qc_qasm)
    g = qc_pyzx.to_graph()
    zx.full_reduce(g) # simplifies the Graph in-place
    g.normalize()
    c_opt = zx.extract_circuit(g.copy())
    simplified_qc = QuantumCircuit.from_qasm_str(c_opt.to_qasm())

    return simplified_qc

def _rref_binary(matrix: np.array) -> np.array:
    """ Row-reduced echelon form over the binary field (GF2) - rows are not reordered 
    here for efficiency (not required in some use cases, e.g. symmetry identification)
    """
    rref_matrix = matrix.copy()
    # iterate over rows of array
    for i, row_i in enumerate(rref_matrix):
        # if not a row of zeros
        if np.any(row_i):
            # find the first non-zero entry of row i
            pivot = np.where(row_i)[0][0]
            # find the non-zero entries in column i
            update_set = np.setdiff1d(np.where(rref_matrix[:, pivot]), i)
            # XOR the rows containing non-zero entries in column i with row i
            rref_matrix[update_set] = np.bitwise_xor(rref_matrix[update_set], row_i)
            # the rows below i will now be zeroed out in the pivot column
    return rref_matrix

def rref_binary(matrix: np.array) -> np.array:
    """ Full row-reduced echelon form with row reordering
    """
    reduced = _rref_binary(matrix)
    row_order, col_order = zip(
        *sorted(
            [(i,np.where(row)[0][0]) for i,row in enumerate(reduced) if np.any(row)],
            key=lambda x:x[1]
        )
    )
    row_order = list(row_order) + list(set(range(reduced.shape[0])).difference(row_order))
    return reduced[row_order]

def _cref_binary(matrix: np.array) -> np.array:
    """ Column-reduced echelon form with static columns (used in symmetry identification)
    """
    return _rref_binary(matrix.T).T  

def cref_binary(matrix: np.array) -> np.array:
    """ Column-reduced echelon form with ordered columns (used in basis reconstruction)
    """
    return rref_binary(matrix.T).T         

def get_ground_state_sparse(sparse_matrix, initial_guess=None):
    """Compute lowest eigenvalue and eigenstate.
    Args:
        sparse_operator: Operator to find the ground state of.
        initial_guess (ndarray): Initial guess for ground state.  A good
            guess dramatically reduces the cost required to converge.
    Returns
    -------
        eigenvalue:
            The lowest eigenvalue, a float.
        eigenstate:
            The lowest eigenstate in scipy.sparse csc format.
    """
    values, vectors = sp.sparse.linalg.eigsh(sparse_matrix,
                                                k=1,
                                                v0=initial_guess,
                                                which='SA',
                                                maxiter=1e7)

    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    eigenvalue = values[0]
    eigenstate = vectors[:, 0]
    return eigenvalue, eigenstate.T

def exact_gs_energy(sparse_matrix, initial_guess=None) -> Tuple[float, np.array]:
    """ Return the ground state energy and corresponding 
    ground statevector for the input operator

    Full Fock space - can find ground state in wrong particle number subsapce. 
    Refer to chem_utils.exact_gs_state for a specified particle number.
    """

    if sparse_matrix.shape[0] > 2**5:
        ground_energy, ground_state = get_ground_state_sparse(sparse_matrix, initial_guess=initial_guess)
    else:
        dense_matrix = sparse_matrix.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_matrix)
        ground_energy, ground_state = sorted(zip(eigvals,eigvecs.T), key=lambda x:x[0])[0]

    return ground_energy, np.array(ground_state).reshape([-1,1])

def unit_n_sphere_cartesian_coords(angles: np.array) -> np.array:
    """ Input an array of angles of length n, returns the n+1 cartesian coordinates 
    of the corresponding unit n-sphere in (n+1)-dimensional Euclidean space.
    """
    cartesians = [np.prod(np.sin(angles[:i]))*np.cos(angles[i]) for i in range(len(angles))]
    cartesians.append(np.prod(np.sin(angles)))
    return np.array(cartesians)

def QubitOperator_to_dict(op, num_qubits):
    assert(type(op) == of.QubitOperator)
    op_dict = {}
    term_dict = op.terms
    terms = list(term_dict.keys())

    for t in terms:    
        letters = ['I' for i in range(num_qubits)]
        for i in t:
            letters[i[0]] = i[1]
        p_string = ''.join(letters)        
        op_dict[p_string] = term_dict[t]
         
    return op_dict