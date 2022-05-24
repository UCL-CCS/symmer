import numpy as np
import scipy as sp
from typing import Tuple
from qiskit import QuantumCircuit
import pyzx as zx
import openfermion as of

def symplectic_cleanup(symp_matrix, coeff_vec):
    """ Remove duplicated rows of symplectic matrix terms, whilst summing
    the corresponding coefficients of the deleted rows in coeff_vec
    """
    n_terms = symp_matrix.shape[0]
    # order lexicographically
    term_ordering = np.lexsort(symp_matrix.T)
    sorted_terms = symp_matrix[term_ordering]
    sorted_coeff = coeff_vec[term_ordering]
    # unique terms are those with non-zero entries in the adjacent row difference array
    diff_adjacent = np.diff(sorted_terms, axis=0)
    mask_unique_terms = np.array([True]+np.any(diff_adjacent, axis=1).tolist()) #faster than np.append!
    reduced_symp_matrix = sorted_terms[mask_unique_terms]
    # mask the term indices such that those which are skipped are summed under np.reduceat
    summing_indices = np.arange(n_terms)[mask_unique_terms]
    reduced_coeff_vec = np.add.reduceat(sorted_coeff, summing_indices, axis=0)
    
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

def gf2_gaus_elim(gf2_matrix: np.array) -> np.array:
    """
    Function that performs Gaussian elimination over GF2(2)
    GF is the initialism of Galois field, another name for finite fields.

    GF(2) may be identified with the two possible values of a bit and to the boolean values true and false.

    pseudocode: http://dde.binghamton.edu/filler/mct/hw/1/assignment.pdf

    Args:
        gf2_matrix (np.array): GF(2) binary matrix to preform Gaussian elimination over
    Returns:
        gf2_matrix_rref (np.array): reduced row echelon form of M
    """
    gf2_matrix_rref = gf2_matrix.copy()
    m_rows, n_cols = gf2_matrix_rref.shape

    row_i = 0
    col_j = 0

    while row_i < m_rows and col_j < n_cols:

        if sum(gf2_matrix_rref[row_i:, col_j]) == 0:
            # case when col_j all zeros
            # No pivot in this column, pass to next column
            col_j += 1
            continue

        # find index of row with first "1" in the vector defined by column j (note previous if statement removes all zero column)
        k = np.argmax(gf2_matrix_rref[row_i:, col_j]) + row_i
        # + row_i gives correct index (as we start search from row_i!)

        # swap row k and row_i (row_i now has 1 at top of column j... aka: gf2_matrix_rref[row_i, col_j]==1)
        gf2_matrix_rref[[k, row_i]] = gf2_matrix_rref[[row_i, k]]
        # next need to zero out all other ones present in column j (apart from on the i_row!)
        # to do this use row_i and use modulo addition to zero other columns!

        # make a copy of j_th column of gf2_matrix_rref, this includes all rows (0 -> M)
        Om_j = np.copy(gf2_matrix_rref[:, col_j])

        # zero out the i^th position of vector Om_j (this is why copy needed... to stop it affecting gf2_matrix_rref)
        Om_j[row_i] = 0
        # note this was orginally 1 by definition...
        # This vector now defines the indices of the rows we need to zero out
        # by setting ith position to zero - it stops the next steps zeroing out the i^th row (which we need as our pivot)


        # next from row_i of rref matrix take all columns from j->n (j to last column)
        # this is vector of zero and ones from row_i of gf2_matrix_rref
        i_jn = gf2_matrix_rref[row_i, col_j:]
        # we use i_jn to zero out the rows in gf2_matrix_rref[:, col_j:] that have leading one (apart from row_i!)
        # which rows are these? They are defined by that Om_j vector!

        # the matrix to zero out these rows is simply defined by the outer product of Om_j and i_jn
        # this creates a matrix of rows of i_jn terms where Om_j=1 otherwise rows of zeros (where Om_j=0)
        Om_j_dependent_rows_flip = np.einsum('i,j->ij', Om_j, i_jn, optimize=True)
        # note flip matrix is contains all m rows ,but only j->n columns!

        # perfrom bitwise xor of flip matrix to zero out rows in col_j that that contain a leading '1' (apart from row i)
        gf2_matrix_rref[:, col_j:] = np.bitwise_xor(gf2_matrix_rref[:, col_j:], Om_j_dependent_rows_flip)

        row_i += 1
        col_j += 1

    return gf2_matrix_rref


def gf2_basis_for_gf2_rref(gf2_matrix_in_rreform: np.array) -> np.array:
    """
    Function that gets the kernel over GF2(2) of ow reduced  gf2 matrix!

    uses method in: https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Basis

    Args:
        gf2_matrix_in_rreform (np.array): GF(2) matrix in row reduced form
    Returns:
        basis (np.array): basis for gf2 input matrix that was in row reduced form
    """
    rows_to_columns = gf2_matrix_in_rreform.T
    eye = np.eye(gf2_matrix_in_rreform.shape[1], dtype=int)

    # do column reduced form as row reduced form
    rrf = gf2_gaus_elim(np.hstack((rows_to_columns, eye.T)))

    zero_rrf = np.where(~rrf[:, :gf2_matrix_in_rreform.shape[0]].any(axis=1))[0]
    basis = rrf[zero_rrf, gf2_matrix_in_rreform.shape[0]:]

    return basis              

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
    """

    if sparse_matrix.shape[0] > 2**5:
        ground_energy, ground_state = get_ground_state_sparse(sparse_matrix, initial_guess=initial_guess)
    else:
        dense_matrix = sparse_matrix.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_matrix)
        ground_energy, ground_state = sorted(zip(eigvals,eigvecs.T), key=lambda x:x[0])[0]

    return ground_energy, np.array(ground_state)

def unit_n_sphere_cartesian_coords(angles: np.array) -> np.array:
    """ Input an array of angles of length n, returns the n+1 cartesian coordinates 
    of the corresponding unit n-sphere in (n+1)-dimensional Euclidean space.
    """
    cartesians = [np.prod(np.sin(angles[:i]))*np.cos(angles[i]) for i in range(len(angles))]
    cartesians.append(np.prod(np.sin(angles)))
    return np.array(cartesians)
    
# comment out due to incompatible versions of Cirq and OpenFermion in Orquestra
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