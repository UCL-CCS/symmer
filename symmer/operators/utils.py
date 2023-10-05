import numpy as np
import scipy as sp
from typing import Tuple, Dict
from openfermion import QubitOperator
from qiskit.opflow import PauliSumOp


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

    Y_loc = np.logical_and(X_block, Z_block)
    X_loc = np.logical_xor(Y_loc, X_block)
    Z_loc = np.logical_xor(Y_loc, Z_block)

    char_aray = np.array(list("I" * n_qubits), dtype=str)

    char_aray[Y_loc] = "Y"
    char_aray[X_loc] = "X"
    char_aray[Z_loc] = "Z"

    Pword_string = "".join(char_aray)

    return Pword_string


def string_to_symplectic(pauli_str, n_qubits):
    """
    Args:
        pauli_str (str): Pauli String to be converted to symplectic array
        n_qubits (int): Number of qubits

    Returns:
        symp_vec (array): symplectic Pauliword array
    """
    assert (
        len(pauli_str) == n_qubits
    ), "Number of qubits is incompatible with pauli string"
    assert set(pauli_str).issubset(
        {"I", "X", "Y", "Z"}
    ), "pauliword must only contain X,Y,Z,I terms"

    char_aray = np.array(list(pauli_str), dtype=str)
    X_loc = char_aray == "X"
    Z_loc = char_aray == "Z"
    Y_loc = char_aray == "Y"

    symp_vec = np.zeros(2 * n_qubits, dtype=int)
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

    Args:
        i (int):  Integer

    Returns:
        Number of "1" bits in the bitstring of integer 'i'.
    """
    i = i - ((i >> 1) & 0x55555555)  # add pairs of bits
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)  # quads
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xFFFFFFFF) >> 24


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
    vals = global_phase * (-1) ** (count1_in_int_bitstring(row_inds_and_Zint) % 2)

    sparse_matrix = sp.sparse.csr_matrix(
        (vals, (row_ind, col_ind)), shape=(dimension, dimension), dtype=complex
    )

    return coeff * sparse_matrix


def symplectic_cleanup(
    symp_matrix: np.array, coeff_vec: np.array, zero_threshold: float = None
) -> Tuple[np.array, np.array]:
    """
    Remove duplicated rows of symplectic matrix terms, whilst summing
    the corresponding coefficients of the deleted rows in coeff_vec

    Args:
        symp_matrix (np.array): Symplectic matrix.
        coeff_vec (np.array): Coefficient Vector.
        zero_threshold (float): Zero Threshold Value. By default it is set to 'None'.

    Returns:
        Reduced symplectic matrix and reduced coefficient vector.
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
        mask_nonzero = abs(reduced_coeff_vec) > zero_threshold
        reduced_symp_matrix = reduced_symp_matrix[mask_nonzero]
        reduced_coeff_vec = reduced_coeff_vec[mask_nonzero]

    return reduced_symp_matrix, reduced_coeff_vec


def random_symplectic_matrix(n_qubits, n_terms, diagonal=False, density=0.3):
    """
    Generates a random binary matrix of dimension (n_terms) x (2*n_qubits)
    Specifying diagonal=True will set the left hand side (X_block) to all zeros
    """
    if diagonal:
        Z_block = np.random.choice(
            [True, False], size=[n_terms, n_qubits], p=[density / 2, 1 - density / 2]
        )
        return np.hstack([np.zeros_like(Z_block), Z_block])
    else:
        return np.random.choice(
            [True, False], size=[n_terms, 2 * n_qubits], p=[density, 1 - density]
        )


def _rref_binary(matrix: np.array) -> np.array:
    """
    Row-reduced echelon form over the binary field (GF2) - rows are not reordered
    here for efficiency (not required in some use cases, e.g. symmetry identification).

    Args:
        matrix(np.array): Matrix whose row-reduced echelon form over the binary field has to be found.

    Returns:
        Row-reduced echelon form over the binary field of input matrix.
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
    """
    Full row-reduced echelon form with row reordering.

    Args:
        matrix(np.array): Matrix whose row-reduced echelon form with row reordering has to be found.

    Returns:
        Row-reduced echelon form of input matrix with row reordering of input matrix.
    """
    reduced = _rref_binary(matrix)
    row_order, col_order = zip(
        *sorted(
            [(i, np.where(row)[0][0]) for i, row in enumerate(reduced) if np.any(row)],
            key=lambda x: x[1],
        )
    )
    row_order = list(row_order) + list(
        set(range(reduced.shape[0])).difference(row_order)
    )
    return reduced[row_order]


def _cref_binary(matrix: np.array) -> np.array:
    """
    Column-reduced echelon form with static columns (used in symmetry identification).

    Args:
        matrix(np.array): Matrix whose column-reduced echelon form with static columns has to be found.

    Returns:
        Column-reduced echelon form of input matrix with static columns.
    """
    return _rref_binary(matrix.T).T


def cref_binary(matrix: np.array) -> np.array:
    """
    Column-reduced echelon form with ordered columns (used in basis reconstruction).

    Args:
        matrix(np.array): Matrix whose column-reduced echelon form with ordered columns has to be found.

    Returns:
        Column-reduced echelon form of input matrix with ordered columns.
    """
    return rref_binary(matrix.T).T


def QubitOperator_to_dict(op: QubitOperator, num_qubits: int):
    """
    OpenFermion

    Args:
        op (QubitOperator): Qubit Operator
        num_qubits (int) : Number of qubiits.

    Returns:
        Dictionary format of Qubit Operator.
    """
    assert type(op) == QubitOperator
    op_dict = {}
    term_dict = op.terms
    terms = list(term_dict.keys())

    for t in terms:
        letters = ["I" for i in range(num_qubits)]
        for i in t:
            letters[i[0]] = i[1]
        p_string = "".join(letters)
        op_dict[p_string] = term_dict[t]

    return op_dict


def PauliSumOp_to_dict(op: PauliSumOp) -> dict:
    """
    Qiskit

    Args:
        op (PauliSumOp): Pauli Sum Operator

    Returns:
        Dictionary format of Pauli Sum Operator.
    """
    H_dict = {}
    for P_term in op.to_pauli_op():
        Pstr = P_term.primitive.to_label()
        H_dict[Pstr] = P_term._coeff
    return H_dict


def safe_PauliwordOp_to_dict(op) -> Dict[str, Tuple[float, float]]:
    """
    Stores the real and imaginary parts of the coefficient separately in a tuple.

    Args:
        op (PauliwordOp): Weighted linear combination of N-fold Pauli operators.
    Returns:
        dict_out (dict): Dictionary of the form {pstring:(real, imag)}.
    """
    terms, coeffs = zip(*op.to_dictionary.items())
    coeffs = [(n.real, n.imag) for n in coeffs]
    dict_out = dict(zip(terms, coeffs))
    return dict_out


def safe_QuantumState_to_dict(psi) -> Dict[str, Tuple[float, float]]:
    """
    Stores the real and imaginary parts of the coefficient separately in a tuple.

    Args:
        op (QuantumState): Weighted linear combination of N-fold Pauli operators.
    Returns:
        dict_out (dict): Dictionary of the form {pstring:(real, imag)}.
    """
    terms, coeffs = zip(*psi.to_dictionary.items())
    coeffs = [(n.real, n.imag) for n in coeffs]
    dict_out = dict(zip(terms, coeffs))
    return dict_out


def mul_symplectic(
    symp_vec1: np.array, coeff1: complex, symp_vec2: np.array, coeff2: complex
) -> Tuple[np.array, complex, int]:
    """
    Performs Pauli multiplication with phases at the level of the symplectic
    vector (1D here!). The phase compensation is implemented as per https://doi.org/10.1103/PhysRevA.68.042318.

    P1 * P2 is performed

    Args:
        symp_vec1 (np.array) : 1D vector of left Pauli operator
        coeff1 (float): coefficient of left  Pauli operator

        symp_vec2 (np.array) : 1D vector of right Pauli operator
        coeff2 (float): coefficient of right  Pauli operator

    Returns:
        output_symplectic_vec (np.array): binary symplectic output (Pauli opertor out)
        coeff_vec (complex): complex coeff with correct phase
    """
    X_block1, Z_block1 = np.split(symp_vec1, 2)
    X_block2, Z_block2 = np.split(symp_vec2, 2)

    # number of Y terms in left Pauli operator
    Y_count1 = np.sum(np.bitwise_and(X_block1, Z_block1), axis=0)
    # number of Y terms in right Pauli operator
    Y_count2 = np.sum(np.bitwise_and(X_block2, Z_block2), axis=0)

    # phaseless multiplication is binary addition in symplectic representation
    output_symplectic_vec = np.bitwise_xor(symp_vec1, symp_vec2)
    # phase is determined by Y counts plus additional sign flip
    Y_count_out = np.sum(
        np.bitwise_and(*np.split(output_symplectic_vec, 2)), axis=0
    )  # number of Y terms in output
    # X_block of first op and Z_block of second op
    sign_change = (-1) ** (
        np.sum(np.bitwise_and(X_block1, Z_block2), axis=0) % 2
    )  # mod 2 as only care about parity
    # final phase modification
    phase_mod = sign_change * (1j) ** (
        (3 * (Y_count1 + Y_count2) + Y_count_out) % 4
    )  # mod 4 as roots of unity
    coeff_vec = phase_mod * coeff1 * coeff2
    return output_symplectic_vec, coeff_vec  # , Y_count_out


def unit_n_sphere_cartesian_coords(angles: np.array) -> np.array:
    """
    Input an array of angles of length n, returns the n+1 cartesian coordinates
    of the corresponding unit n-sphere in (n+1)-dimensional Euclidean space.

    Args:
        angles (np.array): Array of angles of length n of the corresponding unit n-sphere in (n+1)-dimensional Euclidean space.

    Returns:
        Numpy Array of n+1 cartesian coordinates.
    """
    cartesians = [
        np.prod(np.sin(angles[:i])) * np.cos(angles[i]) for i in range(len(angles))
    ]
    cartesians.append(np.prod(np.sin(angles)))
    return np.array(cartesians)


def binomial_coefficient(n, k):
    """
    Calculate the binomial coefficient "n choose k" or denoted as "C(n, k)," represents the number of ways to choose k objects from a set of n objects without considering their order.
    Differs from np.math.comb as this allows non-integer n.

    Args:
        n: Total number of objects.
        k: Number of object to choose from a set of n objects.

    Returns:
        Binomial coefficient "n choose k".
    """
    prod = 1
    for r in range(k):
        prod *= (n - r) / (k - r)
    return prod


def check_independent(operators):
    """
    Check if the input PauliwordOp contains algebraically dependent terms.

    Args:
        operators (PauliwordOp): Operators.

    Returns:
        Returns True, if input operators contains algebraically dependent terms.
    """
    check_independent = _rref_binary(operators.symp_matrix)
    return ~np.any(np.all(~check_independent, axis=1))


def check_jordan_independent(operators):
    """
    Check if the input PauliwordOp contains algebraically dependent terms under jordan product
    (note input can be noncontextual, but contain dependent terms!)

    Args:
        operators (PauliwordOp): Operators.

    Returns:
        Returns True, if input operators contains algebraically dependent terms.


    test with : H ={
                     'IIIZ': (1+0j),
                     'IIZI': (1+0j),
                     'ZIII': (1+0j),
                     'IXII': (1+0j),
                     'XIIX': (1+0j)
                     }
    # clique =  [IIIZ, XIIX]
    # Z2     =  [IIZI, ZIIZ, IXII]

    """
    # mask_symmetries = np.all(operators.adjacency_matrix, axis=1)
    # Symmetries = operators[mask_symmetries]
    # Anticommuting = operators[~mask_symmetries]
    # return (
    #     check_independent(Symmetries) &
    #     np.all(Anticommuting.adjacency_matrix == np.eye(Anticommuting.n_terms))
    # )

    # if operators.n_terms<= 2*operators.n_qubits+1:
    #     # check if input is fully anticommuting
    #     adj_mat = operators.adjacency_matrix
    #     adj_mat[np.diag_indices_from(adj_mat)] = False
    #     if np.all(~adj_mat):
    #         return True

    # Z2_gen_mask = np.sum(operators.commutes_termwise(operators), axis=1) == operators.n_terms
    # Z2_terms = operators[Z2_gen_mask]
    #
    # # find terms not generated these Z2 symmerties
    # _, z2_mask = operators.generator_reconstruction(Z2_terms, override_independence_check=True)
    #
    # ac_terms = operators[~z2_mask]
    # if ac_terms.n_terms > 0:
    #     decomposed = ac_terms.clique_cover(edge_relation='C')
    #     clique_rep_list = [C.sort()[0] for C in decomposed.values()]
    #     ac_op = sum(clique_rep_list).cleanup()
    #     # if ac_op.n_terms>0:
    #     #     assert (np.sum(ac_op.adjacency_matrix.astype(int)
    #     #                    - np.eye(ac_op.adjacency_matrix.shape[0])) == 0), f'ac_op is not anticommuting: {ac_op}'
    #     # del ac_op
    #     if ac_op.n_terms > 0:
    #         # check all operators anticommute
    #         adj_mat = ac_op.adjacency_matrix
    #         adj_mat[np.diag_indices_from(adj_mat)] = False
    #         if not np.all(~adj_mat):
    #             return False
    #
    #     Z2_from_cliques = sum((decomposed[n] - C_rep) * C_rep for n, C_rep in enumerate(clique_rep_list) if
    #                           decomposed[n].n_terms > 1)
    #     if Z2_from_cliques:
    #         Z2_terms += Z2_from_cliques
    #
    #     if not np.all(Z2_terms.commutes_termwise(ac_op)):
    #         return False
    #
    # return check_independent(Z2_terms)

    # get fully commuting terms
    z2_mask = (
        np.sum(operators.commutes_termwise(operators), axis=1) == operators.n_terms
    )
    Z2_symm = operators[z2_mask]

    _, missing_mask = operators.generator_reconstruction(
        Z2_symm, override_independence_check=True
    )
    remaining = operators[~missing_mask]

    if remaining.n_terms > 0:
        # for remaining to be jordan independent remaining must be disjoint union of cliques
        # we can test for this below

        adj_matrix_view = np.ascontiguousarray(remaining.adjacency_matrix).view(
            np.dtype(
                (
                    np.void,
                    remaining.adjacency_matrix.dtype.itemsize
                    * remaining.adjacency_matrix.shape[1],
                )
            )
        )
        re_order_indices = np.argsort(adj_matrix_view.ravel())
        # sort the adj matrix and vector of coefficients accordingly
        sorted_terms = remaining.adjacency_matrix[re_order_indices]
        # unique terms are those with non-zero entries in the adjacent row difference array
        diff_adjacent = np.diff(sorted_terms, axis=0)
        mask_unique_terms = np.append(True, np.any(diff_adjacent, axis=1))
        clique_mask = sorted_terms[mask_unique_terms]

        # check for overlap (array of ones == no overlap)
        return np.all(np.sum(clique_mask, axis=0) == 1)
    else:
        # operators is made up of pairwise commuting ops and thus must be jordan independent
        return True


def check_adjmat_noncontextual(adjmat) -> bool:
    """
    Check whether the input boolean square matrix has a noncontextual structure...
    ... see https://doi.org/10.1103/PhysRevLett.123.200501 for details.

    Args:
        adjmat: Boolean square matrix.

    Returns:
        Returns True, if input matrix has a noncontextual structure.
    """
    # mask the terms that do not commute universally amongst the operator
    mask_non_universal = np.where(~np.all(adjmat, axis=1))[0]
    # look only at the unique rows in the masked adjacency matrix -
    # identical rows correspond with operators of the same clique
    unique_commutation_character = np.unique(
        adjmat[mask_non_universal, :][:, mask_non_universal], axis=0
    )
    # if the unique commutation characteristics are disjoint, i.e. no overlapping ones
    # between rows, the operator is noncontextual - hence we sum over rows and check
    # the resulting vector consists of all ones.
    return np.all(np.count_nonzero(unique_commutation_character, axis=0) == 1)


def perform_noncontextual_sweep(operator) -> "PauliwordOp":
    """
    Given an ordered operator, sweep over its terms once in order keeping terms that are noncontextual

    Args:
        operator (PauliwordOp): Ordered operator.

    Returns:
        List of terms maintaining noncontextuality.
    """

    # # initialize noncontextual operator with first element of input operator
    # noncon_indices = np.array([0])
    # adjmat = np.array([[True]], dtype=bool)
    # for index, term in enumerate(operator[1:]):
    #     # pad the adjacency matrix term-by-term - avoids full construction each time
    #     adjmat_vector = np.append(term.commutes_termwise(operator[noncon_indices]), True)
    #     adjmat_padded = np.pad(adjmat, pad_width=((0, 1), (0, 1)), mode='constant')
    #     adjmat_padded[-1,:] = adjmat_vector; adjmat_padded[:,-1] = adjmat_vector
    #     # check whether the adjacency matrix has a noncontextual structure
    #     if check_adjmat_noncontextual(adjmat_padded):
    #         noncon_indices = np.append(noncon_indices, index+1)
    #         adjmat = adjmat_padded

    # return operator[noncon_indices]

    ## new method uses generators to speed up sweep
    from symmer.operators import PauliwordOp

    mask = np.zeros(operator.n_terms, dtype=bool)

    for ind in range(operator.n_terms):
        mask[ind] = ~mask[ind]
        running_op = PauliwordOp(operator.symp_matrix[mask], np.ones(np.sum(mask)))

        if not running_op.is_noncontextual:
            mask[ind] = ~mask[ind]

    return operator[mask]


def binary_array_to_int(bin_arr):
    """
    Function to convert an array composed of rows of binary into integers.

    Args:
        bin_arr(np.array): 2D numpy array of binary.
    Returns:
        int_arr (np.array): 1D numpy array of ints.
    """

    if bin_arr.shape[1] < 64:
        b2i = 2 ** np.arange(bin_arr.shape[1] - 1, -1, -1)
    else:
        b2i = 2 ** np.arange(bin_arr.shape[1] - 1, -1, -1, dtype=float)
        # b2i = 2 ** np.arange(bin_arr.shape[1] - 1, -1, -1, dtype=object)

    int_arr = np.einsum("j, ij->i", b2i, bin_arr)
    # int_arr = (b2i*bin_arr).sum(axis=1)

    ## slower as does matrix product rather than multiplication along rows followed by a sum!
    # int_arr = bin_arr @ b2i

    return int_arr
