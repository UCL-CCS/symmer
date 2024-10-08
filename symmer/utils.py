from symmer.operators import PauliwordOp, QuantumState, AntiCommutingOp
import numpy as np
import scipy as sp
from typing import List, Tuple, Union
from functools import reduce
from scipy.sparse import csr_matrix
from scipy.sparse import kron as sparse_kron
from symmer.operators.utils import _rref_binary
# import ray
from ray import remote, get
import os
# from psutil import cpu_count

def exact_gs_energy(
        sparse_matrix, 
        initial_guess=None, 
        n_particles=None, 
        number_operator=None, 
        n_eigs=6
    ) -> Tuple[float, np.array]:
    """ 
    Return the ground state energy and corresponding ground statevector for the input operator
    
    Specifying a particle number will restrict to eigenvectors |ψ> such that <ψ|N_op|ψ> = n_particles
    where N_op is the given number operator.

    Args:
        sparse_matrix (csr_matrix): The sparse matrix for which we want to compute the eigenvalues and eigenvectors.
        initial_guess (array): The initial guess for the eigenvectors.
        n_particles (int):  Particle number to restrict eigenvectors |ψ> such that <ψ|N_op|ψ> = n_particles where N_op is the given number operator.
        number_operator (array): Number Operator to restrict eigenvectors |ψ> such that <ψ|N_op|ψ> = n_particles.
        n_eigs (int): The number of eigenvalues and eigenvectors to compute.

    Returns:
        evl(float): The ground state energy for the input operator
        QState(QuantumState): Ground statevector for the input operator corresponding to evl.
    """
    if number_operator is None:
        # if no number operator then need not compute any further eigenvalues
        n_eigs = 1

    # Note the eigenvectors are stored column-wise so need to transpose
    if sparse_matrix.shape[0] > 2**5:
        eigvals, eigvecs = sp.sparse.linalg.eigsh(
            sparse_matrix,k=n_eigs,v0=initial_guess,which='SA',maxiter=1e7
        )
    else:
        # for small matrices the dense representation can be more efficient than sparse!
        eigvals, eigvecs = np.linalg.eigh(sparse_matrix.toarray())
    
    # order the eigenvalues by increasing size
    order = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    if n_particles is None:
        # if no particle number is specified then return the smallest eigenvalue
        return eigvals[0], QuantumState.from_array(eigvecs[:,0].reshape([-1,1]))
    else:
        assert(number_operator is not None), 'Must specify the number operator.'
        # otherwise, search through the first n_eig eigenvalues and check the Hamming weight
        # of the the corresponding eigenvector - return the first match with n_particles
        for evl, evc in zip(eigvals, eigvecs.T):
            psi = QuantumState.from_array(evc.reshape([-1,1])).cleanup(zero_threshold=1e-5)
            assert(~np.any(number_operator.X_block)), 'Number operator not diagonal'
            expval_n_particle = 0
            for Z_symp, Z_coeff in zip(number_operator.Z_block, number_operator.coeff_vec):
                sign = (-1) ** np.einsum('ij->i', 
                    np.bitwise_and(
                        Z_symp, psi.state_matrix
                    )
                )
                expval_n_particle += Z_coeff * np.sum(sign * np.square(abs(psi.state_op.coeff_vec)))
            if np.round(expval_n_particle) == n_particles:
                return evl, QuantumState.from_array(evc.reshape([-1,1]))
        # if a solution is not found within the first n_eig eigenvalues then error
        raise RuntimeError('No eigenvector of the correct particle number was identified - try increasing n_eigs.')

def get_entanglement_entropy(psi: QuantumState, qubits: List[int]) -> float:
    """
    Get the Von Neumann entropy of the biprtition defined by the specified subsystem 
    qubit indices and those remaining (i.e. those that will be subsequently traced out)

    Args:
        psi (QuantumState): the quantum state for which we wish to extract the entanglement entropy
        qubits (List[int]): the qubit indices to project onto (the remaining qubits will be traced over)
    
    Returns:
        entropy (float): the Von Neumann entropy of the reduced subsystem
    """
    reduced = psi.get_rdm(qubits)
    eigvals, eigvecs = np.linalg.eig(reduced)
    eigvals = eigvals[eigvals>0]
    entropy = -np.sum(eigvals*np.log(eigvals)).real
    return entropy

def random_anitcomm_2n_1_PauliwordOp(n_qubits, complex_coeff=False, apply_clifford=True):
    """ 
    Generate a anticommuting PauliOperator of size 2n+1 on n qubits (max possible size)
    with normally distributed coefficients. Generates in structured way then uses Clifford rotation (default)
    to try and make more random (can stop this to allow FAST build, but inherenet structure
    will be present as operator is formed in specific way!)

    Args:
        n_qubits (int): Number of Qubits
        complex_coeff (bool): Boolean representing whether if we want complex coefficents or not.
        apply_clifford (bool): Boolean representing whether we want to apply clifford rotations to get rid of structure or not.

    Returns:
        P_anticomm (csr_matrix): Anticommuting PauliOperator of size 2n+1 on n qubits with normally distributed coefficients.
    """
    # base = 'X' * n_qubits
    # I_term = 'I' * n_qubits
    # P_list = [base]
    # for i in range(n_qubits):
    #     # Z_term
    #     P_list.append(base[:i] + 'Z' + I_term[i + 1:])
    #     # Y_term
    #     P_list.append(base[:i] + 'Y' + I_term[i + 1:])
    # coeff_vec = np.random.randn(len(P_list)).astype(complex)
    # if complex_coeff:
    #     coeff_vec += 1j * np.random.randn((len(P_list)))
    # P_anticomm = PauliwordOp.from_dictionary((dict(zip(P_list, coeff_vec))))

    Y_base = np.hstack((np.eye(n_qubits), np.tril(np.ones(n_qubits))))
    X_base = Y_base.copy()
    X_base[:, n_qubits:] = np.tril(np.ones(n_qubits), -1)

    ac_symp = np.vstack((Y_base, X_base))

    Z_symp = np.zeros(2 * n_qubits)
    Z_symp[n_qubits:] = np.ones(n_qubits)

    ac_symp = np.vstack((ac_symp, Z_symp)).astype(bool)

    coeff_vec = np.random.randn(ac_symp.shape[0]).astype(complex)
    if complex_coeff:
        coeff_vec += 1j * np.random.randn(2 * n_qubits + 1).astype(complex)

    P_anticomm = PauliwordOp(ac_symp, coeff_vec)

    if apply_clifford:
        # apply clifford rotations to get rid of structure
        U_cliff_rotations = []
        for _ in range(n_qubits * 5):
            P_rand = PauliwordOp.random(n_qubits, n_terms=1)
            P_rand.coeff_vec = [1]
            U_cliff_rotations.append((P_rand, np.random.choice([np.pi/2, -np.pi/2])))

        P_anticomm = P_anticomm.perform_rotations(U_cliff_rotations)

    assert P_anticomm.n_terms == 2 * n_qubits + 1

    ## expensive check
    # anti_comm_check = P_anticomm.adjacency_matrix.astype(int) - np.eye(P_anticomm.adjacency_matrix.shape[0])
    # assert(np.sum(anti_comm_check) == 0), 'operator needs to be made of anti-commuting Pauli operators'

    return P_anticomm


def tensor_list(factor_list:List[PauliwordOp]) -> PauliwordOp:
    """ 
    Given a list of PauliwordOps, recursively tensor from the right
    
    Args:
        factor_list (list): list of PauliwordOps
    
    Returns: 
        Tensor Product of items in factor_list from the right 
    """
    return reduce(lambda x,y:x.tensor(y), factor_list)


def product_list(product_list:List[PauliwordOp]) -> PauliwordOp:
    """ 
    Given a list of PauliwordOps, recursively take product from the right

    Args:
        product_list (list): list of PauliwordOps

    Returns:
        Product of items in product_list from the right 
    """
    return reduce(lambda x,y:x*y, product_list)


def gram_schmidt_from_quantum_state(state:Union[np.array, list, QuantumState]) ->np.array:
    """
    build a unitary to build a quantum state from the zero state (aka state defines first column of unitary)
    uses gram schmidt to find other (orthogonal) columns of matrix

    Args:
        state (np.array): 1D array of quantum state (size 2^N qubits)
    Returns:
        M (np.array): unitary matrix preparing input state from zero state
    """

    if isinstance(state, QuantumState):
        N_qubits = state.n_qubits
        state = state.to_sparse_matrix.toarray().reshape([-1])
    else:
        state = np.asarray(state).reshape([-1])
        N_qubits = round(np.log2(state.shape[0]))
        missing_amps = 2**N_qubits - state.shape[0]
        state = np.hstack((state, np.zeros(missing_amps, dtype=complex)))

    assert state.shape[0] == 2**N_qubits, 'state is not defined on power of two'
    assert np.isclose(np.linalg.norm(state), 1), 'state is not normalized'

    M = np.eye(2**N_qubits, dtype=complex)

    # reorder if state has 0 amp on zero index
    if np.isclose(state[0], 0):
        max_amp_ind = np.argmax(state)
        M[:, [0, max_amp_ind]] = M[:, [max_amp_ind,0]]

    # defines first column
    M[:, 0] = state
    for a in range(M.shape[0]):
        for b in range(a):
            M[:, a]-= (M[:, b].conj().T @ M[:, a]) * M[:, b]

        # normalize
        M[:, a] = M[:, a] / np.linalg.norm( M[:, a])

    return M

def get_sparse_matrix_large_pauliwordop(P_op: PauliwordOp) -> csr_matrix:
    """
    In order to build the sparse matrix (e.g. above 18 qubits), this function goes through each pauli term
    divides into two equally sized tensor products finds the sparse matrix of those and then does a sparse
    kron product to get the large matrix.

    TODO:  Could also add how many chunks to split problem into (e.g. three/four/... tensor products).

    Args:
        P_op (PauliwordOp): Pauli operator to convert into sparse matrix
    Returns:
        mat (csr_matrix): sparse matrix of P_op
    """
    nq = P_op.n_qubits
    if nq<16:
        mat = P_op.to_sparse_matrix
    else:
        # n_cpus = mp.cpu_count()
        # P_op_chunks_inds = np.rint(np.linspace(0, P_op.n_terms, min(n_cpus, P_op.n_terms))).astype(set).astype(int)
        #
        # # miss zero index out (as emtpy list)
        # P_op_chunks = [P_op[P_op_chunks_inds[ind_i]: P_op_chunks_inds[ind_i+1]] for ind_i, _ in enumerate(P_op_chunks_inds[1:])]
        # with mp.Pool(n_cpus) as pool:
        #     tracker = pool.map(_get_sparse_matrix_large_pauliwordop, P_op_chunks)

        # plus one below due to indexing (actual number of chunks ignores this value)
        n_chunks = os.cpu_count()
        if (n_chunks<=1) or (P_op.n_terms<=1):
            # no multiprocessing possible
            mat = get(_get_sparse_matrix_large_pauliwordop.remote(P_op))
        else:
            # plus one below due to indexing (actual number of chunks ignores this value)
            n_chunks += 1
            P_op_chunks_inds = np.rint(np.linspace(0, P_op.n_terms, min(n_chunks, P_op.n_terms+1))).astype(set).astype(int)
            P_op_chunks = [P_op[P_op_chunks_inds[ind_i]: P_op_chunks_inds[ind_i + 1]] for ind_i, _ in
                           enumerate(P_op_chunks_inds[1:])]
            tracker = np.array(get(
                [_get_sparse_matrix_large_pauliwordop.remote(op) for op in P_op_chunks]))
            mat = reduce(lambda x, y: x + y, tracker)

    return mat

@remote(num_cpus=os.cpu_count(),
            runtime_env={
                "env_vars": {
                    "NUMBA_NUM_THREADS": os.getenv("NUMBA_NUM_THREADS"),
                    # "OMP_NUM_THREADS": str(os.cpu_count()),
                    "OMP_NUM_THREADS": os.getenv("NUMBA_NUM_THREADS"),
                    "NUMEXPR_MAX_THREADS": str(os.cpu_count())
                }
            }
            )
def _get_sparse_matrix_large_pauliwordop(P_op: PauliwordOp) -> csr_matrix:
    """
    """
    nq = P_op.n_qubits
    mat = csr_matrix(([], ([],[])), shape=(2**nq,2**nq))
    for op in P_op:
        left_tensor = np.hstack((op.X_block[:, :nq // 2],
                                 op.Z_block[:, :nq // 2]))
        left_coeff = op.coeff_vec

        right_tensor = np.hstack((op.X_block[:, nq // 2:],
                                  op.Z_block[:, nq // 2:]))
        right_coeff = np.array([1])

        mat += sparse_kron(PauliwordOp(left_tensor, left_coeff).to_sparse_matrix,
                           PauliwordOp(right_tensor, right_coeff).to_sparse_matrix,
                           format='csr')  # setting format makes this faster!

    return mat


def matrix_allclose(A: Union[csr_matrix, np.array], B:Union[csr_matrix, np.array], tol:int = 1e-15) -> bool:
    """
    check matrix A and B have the same entries up to a given tolerance
    Args:
        A : matrix A
        B:  matrix B
        tol: allowed difference

    Returns:
        bool

    """
    if isinstance(A, csr_matrix) and isinstance(B, csr_matrix):
        max_diff = np.abs(A-B).max()
        return max_diff <= tol
    else:
        if isinstance(A, csr_matrix):
            A = A.toarray()

        if isinstance(B, csr_matrix):
            B = B.toarray()

        return np.allclose(A, B, atol=tol)


def get_PauliwordOp_root(power: int, pauli: PauliwordOp) -> PauliwordOp:
    """
    Get arbitrary power of a single Pauli operator. See eq1 in https://arxiv.org/pdf/2012.01667.pdf

    Log(A) in paper given by = 1j*pi*(I-P)/2 here

    P^{k} = e^{k i pi Q}

    Q = (I-P)/2, where P in {X,Y,Z}

    e^{k i pi (I-P)/2} = e^{k i pi/2 I} * e^{ - k i pi/2 P} <- expand product!

    Args:
        power (int): power to take
        pauli (PauliwordOp): Pauli operator to take power of
    Returns:
        Pk (PauliwordOp): Pauli operator that is power of input

    """
    assert pauli.n_terms == 1, 'can only take power of single operators'

    I_term = PauliwordOp.from_list(['I' * pauli.n_qubits])

    cos_term = np.cos(power * np.pi / 2)
    sin_term = np.sin(power * np.pi / 2)

    Pk = (I_term.multiply_by_constant(cos_term ** 2 + 1j * cos_term * sin_term) +
          pauli.multiply_by_constant(-1j * cos_term * sin_term + sin_term ** 2))

    return Pk


def Get_AC_root(power: float, operator: AntiCommutingOp) -> PauliwordOp:
    """
    Get arbitrary power of an anticommuting Pauli operator.

    ** test **
    from symmer.operators import AntiCommutingOp
    from symmer.utils import random_anitcomm_2n_1_PauliwordOp, Get_AC_root

    op = random_anitcomm_2n_1_PauliwordOp(3)
    AC = AntiCommutingOp.from_PauliwordOp(op)

    p = 0.25
    root = Get_AC_root(p, AC)
    print((root*root*root*root - AC).cleanup(zero_threshold=1e-12)

    Args:
        power (float): any power
        operator (AntiCommutingOp) Anticommuting Pauli operator

    Returns:
        AC_root (PauliwordOp): operator representing power of AC input

    """
    Ps, rot, gamma_l, AC_normed = operator.unitary_partitioning(up_method='LCU')

    Ps_root = get_PauliwordOp_root(power, Ps)

    AC_root = (rot.dagger * Ps_root * rot).multiply_by_constant(gamma_l ** power)

    return AC_root