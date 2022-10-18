from symmer.symplectic import PauliwordOp, QuantumState
import numpy as np
import scipy as sp
from typing import Union, List, Tuple

def exact_gs_energy(
        sparse_matrix, 
        initial_guess=None, 
        n_particles=None, 
        number_operator=None, 
        n_eigs=6
    ) -> Tuple[float, np.array]:
    """ Return the ground state energy and corresponding ground statevector for the input operator
    
    Specifying a particle number will restrict to eigenvectors |ψ> such that <ψ|N_op|ψ> = n_particles
    where N_op is the given number operator.
    """
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
        return eigvals[0], eigvecs[:,0].reshape([-1,1])
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
            if round(expval_n_particle) == n_particles:
                return evl, evc.reshape([-1,1])
        # if a solution is not found within the first n_eig eigenvalues then error
        raise RuntimeError('No eigenvector of the correct particle number was identified - try increasing n_eigs.')


def random_anitcomm_2n_1_PauliwordOp(n_qubits, complex_coeff=True, apply_clifford=True):
    """ Generate a anticommuting PauliOperator of size 2n+1 on n qubits (max possible size)
        with normally distributed coefficients. Generates in structured way then uses Clifford rotation (default)
        to try and make more random (can stop this to allow FAST build, but inherenet structure
         will be present as operator is formed in specific way!)
    """
    base = 'X' * n_qubits
    I_term = 'I' * n_qubits

    P_list = [base]
    for i in range(n_qubits):
        # Z_term
        P_list.append(base[:i] + 'Z' + I_term[i + 1:])
        # Y_term
        P_list.append(base[:i] + 'Y' + I_term[i + 1:])

    coeff_vec = np.random.randn(len(P_list)).astype(complex)
    if complex_coeff:
        coeff_vec += 1j * np.random.randn((len(P_list)))

    P_anticomm = PauliwordOp.from_dictionary((dict(zip(P_list, coeff_vec))))

    # random rotations to get rid of structure
    if apply_clifford:
        for _ in range(10):
            P_rand = PauliwordOp.random(n_qubits, 1, complex_coeffs=complex_coeff)
            P_rand.coeff_vec[0] = 1
            P_anticomm = P_anticomm._rotate_by_single_Pword(P_rand,
                                                            None)

    anti_comm_check = P_anticomm.adjacency_matrix.astype(int) - np.eye(P_anticomm.adjacency_matrix.shape[0])
    assert(np.sum(anti_comm_check) == 0), 'operator needs to be made of anti-commuting Pauli operators'

    return P_anticomm


def get_PauliwordOp_projector(projector: Union[str, List[str], np.array]) -> "PauliwordOp":
    """
    Build PauliwordOp projector onto different qubit states. Using I to leave state unchanged and 0,1,+,-,*,% to fix
    qubit.

    key:
        I leaves qubit unchanged
        0,1 fixes qubit as |0>, |1> (Z basis)
        +,- fixes qubit as |+>, |-> (X basis)
        *,% fixes qubit as |i+>, |i-> (Y basis)

    e.g.
     'I+0*1II' defines the projector the state I ⊗ [ |+ 0 i+ 1>  <+ 0 i+ 1| ]  ⊗ II

    TODO: could be used to develop a control version of PauliWordOp

    Args:
        projector (str, list) : either string or list of strings defininng projector

    Returns:
        projector (PauliwordOp): operator that performs projection
    """
    if isinstance(projector, str):
        projector = np.array(list(projector))
    else:
        projector = np.asarray(projector)
    basis_dict = {'I':1,
                  '0':1, '1':-1,
                  '+':1, '-':-1,
                  '*':1, '%':-1}
    assert len(projector.shape) == 1, 'projector can only be defined over a single string or single list of strings (each a single letter)'
    assert set(projector).issubset(list(basis_dict.keys())), 'unknown qubit state (must be I,X,Y,Z basis)'


    N_qubits = len(projector)
    qubit_inds_to_fix = np.where(projector!='I')[0]
    N_qubits_fixed = len(qubit_inds_to_fix)
    state_sign = np.array([basis_dict[projector[active_ind]] for active_ind in qubit_inds_to_fix])

    if N_qubits_fixed < 64:
        binary_vec = (((np.arange(2 ** N_qubits_fixed).reshape([-1, 1]) & (1 << np.arange(N_qubits_fixed))[
                                                                          ::-1])) > 0).astype(int)
    else:
        binary_vec = (((np.arange(2 ** N_qubits_fixed, dtype=object).reshape([-1, 1]) & (1 << np.arange(N_qubits_fixed,
                                                                                                        dtype=object))[
                                                                                        ::-1])) > 0).astype(int)

    # assign a sign only to 'active positions' (0 in binary not relevent)
    sign_from_binary = binary_vec * state_sign

    # need to turn 0s in matrix to 1s before taking product across rows
    sign_from_binary = sign_from_binary + (sign_from_binary + 1) % 2

    sign = np.product(sign_from_binary, axis=1)

    coeff = 1 / 2 ** (N_qubits_fixed) * np.ones(2 ** N_qubits_fixed)
    sym_arr = np.zeros((coeff.shape[0], 2 * N_qubits))

    # assumed in Z basis
    sym_arr[:, qubit_inds_to_fix + N_qubits] = binary_vec
    sym_arr = sym_arr.astype(bool)

    ### fix for Y and X basis

    X_inds_fixed = np.where(np.logical_or(projector == '+', projector == '-'))[0]
    # swap Z block and X block
    (sym_arr[:, X_inds_fixed],
     sym_arr[:,  X_inds_fixed+N_qubits]) = (sym_arr[:, X_inds_fixed+N_qubits],
                                            sym_arr[:, X_inds_fixed].copy())

    # copy Z block into X block
    Y_inds_fixed = np.where(np.logical_or(projector == '*', projector == '%'))[0]
    sym_arr[:, Y_inds_fixed] = sym_arr[:, Y_inds_fixed + N_qubits]

    projector = PauliwordOp(sym_arr, coeff * sign)
    return projector