import numpy as np
from ncon import ncon
from typing import Union, List, Dict
from symmer.operators import PauliwordOp, QuantumState
from copy import copy
from cached_property import cached_property
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_dmrg import DMRG2


class MPOOp:
    """
    Class to build MPO operator from Pauli strings and coeffs.
    """

    def __init__(self,
            pauliList: List[str],
            coeffList: List[complex],
            Dmax: int = None) -> None:
        """Initialize an MPO to represent an operator from Pauli strings and
        coefficients. MPO tensors are shapes as (σ, l, i, j) where σ is the
        physical leg, l is the output leg and i and j are the remaining
        legs."""
        self.mpo = pstrings_to_mpo(pauliList, coeffList, Dmax)

    @classmethod
    def from_dictionary(cls,
            operator_dict: Dict[str, complex],
            Dmax: int = None) -> "MPOApproximator":
        """
        Initalize MPOApproximator using Pauli terms and coefficients stored in
        a dictionary like {pauli: coeff}
        """
        paulis, coeffs = zip(*operator_dict.items())
        paulis = list(paulis)
        return cls(paulis, coeffs, Dmax)

    @classmethod
    def from_WordOp(cls,
            WordOp: PauliwordOp) -> "MPOApproximator":
        """
        Initialize MPOApproximator using PauliwordOp
        """
        return cls.from_dictionary(WordOp.to_dictionary())

    @cached_property
    def to_matrix(self) -> np.ndarray:
        '''
        Contract MPO to produce matrix representation of operator.
        '''
        mpo = self.mpo
        contr = mpo[0]
        for tensor in mpo[1:]:
            σ1, l1, i1, j1 = contr.shape
            σ2, l2, i2, j2 = tensor.shape
            contr = ncon([contr, tensor], ((-1, -3, -5, 1), (-2, -4, 1, -6)))
            contr = np.reshape(contr, (σ1 * σ2, l1 * l2, i1, j2))

        contr = np.squeeze(contr)
        return contr
    
def get_MPO(operator: PauliwordOp, max_bond_dimension: int) -> MPOOp:
    """ Return the Matrix Product Operator (MPO) of a PauliwordOp 
    (linear combination of Paulis) given a maximum bond dimension
    """
    pstrings, coefflist = zip(*operator.to_dictionary.items())
    mpo = MPOOp(pstrings, coefflist, Dmax=max_bond_dimension)
    return mpo

def find_groundstate_quimb(MPOOp: MPOOp, dmrg=None, gs_guess=None) -> QuantumState:
    """
    Use quimb's DMRG2 optimiser to approximate groundstate of MPOOp

    Args:
        MPOOp: MPOOp representing operator
        dmrg: Quimb DMRG solver class
        gs_guess: Guess for the ground state, used as intialisation for the
                DMRG optimiser. Represented as a dense array.
    Returns:
        dmrg_state (QuantumState): Approximated groundstate

    """
    mpo = [np.squeeze(m) for m in MPOOp.mpo]
    MPO = MatrixProductOperator(mpo, 'dulr')

    if gs_guess is not None:
        no_qubits = int(np.log2(gs_guess.shape[0]))
        dims = [2] * no_qubits
        gs_guess = MatrixProductState.from_dense(gs_guess, dims)

    # Useful default for DMRG optimiser
    if dmrg is None:
        dmrg = DMRG2(MPO, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-10, p0=gs_guess)
    dmrg.solve(verbosity=0, tol=1e-6)

    dmrg_state = dmrg.state.to_dense()
    dmrg_state = QuantumState.from_array(dmrg_state).cleanup(zero_threshold=1e-5)

    return dmrg_state


Paulis = {
        'I': np.eye(2, dtype=np.complex64),
        'X': np.array([[0, 1],
                       [1, 0]], dtype=np.complex64),
        'Y': np.array([[0, -1j],
                       [1j, 0]], dtype=np.complex64),
        'Z': np.array([[1, 0],
                       [0, -1]], dtype=np.complex64),
        }

def coefflist_to_complex(coefflist):
    '''
    Convert a list of real + imaginary components into a complex vector
    '''
    arr = np.array(coefflist, dtype=complex)

    return arr[:, 0] + 1j*arr[:, 1]

def pstrings_to_mpo(pstrings, coeffs=None, Dmax=None):
    ''' Convert a list of Pauli Strings into an MPO. If coeff list is given,
    rescale each Pauli string by the corresponding element of the coeff list.
    Bond dim specifies the maximum bond dimension, if None, no maximum bond
    dimension.

    '''
    if coeffs is None:
        coeffs = np.ones(len(pstrings))

    if Dmax is None:
        Dmax = np.inf

    mpo = pstring_to_mpo(pstrings[0], coeffs[0])

    for pstr, coeff in zip(pstrings[1:], coeffs[1:]):
        _mpo = pstring_to_mpo(pstr, coeff)
        mpo = sum_mpo(mpo, _mpo)
        mpo = truncate_MPO(mpo, Dmax)

    return mpo

def pstring_to_mpo(pstring, scaling=None):

    As = []
    for p in pstring:
        pauli = Paulis[p]
        pauli_tensor = np.expand_dims(pauli, axis=(2, 3))
        As.append(pauli_tensor)

    if scaling is not None:
        As[0] = As[0] * scaling
    return As


def truncated_SVD(M, Dmax=None):
    U, S, V = np.linalg.svd(M, full_matrices=False)

    if Dmax is not None and len(S) > Dmax:
        S = S[:Dmax]
        U = U[:, :Dmax]
        V = V[:Dmax, :]

    return U, S, V

def truncate_MPO(mpo, Dmax):
    As = []
    for n in range(len(mpo) - 1):  # Don't need to run on the last term
        A = mpo[n]
        σ, l, i, j = A.shape
        A = A.reshape(σ * l * i, j)
        U, S, V = truncated_SVD(A, Dmax)
        D = len(S)

        # Update current term
        _A = U.reshape(σ, l, i, D)
        As.append(_A)

        # Update the next term
        M = np.diag(S) @ V
        _A1 = ncon([M, mpo[n+1]], ((-3, 1), (-1, -2, 1, -4)))
        mpo[n+1] = _A1

    As.append(mpo[-1])

    return As

def sum_mpo(mpo1, mpo2):
    summed = [None] * len(mpo1)
    σ10, l10, i10, j10 = mpo1[0].shape
    σ20, l20, i20, j20 = mpo2[0].shape
    t10 = copy(mpo1[0])
    t20 = copy(mpo2[0])
    first_sum = np.zeros((σ10, l10, i10, j10+j20), dtype=complex)
    first_sum[:, :, :, :j10] = t10
    first_sum[:, :, :, j10:] = t20
    summed[0] = first_sum

    σ1l, l1l, i1l, j1l = mpo1[-1].shape
    σ2l, l2l, i2l, j2l = mpo2[-1].shape
    t1l = copy(mpo1[-1])
    t2l = copy(mpo2[-1])
    first_sum = np.zeros((σ1l, l1l, i1l + i2l, j1l), dtype=complex)
    first_sum[:, :, :i1l, :] = t1l
    first_sum[:, :, i1l:, :] = t2l
    summed[-1] = first_sum
    for i in range(1, len(mpo1) - 1):
        σ1, l1, i1, j1 = mpo1[i].shape
        σ2, l2, i2, j2 = mpo2[i].shape
        t1 = copy(mpo1[i])
        t2 = copy(mpo2[i])


        new_shape = (σ1, l1, i1+i2, j1+j2)

        new_tensor = np.zeros(new_shape, dtype=complex)

        new_tensor[:, :, :i1, :j1] = t1
        new_tensor[:, :, i1:, j1:] = t2
        summed[i] = new_tensor

    return summed

