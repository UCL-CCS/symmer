import numpy as np
from ncon import ncon
from typing import Union, List, Dict
from symmer.symplectic import PauliwordOp
from copy import copy


class MPOApproximator:
    """
    Class to build MPO approximator for ground states.
    """

    def __init__(self,
            pauliList: List[str],
            coeffList: List[complex],
            Dmax: int = None) -> None:
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

Paulis = {
        'I': np.eye(2, dtype=np.complex64),
        'X': np.array([[0, 1],
                       [1, 0]], dtype=np.complex64),
        'Y': np.array([[0, -1j],
                       [1j, 0]], dtype=np.complex64),
        'Z': np.array([[1, 0],
                       [0, -1]], dtype=np.complex64),
        }

def pstrings_to_mpo(pstrings, coeffs=None, Dmax=None, debug=False):
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

    centre = int(len(mpo) / 2)

    for pstr, coeff in zip(pstrings[1:], coeffs[1:]):
        _mpo = pstring_to_mpo(pstr, coeff)

        mpo = sum_mpo(mpo, _mpo)

        if debug:
            print("Summed mpo centre shape: {}".format(mpo[centre].shape))
        mpo = truncate_MPO(mpo, Dmax)
        if debug:
            print("Truncated centre mpo shape: {}".format(mpo[centre].shape))
            print("")

    return mpo

def pstring_to_mpo(pstring, scaling=None, debug=False):

    As = []
    for p in pstring:
        pauli = Paulis[p]
        pauli_tensor = np.expand_dims(pauli, axis=(2, 3))
        if debug:
            print(p)
            print(pauli_tensor.shape)
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

def truncate_MPO(mpo, Dmax, debug=False):
    if debug:
        print(f"Curr Dmax: {Dmax}")
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
        if debug:
            print(f"n: {n}")
            print(f"A: {mpo[n].shape}")
            print(f"M: {M.shape}")
            print(f"B: {mpo[n+1].shape}")
            print(f"A': {_A.shape}")
        _A1 = ncon([M, mpo[n+1]], ((-3, 1), (-1, -2, 1, -4)))
        mpo[n+1] = _A1

    As.append(mpo[-1])

    return As

def sum_mpo(mpo1, mpo2, debug=False):
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
        if debug:
            print(f"MPO1 Shape: {mpo1[i].shape}")
            print(f"MPO2 Shape: {mpo2[i].shape}")
            print(f"New Shape: {new_tensor.shape}")
            print("")
        summed[i] = new_tensor

    return summed

