import numpy as np
from typing import List
from functools import reduce
from symmer.symplectic import PauliwordOp

def _exponentiate(P: PauliwordOp) -> PauliwordOp:
    """ exponentiate a single Pauli term
    """
    assert(P.n_terms == 1), 'Can only exponentiate single Pauli terms'
    P_copy = P.copy()
    P_copy.coeff_vec[0] = 1
    exp_P = (
        (P_copy**0).multiply_by_constant(np.cosh(P.coeff_vec[0])) +
        (P_copy**1).multiply_by_constant(np.sinh(P.coeff_vec[0]))
    )
    return exp_P

def trotter(op:PauliwordOp, trotnum:int=1) -> PauliwordOp:
    """ Computes the exponential exp(op). This is exact only when 
    op is fully commuting, otherwise approximates the exponential
    and increasing trotnum will improve precision.
    """
    op_copy = op.copy().multiply_by_constant(1/trotnum)
    factors = [_exponentiate(P) for P in op_copy]*trotnum
    return reduce(lambda x,y:x*y, factors)

def truncated_exponential(op:PauliwordOp, truncate_at:int=10) -> PauliwordOp:
    raise NotImplementedError

def _tensor(left:PauliwordOp, right:PauliwordOp) -> PauliwordOp:
    """ Tensor two Pauli operators for left to right (cannot interlace currently)
    """
    identity_block_right = np.zeros([right.n_terms, left.n_qubits]).astype(int)
    identity_block_left  = np.zeros([left.n_terms,  right.n_qubits]).astype(int)
    padded_left_symp = np.hstack([left.X_block, identity_block_left, left.Z_block, identity_block_left])
    padded_right_symp = np.hstack([identity_block_right, right.X_block, identity_block_right, right.Z_block])
    left_factor = PauliwordOp(padded_left_symp, left.coeff_vec)
    right_factor = PauliwordOp(padded_right_symp, right.coeff_vec)
    return left_factor * right_factor

def tensor(factor_list:List[PauliwordOp]) -> PauliwordOp:
    return reduce(lambda x,y:_tensor(x,y), factor_list)