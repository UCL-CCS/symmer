import numpy as np
from typing import List
from functools import reduce
from symmer.operators import PauliwordOp

def exponentiate_single_Pop(P: PauliwordOp) -> PauliwordOp:
    """
    Exponentiate a single Pauli term as e^{P}

    If goal is to implement e^{iθP} then coefficient of P must be iθ (note imaginary part must be included in coeff)

    Args:
        P (PauliwordOp): Pauli operator to exponentiate
    Returns:
        exp_P (PauliwordOp): PauliwordOp representation of exponentiated operator
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
    """ 
    Computes the exponential exp(op). This is exact only when 
    op is fully commuting, otherwise approximates the exponential
    and increasing trotnum will improve precision.

    Args:
        op (PauliwordOp): Pauli operator to exponentiate
        tortnum (int): Increasing trotnum will improve precision when exact exponential is not computed. By default, it is set to 1.
    """
    op_copy = op.copy().multiply_by_constant(1/trotnum)
    factors = [exponentiate_single_Pop(P) for P in op_copy]*trotnum
    return reduce(lambda x,y:x*y, factors)

def truncated_exponential(op:PauliwordOp, truncate_at:int=10) -> PauliwordOp:
    raise NotImplementedError