import numpy as np
from typing import List
from functools import reduce
from symmer.symplectic import PauliwordOp

def exponentiate_single_Pop(P: PauliwordOp) -> PauliwordOp:
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
    factors = [exponentiate_single_Pop(P) for P in op_copy]*trotnum
    return reduce(lambda x,y:x*y, factors)

def truncated_exponential(op:PauliwordOp, truncate_at:int=10) -> PauliwordOp:
    raise NotImplementedError