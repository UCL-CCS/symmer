import numpy as np
from symmer.symplectic import PauliwordOp, StablizerOp

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

def unit_n_sphere_cartesian_coords(angles: np.array) -> np.array:
    """ Input an array of angles of length n, returns the n+1 cartesian coordinates 
    of the corresponding unit n-sphere in (n+1)-dimensional Euclidean space.
    """
    cartesians = [np.prod(np.sin(angles[:i]))*np.cos(angles[i]) for i in range(len(angles))]
    cartesians.append(np.prod(np.sin(angles)))
    return np.array(cartesians)

def basis_score(
        weighting_operator: PauliwordOp,
        basis: StabilizerOp,
        p:int=1
    ) -> float:
    """ Evaluate the score of an input basis according 
    to the basis weighting operator, for example:
        - set Hamiltonian cofficients to 1 for unweighted number of commuting terms
        - specify as the SOR Hamiltonian to weight according to second-order response
        - input UCC operator to weight according to coupled-cluster theory <- best performance
        - if None given then weights by Hamiltonian coefficient magnitude
    
    p determines which norm is used, i.e. lp --> (\sum_{t} |t|^p)^(1/p)
    """
    # mask terms of the weighting operator that are preserved under projection over the basis
    mask_preserved = np.where(np.all(weighting_operator.commutes_termwise(basis),axis=1))[0]
    return (
        lp_norm(weighting_operator.coeff_vec[mask_preserved], p=p) /
        lp_norm(weighting_operator.coeff_vec, p=p)
    )
