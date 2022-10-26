import numpy as np
from symmer.symplectic import PauliwordOp, StabilizerOp

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

def update_eigenvalues(
        basis: StabilizerOp, 
        stabilizers: StabilizerOp
    ) -> None:
    """ Update the +/-1 eigenvalue assigned to the input stabilizer
    according to the noncontextual ground state configuration
    """
    reconstruction, successfully_reconstructed = stabilizers.basis_reconstruction(basis)
    if ~np.all(successfully_reconstructed):
        raise ValueError('Basis not sufficient to reconstruct symmetry operators')
    stabilizers.coeff_vec = (-1) ** np.count_nonzero(
        np.bitwise_and(
            reconstruction, 
            basis.coeff_vec==-1
        ),
        axis=1
    )