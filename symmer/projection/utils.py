import numpy as np

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
