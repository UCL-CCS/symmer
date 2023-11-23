import numpy as np
from copy import deepcopy
from scipy.optimize import differential_evolution
from symmer.operators import PauliwordOp, IndependentOp
from typing import Union, Optional
from symmer.utils import random_anitcomm_2n_1_PauliwordOp, product_list

def norm(vector: np.array) -> float:
    """
    Args:
        vector (np.array): Vector whose 12-norm has to be found.

    Returns:
        l2-norm of input vector
    """
    return np.sqrt(np.dot(vector, vector.conjugate()))

def lp_norm(vector: np.array, p:int=2) -> float:
    """
    Args:
        vector (np.array): Vector whose lp-norm has to be found.
        p (int): Power.  It is the power to which the absolute value of elements are raised before summation. is a positive real number.

    Returns:
        lp-norm of vector
    """
    return np.power(np.sum(np.power(np.abs(vector), p)), 1/p)

def one_qubit_noncontextual_gs(op: PauliwordOp):
    assert op.n_qubits == 1, 'Operator consists of more than one qubit'
    op.to

def basis_score(
        weighting_operator: PauliwordOp,
        basis: IndependentOp,
        p:int=1
    ) -> float:
    """ 
    Evaluate the score of an input basis according 
    to the basis weighting operator, for example:
        - set Hamiltonian cofficients to 1 for unweighted number of commuting terms
        - specify as the SOR Hamiltonian to weight according to second-order response
        - input UCC operator to weight according to coupled-cluster theory <- best performance
        - if None given then weights by Hamiltonian coefficient magnitude
    
    p determines which norm is used, i.e. lp --> (\sum_{t} |t|^p)^(1/p)

    Args:
        weighting_operator (PauliwordOp): Basis weighting operator.
        basis (IndependentOp): Basis whose basis score has to be determined.
        p (int): Power which determines which norm is used. Its default value is 1.

    Returns:
        Basis score (float) of the input basis.
    """
    # mask terms of the weighting operator that are preserved under projection over the basis
    mask_preserved = np.where(np.all(weighting_operator.commutes_termwise(basis),axis=1))[0]
    return (
        lp_norm(weighting_operator.coeff_vec[mask_preserved], p=p) /
        lp_norm(weighting_operator.coeff_vec, p=p)
    )

def update_eigenvalues(
        generators: IndependentOp, 
        stabilizers: IndependentOp
    ) -> None:
    """ 
    Update the +/-1 eigenvalue assigned to the input stabilizer according to the noncontextual ground state configuration.
    
    Args:
        generators (IndependentOp): Generator
        stabilizers (IndependentOp): Stabilizer 
    """
    reconstruction, successfully_reconstructed = stabilizers.generator_reconstruction(generators)
    if ~np.all(successfully_reconstructed):
        raise ValueError('Generators not sufficient to reconstruct symmetry operators')
    stabilizers.coeff_vec = (-1) ** np.count_nonzero(
        np.bitwise_and(
            reconstruction, 
            np.asarray(generators.coeff_vec)==-1
        ),
        axis=1
    )

class StabilizerIdentification:
    def __init__(self,
        weighting_operator: PauliwordOp,
        use_X_only = False
        ) -> None:
        """
        Args:
            weighting_operator (PauliwordOp): Basis weighting operator. By default, it is set to None.
            use_X_only (bool): Default value is 'False'.
        """
        self.weighting_operator = weighting_operator
        self.use_X_only = use_X_only
        self.build_basis_weighting_operator()

    def build_basis_weighting_operator(self):
        if self.use_X_only:
            X_block = self.weighting_operator.X_block
            self.weighting_operator = PauliwordOp(
                np.hstack([X_block, np.zeros_like(X_block)]), 
                np.abs(self.weighting_operator.coeff_vec)
            ).cleanup()
        self.basis_weighting = self.weighting_operator.sort(by='magnitude')
        self.qubit_positions = np.arange(self.weighting_operator.n_qubits)
        self.term_region = [0,self.basis_weighting.n_terms]
        
    def symmetry_generators_by_term_significance(self, n_preserved):
        """ 
        Set the number of terms to be preserved in order of coefficient magnitude,
        Then generate the largest symmetry basis that preserves them.

        Args: 
            n_preserved (int): Number of terms to be preserved in order of coefficient magnitude.
        
        Returns:
            The largest symmetry basis that preserves order of coefficient magnitude.
        """
        preserve = self.basis_weighting[:n_preserved]
        stabilizers = IndependentOp.symmetry_generators(preserve, commuting_override=True)
        mask_diag = np.where(~np.any(stabilizers.X_block, axis=1))[0]
        return IndependentOp(stabilizers.symp_matrix[mask_diag], stabilizers.coeff_vec[mask_diag])

    def symmetry_generators_by_subspace_dimension(self, n_sim_qubits, region=None):
        """
        Args:
            n_sim_qubits (int): Number of qubits to simulate.
            region (list[int]): Region

        Returns:
            Symetry generators by subspace dimension.
        """
        if region is None:
            region = deepcopy(self.term_region)
        assert(n_sim_qubits < self.basis_weighting.n_qubits), 'Number of qubits to simulate exceeds those in the operator'
        assert(region[1]-region[0]>1), 'Search region collapsed without identifying any stabilizers'

        n_terms = sum(region)//2
        stabilizers = self.symmetry_generators_by_term_significance(n_terms)
        current_n_qubits = self.basis_weighting.n_qubits - stabilizers.n_terms
        sign = np.sign(current_n_qubits - n_sim_qubits)

        if sign==0:
            # i.e. n_sim_qubits == current_n_qubits
            return stabilizers
        elif sign==+1:
            # i.e. n_sim_qubits < current_n_qubits
            region[1] = n_terms
        else:
            region[0] = n_terms
            
        return self.symmetry_generators_by_subspace_dimension(n_sim_qubits, region=region)

class ObservableBiasing:
    """ 
    Class for re-weighting Hamiltonian terms based on some criteria, such as HOMO-LUMO bias.
    
    Attributes:
        HOMO_bias (float): HUMO Bias. Its value is in between 0 and 1. By default it's value is set to be 0.2
        LUMO_bias (float): LUMO Bias. Its value is in between 0 and 1. By default it's value is set to be 0.2
        seperation (int):  Separation between the two distributions. By default it is set to 1. A value of 1 means each distribution is peaked either side of the HOMO-LUMO gap.
    """
    # HOMO/LUMO bias is a value between 0 and 1 representing how sharply 
    # peaked the Gaussian distributions centred at each point should be
    HOMO_bias = 0.2
    LUMO_bias = 0.2
    # Can also specify the separation between the two distributions... 
    # a value of 1 means each is peaked either side of the HOMO-LUMO gap
    separation = 1
    
    def __init__(self, base_operator: PauliwordOp, HOMO_LUMO_gap) -> None:
        """
        Args:
            base_operator (PauliwordOp): Base Operator.
            HOMO_LUMO_gap: HOMO-LUMO gap. It should be specified as the mid-point between the HOMO and LUMO indices.
        """
        self.base_operator = base_operator
        assert(
            HOMO_LUMO_gap - int(HOMO_LUMO_gap) == 0.5
        ), 'HOMO_LUMO_gap should be specified as the mid-point between the HOMO and LUMO indices'
        self.HOMO_LUMO_gap = HOMO_LUMO_gap
        # shift qubit positions such that HOMO-LUMO gap is centred at zero
        self.shifted_q_pos = np.arange(base_operator.n_qubits) - self.HOMO_LUMO_gap
        
    def HOMO_LUMO_bias_curve(self) -> np.array:
        """ 
        Curve constructed from two gaussians centred either side of the HOMO-LUMO gap.
        The standard deviation for each distribution can be tuned independently via
        the parameters HOMO_sig (lower population), LUMO_sig (upper population) in [0, pi/2].

        Returns:
            HOMO LUMO bias curve (np.array).
        """
        shift = self.separation - 1/2
        # standard deviation about the HOMO/LUMO-centred Gaussian distributions:
        HOMO_sigma = np.tan((1-self.HOMO_bias)*np.pi/2) 
        LUMO_sigma = np.tan((1-self.LUMO_bias)*np.pi/2)
        # lower population (centred at HOMO)
        if HOMO_sigma!=0:
            L = np.exp(-np.square((self.shifted_q_pos+shift)/HOMO_sigma)/2)
        else:
            non_zero_index = int(self.HOMO_LUMO_gap-shift)
            L = np.eye(1,self.base_operator.n_qubits,non_zero_index).reshape(self.base_operator.n_qubits)
        # upper population (centred at LUMO)
        if LUMO_sigma!=0:
            U = np.exp(-np.square((self.shifted_q_pos-shift)/LUMO_sigma)/2)
        else:
            non_zero_index = int(self.HOMO_LUMO_gap+shift)
            U = np.eye(1,self.base_operator.n_qubits,non_zero_index).reshape(self.base_operator.n_qubits)
        return (L + U)/2
    
    def HOMO_LUMO_biased_operator(self) -> np.array:
        """ 
        - First converts the base operator to a PauliwordOp consisting of Pauli I, X 
            (since only interested in where the terms can affect orbital occupation)
        - Second, assigns a weight to each nontrivial qubit position according to the bias curve
            and sums the total HOM-LUMO-biased contribution. This is multiplied by the coefficient
            vector to re-weight according to how close to the gap each term acts.

        Returns:
            reweighted_operator (PauliwordOp): Reweighted Operator.
        """
        reweighted_operator = self.base_operator.copy()
        reweighted_operator.coeff_vec = np.sum(
            reweighted_operator.X_block*self.HOMO_LUMO_bias_curve(), 
            axis=1
        )*reweighted_operator.coeff_vec
        return reweighted_operator

def stabilizer_walk(
        n_sim_qubits,
        biasing_operator: ObservableBiasing,
        weighting_operator: PauliwordOp = None,
        print_info: bool = False,
        use_X_only: bool = False
    ) -> IndependentOp:
    """
    Args:
        n_sim_qubits (int): Number of qubits to simulate.
        biasing_operator (ObservableBiasing): Biasing Operator.
        weighting_operator (PauliwordOp): Basis weighting operator. By default, it is set to None.
        print_info (bool): If True, Info about optimal score for HUMO/LUMO bias is printed. By default, it is set to 'False'.
        use_X_only:  Default value is 'False'.

    Returns:
        S IndependentOp: Stablizers
    """
    if weighting_operator is None:
        weighting_operator = biasing_operator.base_operator
        
    def get_stabilizers(x):
        biasing_operator.HOMO_bias,biasing_operator.LUMO_bias = x
        biased_op = biasing_operator.HOMO_LUMO_biased_operator()
        stabilizers = StabilizerIdentification(biased_op, use_X_only=use_X_only)
        S = stabilizers.symmetry_generators_by_subspace_dimension(n_sim_qubits)
        return(S)
    
    def objective(x):
        S = get_stabilizers(x)
        stab_score = basis_score(weighting_operator, S)
        return -stab_score
    
    opt_out = differential_evolution(objective, bounds=[(0,1),(0,1)])
    stab_score =-opt_out['fun']
    bias_param =opt_out['x']
    S = get_stabilizers(bias_param)
    
    if print_info:
        print(f'Optimal score w(S)={stab_score} for HOMO/LUMO bias {bias_param}')
    
    return S

def get_noncon_generators_from_commuting_stabilizers(stabilizers: Union[PauliwordOp, IndependentOp],
                                                     weighting_operator: PauliwordOp,
                                                     return_clique_only: Optional[bool] = False) -> IndependentOp:
    """
    Given a set of commuting stabilizers and weighting operator find best noncontextual generating set
    (ie works out best anticommuting addition to generators that reconstructs most of the weighting_operator)

    Args:
        stabilizers (PauliwordOp): operator containing commuting symmetries
        weighting_operator (PauliwordOp): operator to inform ac choice

    Returns:
        new_stabilizers (IndependentOp): noncontextual generators that contain ac component (generates noncon op under Jordan product)
    """
    if not np.all(stabilizers.commutes_termwise(stabilizers)):
        # stabilizers already contain ac component
        return stabilizers #, PauliwordOp.empty(stabilizers.n_qubits).cleanup()
    else:
        # below generates the generators of inout stabilizers
        generators = stabilizers.generators

    best_l1_norm = -1
    # find qubits uniquely defined by generators
    unique_q_inds = ~(np.sum(np.logical_xor(generators.Z_block, generators.X_block), axis=0)-1).astype(bool)
    for stab in generators:
        # find unique non identity positions
        act_positions = np.logical_and(np.logical_xor(stab.Z_block, stab.X_block)[0], unique_q_inds)

        # work out number of qubits on these positions
        n_act_qubits = np.sum(act_positions)

        # find AC clique of size 2n containing given stabilizer
        ac_basis = random_anitcomm_2n_1_PauliwordOp(n_act_qubits, apply_clifford=False)[1:]
        new_basis = PauliwordOp(np.zeros((n_act_qubits * 2, stab.n_qubits * 2), dtype=bool),
                                np.ones(n_act_qubits * 2))

        new_basis.symp_matrix[:, [*act_positions, *act_positions]] = ac_basis.symp_matrix

        # ensure stab is in new_basis
        gen, mask = stab.generator_reconstruction(new_basis)
        required_products = gen[0].nonzero()[0][1:]
        if len(required_products) > 0:
            prod = product_list(new_basis[required_products])
            new_basis = (new_basis * prod).cleanup()
        new_basis.coeff_vec = np.ones_like(new_basis.coeff_vec)

        # find best reconstruction
        _, mask = weighting_operator.generator_reconstruction(new_basis)
        success = weighting_operator[mask]
        l1_norm = np.linalg.norm(success.coeff_vec, ord=1)

        if l1_norm > best_l1_norm:
            new_stabilizers = generators - stab + new_basis
            best_l1_norm = l1_norm
            stab_used = stab.copy()

    assert new_stabilizers.is_noncontextual, 'new stabilizers are not noncontextual'

    # commuting_stabs = IndependentOp.from_PauliwordOp(stabilizers)
    # anticommuting_stabs = IndependentOp.from_PauliwordOp(new_stabilizers) - commuting_stabs
    # return commuting_stabs, anticommuting_stabs
    if return_clique_only:
        return IndependentOp.from_PauliwordOp(new_stabilizers) - generators, stab_used
    else:
        return IndependentOp.from_PauliwordOp(new_stabilizers)