import numpy as np
from copy import deepcopy
from symred.utils import gf2_gaus_elim, gf2_basis_for_gf2_rref
from symred.symplectic import PauliwordOp, StabilizerOp

class HamiltonianBiasing:
    """ Class for re-weighting Hamiltonian terms based on some criteria, such as HOMO-LUMO bias
    """
    # HOMO/LUMO bias is a value between 0 and 1 representing how sharply 
    # peaked the Gaussian distributions centred at each point should be
    HOMO_bias = 0.2
    LUMO_bias = 0.2
    # Can also specify the separation between the two distributions... 
    # a value of 1 means each is peaked either side of the HOMO-LUMO gap
    separation = 1
    
    def __init__(self, base_operator: PauliwordOp, HOMO_LUMO_gap) -> None:
        self.base_operator = base_operator
        assert(
            HOMO_LUMO_gap - int(HOMO_LUMO_gap) == 0.5
        ), 'HOMO_LUMO_gap should be specified as the mid-point between the HOMO and LUMO indices'
        self.HOMO_LUMO_gap = HOMO_LUMO_gap
        # shift qubit positions such that HOMO-LUMO gap is centred at zero
        self.shifted_q_pos = np.arange(base_operator.n_qubits) - self.HOMO_LUMO_gap
        
    def HOMO_LUMO_bias_curve(self) -> np.array:
        """ Curve constructed from two gaussians centred either side of the HOMO-LUMO gap
        The standard deviation for each distribution can be tuned independently via
        the parameters HOMO_sig (lower population), LUMO_sig (upper population) in [0, pi/2]
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
            vector to re-weight according to how close to the gap each term acts
        """
        X_block = self.base_operator.X_block
        X_op = PauliwordOp(
            np.hstack([X_block, np.zeros_like(X_block)]), 
            np.abs(self.base_operator.coeff_vec)
        ).cleanup()
        X_op.coeff_vec = np.sum(X_op.X_block*self.HOMO_LUMO_bias_curve(), axis=1)*X_op.coeff_vec
        return X_op

class StabilizerIdentification:
    def __init__(self,weighting_operator: PauliwordOp) -> None:
        self.basis_weighting = weighting_operator
        self.qubit_positions = np.arange(weighting_operator.n_qubits)
        self.term_region = [0,self.basis_weighting.n_terms]
        
    def symmetry_basis_by_term_significance(self, n_preserved):
        """ Set the number of terms to be preserved in order of coefficient magnitude
        Then generate the largest symmetry basis that preserves them
        """
        preserve = self.basis_weighting.sort(key='magnitude')[:n_preserved]
        ZX_symp = np.hstack([preserve.Z_block, preserve.X_block])
        reduced = gf2_gaus_elim(ZX_symp)
        kernel  = gf2_basis_for_gf2_rref(reduced)
        stabilizers = StabilizerOp(kernel, np.ones(kernel.shape[0]))
        mask_diag = np.where(~np.any(stabilizers.X_block, axis=1))[0]
        return StabilizerOp(stabilizers.symp_matrix[mask_diag], stabilizers.coeff_vec[mask_diag])

    def symmetry_basis_by_subspace_dimension(self, n_sim_qubits, region=None):
        """
        """
        if region is None:
            region = deepcopy(self.term_region)
        assert(n_sim_qubits < self.basis_weighting.n_qubits), 'Number of qubits to simulate exceeds those in the operator'
        assert(region[1]-region[0]>1), 'Search region collapsed without identifying any stabilizers'

        n_terms = sum(region)//2
        stabilizers = self.symmetry_basis_by_term_significance(n_terms)
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
            
        return self.symmetry_basis_by_subspace_dimension(n_sim_qubits, region=region)