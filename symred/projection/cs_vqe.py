from typing import Tuple, List
from cached_property import cached_property
import warnings
import numpy as np
from scipy.optimize import shgo, differential_evolution
from symred.utils import unit_n_sphere_cartesian_coords, gf2_gaus_elim
from symred.symplectic import PauliwordOp, StabilizerOp, symplectic_to_string
from symred.projection import S3_projection
from symred.unitary_partitioning import AntiCommutingOp

class CS_VQE(S3_projection):
    """
    """
    def __init__(self,
            operator: PauliwordOp,
            ref_state: np.array = None,
            target_sqp: str = 'Z',
            noncontextual_form = 'legacy',
            basis_weighting_operator: PauliwordOp = None
        ) -> None:
        """ 
        """
        self.operator = operator
        self.ref_state = ref_state
        self.target_sqp = target_sqp
        self.noncontextual_form = noncontextual_form
        self.contextual_operator = (operator-self.noncontextual_operator).cleanup_zeros()
        if basis_weighting_operator is not None:
            self.basis_weighting_operator = basis_weighting_operator
        else:
            self.basis_weighting_operator = self.contextual_operator
        # decompose the noncontextual set into a dictionary of its 
        # universally commuting elements and anticommuting cliques
        self.noncontextual_reconstruction = (
            self.noncontextual_operator.basis_reconstruction(self.noncontextual_basis)
        )
        self.r_indices = self.noncontextual_reconstruction[:,:self.n_cliques]
        self.G_indices = self.noncontextual_reconstruction[:,self.n_cliques:]
        self.clique_operator = (self.noncontextual_basis[:self.n_cliques])
        symmetry_generators_symp = self.noncontextual_basis.symp_matrix[self.n_cliques:]
        self.symmetry_generators = StabilizerOp(
            symmetry_generators_symp,
            np.ones(symmetry_generators_symp.shape[0])
        )
        # determine the noncontextual ground state - this updates the coefficients of the clique 
        # representative operator C(r) and symmetry generators G with the optimal configuration
        self.solve_noncontextual(ref_state)
        # Determine the unitary partitioning rotations and the single Pauli operator that is rotated onto
        self.clique_operator = AntiCommutingOp(
            self.clique_operator.symp_matrix, 
            self.clique_operator.coeff_vec
        )
        self.SeqRots, self.C0 = self.clique_operator.gen_seq_rotations(
            s_index=np.where(~np.any(self.clique_operator.X_block, axis=1))[0][0], 
            check_reduction=True
        )
        
    def basis_score(self, 
            basis: StabilizerOp
        ) -> float:
        """ Evaluate the score of an input basis according 
        to the basis weighting operator, for example:
            - set Hamiltonian cofficients to 1 for unweighted number of commuting terms
            - specify as the SOR Hamiltonian to weight according to second-order response
            - input UCC operator to weight according to coupled-cluster theory <- best performance
            - if None given then weights by Hamiltonian coefficient magnitude
        """
        # mask terms of the weighting operator that are preserved under projection over the basis
        mask_preserved = np.where(np.all(self.basis_weighting_operator.commutes_termwise(basis),axis=1))[0]
        return (
            np.linalg.norm(self.basis_weighting_operator.coeff_vec[mask_preserved]) /
            np.linalg.norm(self.basis_weighting_operator.coeff_vec)
            )
    
    def update_eigenvalues(self, stabilizers: StabilizerOp) -> None:
        """ Update the +/-1 eigenvalue assigned to the input stabilizer
        according to the noncontextual ground state configuration
        """
        stabilizers.coeff_vec = (-1) ** np.count_nonzero(
            np.bitwise_and(
                stabilizers.basis_reconstruction(self.symmetry_generators), 
                self.symmetry_generators.coeff_vec==-1
            ),
            axis=1
        )

    @cached_property
    def noncontextual_operator(self) -> PauliwordOp:
        """ Extract a noncontextual set of Pauli terms from the operator

        Implementation of the algorithm in https://doi.org/10.1103/PhysRevLett.123.200501
        Does a single pass over the Hamiltonian and appends terms to noncontextual_operator 
        that do not make it contextual - easy to do multiple passes although this does not 
        seem to yield better results from experimentation.

        TODO graph-based approach, currently uses legacy implementation
        """
        if self.noncontextual_form == 'diag':
            # mask diagonal terms of the operator
            mask_diag = np.where(~np.any(self.operator.X_block, axis=1))
            noncontextual_operator = PauliwordOp(
                self.operator.symp_matrix[mask_diag],
                self.operator.coeff_vec[mask_diag]
            )
        elif self.noncontextual_form == 'legacy':
            # order the operator terms by coefficient magnitude
            check_ops = self.operator.sort(key='magnitude')
            # initialise as identity with 0 coefficient
            I_symp = np.zeros(2*self.operator.n_qubits, dtype=int)
            noncontextual_operator = PauliwordOp(I_symp, [0])
            for i in range(check_ops.n_terms):
                if (noncontextual_operator+check_ops[i]).is_noncontextual:
                    noncontextual_operator+=check_ops[i]
        else:
            raise ValueError('noncontextual_form not recognised: must be one of diag or legacy.')
            
        return noncontextual_operator

    @cached_property
    def noncontextual_basis(self) -> StabilizerOp:
        """ Find an independent basis for the noncontextual symmetry
        """
        self.decomposed = {}
        # extract the universally commuting noncontextual terms
        universal_mask = np.where(np.all(self.noncontextual_operator.adjacency_matrix, axis=1))
        universal_operator = PauliwordOp(self.noncontextual_operator.symp_matrix[universal_mask],
                                         self.noncontextual_operator.coeff_vec[universal_mask])
        self.decomposed['symmetry'] = universal_operator
        # build the noncontextual basis by performing Gaussian elimination on the symmetry terms:
        reduced_universal = gf2_gaus_elim(universal_operator.symp_matrix)
        reduced_universal = reduced_universal[np.where(np.any(reduced_universal, axis=1))]
        basis = PauliwordOp(reduced_universal, np.ones(reduced_universal.shape[0]))
       
        # identify the anticommuting cliques
        clique_union = (self.noncontextual_operator - universal_operator).cleanup_zeros()
        if clique_union.n_terms != 0:
            # identify unique rows in the adjacency matrix with inverse mapping 
            # so that terms of the same clique have matching indices
            clique_characters, inverse = np.unique(clique_union.adjacency_matrix, axis=0, return_inverse=True)
            mask_cliques = np.stack([np.where(inverse==i)[0] for i in np.unique(inverse)])
            # mask each clique and select a class represetative for its contribution in the noncontextual basis
            clique_reps = []
            for i, (Ci_symp, Ci_coef) in enumerate(
                zip(clique_union.symp_matrix[mask_cliques],clique_union.coeff_vec[mask_cliques])
            ):
                Ci_operator = PauliwordOp(Ci_symp, Ci_coef)
                self.decomposed[f'clique_{i}'] = Ci_operator
                # choose cliques representative that maximises basis_score
                rep_scores = [(Ci_operator[i], self.basis_score(Ci_operator[i])) for i in range(len(Ci_coef))]
                clique_reps.append(sorted(rep_scores, key=lambda x:-x[1])[0][0].symp_matrix)
            clique_reps = np.vstack(clique_reps)
            basis = basis + PauliwordOp(clique_reps, np.ones(clique_reps.shape[0]))
            # order so clique terms appear first
            basis_order = np.lexsort(basis.adjacency_matrix)
            basis = StabilizerOp(basis.symp_matrix[basis_order],np.ones(basis.n_terms))

        self.n_cliques = np.count_nonzero(np.any(~basis.adjacency_matrix, axis=1))

        return basis
    
    def noncontextual_objective_function(self, 
            nu: np.array, 
            r: np.array
        ) -> float:
        """ The classical objective function that encodes the noncontextual energies
        """
        G_prod = (-1)**np.count_nonzero(np.logical_and(self.G_indices==1, nu == -1), axis=1)
        r_part = np.sum(self.r_indices*r, axis=1)
        r_part[np.where(r_part==0)]=1
        return np.sum(self.noncontextual_operator.coeff_vec*G_prod*r_part).real

    def solve_noncontextual(self, ref_state: np.array = None) -> None:
        """ Minimize the classical objective function, yielding the noncontextual ground state
        """
        def convex_problem(nu):
            """ given +/-1 value assignments nu, solve for the clique operator coefficients.
            Note that, with nu fixed, the optimization problem is now convex.
            """
            # given M cliques, optimize over the unit (M-1)-sphere and convert to cartesians for the r vector
            r_bounds = [(0, np.pi)]*(self.n_cliques-2)+[(0, 2*np.pi)]
            optimizer_output = differential_evolution(
                func=lambda angles:self.noncontextual_objective_function(
                    nu, unit_n_sphere_cartesian_coords(angles)
                    ), 
                bounds=r_bounds
            )
            optimized_energy = optimizer_output['fun']
            optimized_angles = optimizer_output['x']
            r_optimal = unit_n_sphere_cartesian_coords(optimized_angles)
            return optimized_energy, r_optimal

        if ref_state is None:
            # optimize discrete value assignments nu by relaxation to continuous variables
            nu_bounds = [(0, np.pi)]*self.symmetry_generators.n_terms
            optimizer_output = shgo(func=lambda angles:convex_problem(np.cos(angles))[0], bounds=nu_bounds)
            # if optimization was successful the optimal angles should consist of 0 and pi
            self.symmetry_generators.coeff_vec = np.array(np.cos(optimizer_output['x']), dtype=int)
        else:
            # update the symmetry generator G coefficients w.r.t. the reference state
            self.symmetry_generators.update_sector(ref_state=ref_state)
        
        if self.n_cliques != 0:
            # optimize the clique operator coefficients
            fix_nu = self.symmetry_generators.coeff_vec
            self.noncontextual_energy, r = convex_problem(fix_nu)
            self.clique_operator.coeff_vec = r
        else:
            self.noncontextual_energy = self.noncontextual_objective_function(
                nu = self.symmetry_generators.coeff_vec, r=None
            )
        
    def project_onto_subspace(self,
            stabilizers: List[PauliwordOp],
            enforce_clique_operator=False,
            aux_operator: PauliwordOp = None
        ) -> PauliwordOp:
        """ input a list of independent operators one wishes to map onto single-qubit 
        Pauli operators and project into the corresponding stabilizer subspace
        """
        # define the operator to be projected (aux_operator faciliates ansatze to be projected)
        if aux_operator is not None:
            operator_to_project = aux_operator.copy()
        else:
            operator_to_project = self.operator.copy()
        
        # only allow stabilizers that commute with the cliques, else behaviour is unpredictable
        valid_stab_indices = np.where(
            ~np.any(~stabilizers.commutes_termwise(self.clique_operator), axis=1))[0]
        fix_stabilizers = PauliwordOp(
            stabilizers.symp_matrix[valid_stab_indices],
            stabilizers.coeff_vec[valid_stab_indices]
        )
        # raise a warning if any stabilizers are discarded due to anticommutation with a clique
        if len(valid_stab_indices) < stabilizers.n_terms:
            removed = list((stabilizers-fix_stabilizers).cleanup_zeros().to_dictionary.keys())
            warnings.warn(
                'Specified a clique element in the stabilizer set!\n' +
                f'The term(s) {removed} were discarded, but note that the number of ' +
                'qubits in the stabilizer subspace will be greater than expected.'
            )
        # update the eigenvalue assignments to the specified stabilizers 
        # in accordance with the noncontextual ground state
        self.update_eigenvalues(fix_stabilizers)
        
        # if the clique operator is to be enforced, perform unitary partitioning:
        insert_rotations=[]
        if enforce_clique_operator and self.n_cliques != 0:
            # if any stabilizers in the list contain more than one term then apply unitary partitioning
            fix_stabilizers += self.C0
            insert_rotations = self.SeqRots
            
        # instantiate as StabilizerOp to ensure algebraic independence and coefficients are +/-1
        fix_stabilizers = StabilizerOp(
            fix_stabilizers.symp_matrix, 
            np.array(fix_stabilizers.coeff_vec, dtype=int),
            target_sqp=self.target_sqp
        )
        # instantiate the parent S3_projection class with the stabilizers we are enforcing
        super().__init__(fix_stabilizers)

        return self.perform_projection(
            operator=operator_to_project,
            insert_rotations=insert_rotations
        )

class CS_VQE_LW(S3_projection):
    """ A lightweight CS-VQE implementation in which we choose an arbitrary
    Pauli Z-basis and project in accordance with it. Can be interpretted as
    CS-VQE with the noncontextual set taken as the diagonal Hamiltonian terms.
    Same result can be obtained by setting noncontextual_form='diag' in CS_VQE.
    """
    def __init__(self, 
            operator: PauliwordOp,
            ref_state: np.array,
            target_sqp: str = 'Z',
            basis_weighting_operator=None):
        self.operator = operator
        self.ref_state = ref_state
        self.target_sqp = target_sqp
        if basis_weighting_operator is None:
            mask_diag = np.where(~np.any(self.operator.X_block, axis=1))
            noncontextual_operator = PauliwordOp(
                self.operator.symp_matrix[mask_diag],
                self.operator.coeff_vec[mask_diag]
            )
            contextual_operator = (operator-noncontextual_operator).cleanup_zeros()
            self.basis_weighting_operator = contextual_operator
        else:
            self.basis_weighting_operator = basis_weighting_operator

    def project_onto_subspace(self, 
            basis: StabilizerOp,
            aux_operator: PauliwordOp = None
        ) -> PauliwordOp:
        """ Project the operator in accordance with the supplied basis
        """
        # define the operator to be projected (aux_operator faciliates ansatze to be projected)
        if aux_operator is not None:
            operator_to_project = aux_operator.copy()
        else:
            operator_to_project = self.operator.copy()
        
        # instantiate as StabilizerOp to ensure algebraic independence and coefficients are +/-1
        basis = StabilizerOp(
            basis.symp_matrix, 
            np.ones(basis.n_terms, dtype=int), 
            target_sqp=self.target_sqp
        )
        # update symmetry sector in accordance with the reference state
        basis.update_sector(ref_state=self.ref_state)
        
        # instantiate the parent S3_projection class with the stabilizers we are enforcing
        super().__init__(basis)
        
        return self.perform_projection(
            operator=operator_to_project
        )

    def basis_score(self, 
            basis: StabilizerOp
        ) -> float:
        """ Evaluate the score of an input basis according 
        to the basis weighting operator, for example:
            - set Hamiltonian cofficients to 1 for unweighted number of commuting terms
            - specify as the SOR Hamiltonian to weight according to second-order response
            - input UCC operator to weight according to coupled-cluster theory <- best performance
            - if None given then weights by Hamiltonian coefficient magnitude
        """
        # mask terms of the weighting operator that are preserved under projection over the basis
        mask_preserved = np.where(np.all(self.basis_weighting_operator.commutes_termwise(basis),axis=1))[0]
        return (
            np.linalg.norm(self.basis_weighting_operator.coeff_vec[mask_preserved]) /
            np.linalg.norm(self.basis_weighting_operator.coeff_vec)
            )
