from typing import Tuple, List
from cached_property import cached_property
import warnings
import numpy as np
from functools import reduce
from scipy.optimize import shgo, differential_evolution
from symmer.symplectic.base import symplectic_to_string
from symmer.symplectic.stabilizerop import find_symmetry_basis
from symmer.utils import unit_n_sphere_cartesian_coords, lp_norm
from symmer.symplectic import PauliwordOp, StabilizerOp
from symmer.projection import S3_projection
from symmer.unitary_partitioning import AntiCommutingOp

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
        self.contextual_operator = operator-self.noncontextual_operator
        if basis_weighting_operator is not None:
            self.basis_weighting_operator = basis_weighting_operator
        else:
            self.basis_weighting_operator = self.contextual_operator
        # decompose the noncontextual set into a dictionary of its 
        # universally commuting elements and anticommuting cliques
        self.symmetry_generators, self.clique_operator = self.noncontextual_basis()
        # Reconstruct the noncontextual Hamiltonian into its G and C(r) components
        self.G_indices, self.r_indices, self.pauli_mult_signs = self.noncontextual_reconstruction()
        # determine the noncontextual ground state - this updates the coefficients of the clique 
        # representative operator C(r) and symmetry generators G with the optimal configuration
        self.solve_noncontextual(ref_state)
        # Determine the unitary partitioning rotations and the single Pauli operator that is rotated onto
        if self.n_cliques > 0:
            self.unitary_partitioning_rotations, self.C0 = self.clique_operator.gen_seq_rotations()
            self.C0.coeff_vec[0] = round(self.C0.coeff_vec[0].real)
        
    def basis_score(self, 
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
        mask_preserved = np.where(np.all(self.basis_weighting_operator.commutes_termwise(basis),axis=1))[0]
        return (
            lp_norm(self.basis_weighting_operator.coeff_vec[mask_preserved], p=p) /
            lp_norm(self.basis_weighting_operator.coeff_vec, p=p)
        )
    
    def update_eigenvalues(self, stabilizers: StabilizerOp) -> None:
        """ Update the +/-1 eigenvalue assigned to the input stabilizer
        according to the noncontextual ground state configuration
        """
        reconstruction, successfully_reconstructed = stabilizers.basis_reconstruction(self.symmetry_generators)
        if reconstruction.shape[0] != len(successfully_reconstructed):
            raise ValueError('Basis not sufficient to reconstruct symmetry operators')
        stabilizers.coeff_vec = (-1) ** np.count_nonzero(
            np.bitwise_and(
                reconstruction, 
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
        # start with all the diagonal terms
        mask_diag = np.where(~np.any(self.operator.X_block, axis=1))
        noncontextual_operator = PauliwordOp(
            self.operator.symp_matrix[mask_diag],
            self.operator.coeff_vec[mask_diag])
        
        if self.noncontextual_form == 'diag':
            pass
        elif self.noncontextual_form == 'legacy':            
            # order the remaining terms by coefficient magnitude
            off_diag_terms = (self.operator - noncontextual_operator).sort(key='magnitude')
            # append terms that do not make the noncontextual_operator contextual!
            for term in off_diag_terms:
                if (noncontextual_operator+term).is_noncontextual:
                    noncontextual_operator+=term
        else:
            raise ValueError('noncontextual_form not recognised: must be one of diag or legacy.')
            
        return noncontextual_operator

    def noncontextual_basis(self) -> StabilizerOp:
        """ Find an independent basis for the noncontextual symmetry
        """
        self.decomposed = {}
        # identify a basis of universally commuting operators
        symmetry_generators = find_symmetry_basis(self.noncontextual_operator)
        # try to reconstruct the noncontextual operator in this basis
        # not all terms can be decomposed in this basis, so check which can
        reconstructed_indices, succesfully_reconstructed = self.noncontextual_operator.basis_reconstruction(symmetry_generators)
        # extract the universally commuting noncontextual terms (i.e. those which may be constructed from symmetry generators)
        universal_operator = PauliwordOp(self.noncontextual_operator.symp_matrix[succesfully_reconstructed],
                                         self.noncontextual_operator.coeff_vec[succesfully_reconstructed])
        self.decomposed['symmetry'] = universal_operator
        
        # identify the anticommuting cliques
        clique_union = self.noncontextual_operator - universal_operator
        if clique_union.n_terms != 0:
            # identify unique rows in the adjacency matrix with inverse mapping 
            # so that terms of the same clique have matching indices
            clique_characters, clique_inverse_map = np.unique(clique_union.adjacency_matrix, axis=0, return_inverse=True)
            clique_reps = []
            for i in np.unique(clique_inverse_map):
                # mask each clique and select a class represetative for its contribution in the noncontextual basis
                Ci_indices = np.where(clique_inverse_map==i)[0]
                Ci_symp,Ci_coeff = clique_union.symp_matrix[Ci_indices],clique_union.coeff_vec[Ci_indices]
                Ci_operator = PauliwordOp(Ci_symp, Ci_coeff)
                self.decomposed[f'clique_{i}'] = Ci_operator
                # choose cliques representative that maximises basis_score (summed coefficients of commuting terms)
                rep_scores = [(Ci_operator[i], self.basis_score(Ci_operator[i])) for i in range(len(Ci_coeff))]
                clique_reps.append(sorted(rep_scores, key=lambda x:-x[1])[0][0].symp_matrix)
            clique_reps = np.vstack(clique_reps)
            self.n_cliques = clique_reps.shape[0]
            clique_operator = AntiCommutingOp(clique_reps, np.ones(self.n_cliques))
        else:
            clique_operator = None
            self.n_cliques  = 0

        return symmetry_generators, clique_operator

    def noncontextual_reconstruction(self):
        """ Reconstruct the noncontextual operator in each independent basis GuCi - one for every clique.
        This mitigates against dependency between the symmetry generators G and the clique representatives Ci
        """
        if self.n_cliques > 0:
            reconstruction_ind_matrix = np.zeros(
                [self.noncontextual_operator.n_terms, self.symmetry_generators.n_terms + self.n_cliques]
            )
            # Cannot simultaneously know eigenvalues of cliques so zero rows with more than one clique
            # therefore, we decompose the noncontextual terms in the respective independent bases
            for index, Ci in enumerate(self.clique_operator):
                clique_column_index = self.symmetry_generators.n_terms+index
                col_mask_inds = np.append(
                    np.arange(self.symmetry_generators.n_terms), clique_column_index
                )
                GuCi_symp = np.vstack([self.symmetry_generators.symp_matrix, Ci.symp_matrix])
                GuCi = StabilizerOp(GuCi_symp, np.ones(GuCi_symp.shape[0]))
                reconstructed, row_mask_inds = self.noncontextual_operator.basis_reconstruction(GuCi)
                row_col_mask = np.ix_(row_mask_inds, col_mask_inds)
                reconstruction_ind_matrix[row_col_mask] = reconstructed[row_mask_inds]
        else:
            (
                reconstruction_ind_matrix, 
                succesfully_reconstructed
            ) = self.noncontextual_operator.basis_reconstruction(self.symmetry_generators)
        
        G_part = reconstruction_ind_matrix[:,:self.symmetry_generators.n_terms]
        r_part = reconstruction_ind_matrix[:,self.symmetry_generators.n_terms:]
        # individual elements of r_part commute with all of G_part - taking products over G_part with
        # a single element of r_part will therefore never produce a complex phase, but might result in
        # a sign slip that must be accounted for in the basis reconstruction TODO: add to basis_reconstruction!
        pauli_mult_signs = np.ones(self.noncontextual_operator.n_terms)
        for index, (G, r) in enumerate(zip(G_part, r_part)):
            G_inds = np.where(G!=0)[0]
            r_inds = np.where(r!=0)[0]
            G_component = self.symmetry_generators.symp_matrix[G_inds]
            if self.n_cliques > 0:
                r_component = self.clique_operator.symp_matrix[r_inds]
                all_factors_symp_matrix = np.vstack([G_component, r_component])
            else:
                all_factors_symp_matrix = G_component
            all_factors = PauliwordOp(
                all_factors_symp_matrix,
                np.ones(all_factors_symp_matrix.shape[0])
            )
            if all_factors.n_terms > 0:
                gen_mult = reduce(lambda x,y:x*y, list(all_factors))
                pauli_mult_signs[index] = int(gen_mult.coeff_vec.real[0])
        return G_part, r_part, pauli_mult_signs

    def noncontextual_objective_function(self, 
            nu: np.array, 
            r: np.array
        ) -> float:
        """ The classical objective function that encodes the noncontextual energies
        """
        G_prod = (-1)**np.count_nonzero(np.logical_and(self.G_indices==1, nu == -1), axis=1)
        r_part = np.sum(self.r_indices*r, axis=1)
        r_part[np.where(r_part==0)]=1
        return np.sum(self.noncontextual_operator.coeff_vec*G_prod*r_part*self.pauli_mult_signs).real

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
            stabilizers: StabilizerOp = None,
            enforce_clique_operator   = False,
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

        if stabilizers is not None:    
            if self.n_cliques > 0:
                # only allow stabilizers that commute with the cliques, else behaviour is unpredictable
                valid_stab_indices = np.where(
                    np.all(stabilizers.commutes_termwise(self.clique_operator), axis=1))[0]
                # raise a warning if any stabilizers are discarded due to anticommutation with a clique
                if len(valid_stab_indices) < stabilizers.n_terms:
                    invalid_stab_indices = np.setdiff1d(np.arange(stabilizers.n_terms), valid_stab_indices).tolist()
                    removed = [symplectic_to_string(stabilizers[i].symp_matrix[0]) for i in invalid_stab_indices]
                    warnings.warn(
                        'Specified a clique element in the stabilizer set!\n' +
                        f'The term(s) {removed} were discarded, but note that the number of ' +
                        'qubits in the stabilizer subspace will be greater than expected.'
                    )   
            else:
                valid_stab_indices = np.arange(stabilizers.n_terms)
            
            # instantiate as StabilizerOp to ensure algebraic independence and coefficients are +/-1
            fix_stabilizers = StabilizerOp(
                stabilizers.symp_matrix[valid_stab_indices],
                stabilizers.coeff_vec[valid_stab_indices],
                target_sqp=self.target_sqp
            )
            # update the eigenvalue assignments to the specified stabilizers 
            # in accordance with the noncontextual ground state
            self.update_eigenvalues(fix_stabilizers)
            
            # if the clique operator is to be enforced, perform unitary partitioning:
            insert_rotations=[]
            if enforce_clique_operator and self.n_cliques > 0:
                # if any stabilizers in the list contain more than one term then apply unitary partitioning
                fix_stabilizers += self.C0
                insert_rotations = self.unitary_partitioning_rotations
        elif enforce_clique_operator and self.n_cliques > 0:
            fix_stabilizers = StabilizerOp(self.C0.symp_matrix, self.C0.coeff_vec)
            insert_rotations = self.unitary_partitioning_rotations
        else:
            warnings.warn('No stabilizers were specifed so the operator was returned')
            return operator_to_project

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
            contextual_operator = operator-noncontextual_operator
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
        return np.square(
            np.linalg.norm(self.basis_weighting_operator.coeff_vec[mask_preserved]) /
            np.linalg.norm(self.basis_weighting_operator.coeff_vec)
            )
