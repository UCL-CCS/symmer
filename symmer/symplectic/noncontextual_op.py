import numpy as np
from time import time
from functools import reduce
from cached_property import cached_property
from scipy.optimize import differential_evolution, shgo
from symmer.symplectic import PauliwordOp, StabilizerOp, AntiCommutingOp
from symmer.symplectic.utils import unit_n_sphere_cartesian_coords
import itertools

class NoncontextualOp(PauliwordOp):
    """
    """
    def __init__(self,
            symp_matrix,
            coeff_vec
        ):
        """
        """
        super().__init__(symp_matrix, coeff_vec)
        assert(self.is_noncontextual), 'Specified operator is contextual.'
        self.symmetry_generators, self.clique_operator = self.noncontextual_basis()
        # Reconstruct the noncontextual Hamiltonian into its G and C(r) components
        self.noncontextual_reconstruction()
        # determine the noncontextual ground state - this updates the coefficients of the clique 
        # representative operator C(r) and symmetry generators G with the optimal configuration
        
    @classmethod
    def from_PauliwordOp(cls, H):
        """
        """
        noncontextual_operator = cls(
            H.symp_matrix,
            H.coeff_vec
        )
        return noncontextual_operator

    @classmethod
    def from_hamiltonian(cls, H, strategy='diag', DFS_runtime=10):
        if strategy == 'diag':
            return cls._diag_noncontextual_op(H)
        elif strategy.find('DFS') != -1:
            _, strategy = strategy.split('_')
            return cls._dfs_noncontextual_op(H, strategy=strategy, runtime=DFS_runtime)
        elif strategy.find('SingleSweep') != -1:
            _, strategy = strategy.split('_')
            return cls._single_sweep_noncontextual_operator(H, strategy=strategy)
        else:
            raise ValueError(f'Unrecognised noncontextual operator strategy {strategy}')

    @classmethod
    def _diag_noncontextual_op(cls, H):
        """
        """
        mask_diag = np.where(~np.any(H.X_block, axis=1))
        noncontextual_operator = cls(
            H.symp_matrix[mask_diag],
            H.coeff_vec[mask_diag]
        )
        return noncontextual_operator

    @classmethod
    def _dfs_noncontextual_op(cls, H: PauliwordOp, runtime=10, strategy='magnitude'):
        """ function orders operator by coeff mag
        then going from first term adds ops to a pauliword op ensuring it is noncontextual
        adds to a tracking list and then changes the original ordering so first term is now at the end
        repeats from the start (aka generating a list of possible noncon Hamiltonians)
        from this list one can then choose the noncon op with the most terms OR largest sum of abs coeff weights
        cutoff time ensures if the number of possibilities is large the function will STOP and not take too long

        """
        operator = H.sort(by='magnitude')
        noncontextual_ops = []

        n=0
        start_time = time()
        while n < H.n_terms and time()-start_time < runtime:
            order = np.roll(np.arange(H.n_terms), -n)
            ordered_operator = PauliwordOp(
                symp_matrix=operator.symp_matrix[order],
                coeff_vec=operator.coeff_vec[order]
            )
            noncontextual_operator = PauliwordOp.empty(H.n_qubits)
            for op in ordered_operator:
                noncon_check = noncontextual_operator + op
                if noncon_check.is_noncontextual:
                    noncontextual_operator += op
            noncontextual_ops.append(noncontextual_operator)
            n+=1

        if strategy == 'magnitude':
            noncontextual_operator = sorted(noncontextual_ops, key=lambda x:-np.sum(abs(x.coeff_vec)))[0]
        elif strategy == 'largest':
            noncontextual_operator = sorted(noncontextual_ops, key=lambda x:-x.n_terms)[0]
        else:
            raise ValueError('Unrecognised noncontextual operator strategy.')

        return cls.from_PauliwordOp(noncontextual_operator)

    @classmethod
    def _diag_first_noncontextual_op(cls, H: PauliwordOp):
        """
        """
        noncontextual_operator = cls._diag_noncontextual_op(H)
        # order the remaining terms by coefficient magnitude
        off_diag_terms = (H - noncontextual_operator).sort(by='magnitude')
        # append terms that do not make the noncontextual_operator contextual!
        for term in off_diag_terms:
            if (noncontextual_operator+term).is_noncontextual:
                noncontextual_operator+=term
        
        return cls.from_PauliwordOp(noncontextual_operator)

    @classmethod
    def _single_sweep_noncontextual_operator(cls, H, strategy='magnitude'):
        """
        """
        noncontextual_operator = PauliwordOp.empty(H.n_qubits)
        
        if strategy=='magnitude':
            operator = H.sort(by='magnitude')
        elif strategy=='random':
            order = np.arange(H.n_terms)
            np.random.shuffle(order)
            operator = PauliwordOp(
                H.symp_matrix[order],
                H.coeff_vec[order]
            )
        elif strategy =='CurrentOrder':
            operator = H
        else:
            raise ValueError('Unrecognised strategy, must be one of magnitude, random or CurrentOrder')            

        for op in operator:
            test = noncontextual_operator + op
            if test.is_noncontextual:
                noncontextual_operator += op
        
        return cls.from_PauliwordOp(noncontextual_operator)

    def noncontextual_basis(self) -> StabilizerOp:
        """ Find an independent basis for the noncontextual symmetry
        """
        self.decomposed = {}
        # identify a basis of universally commuting operators
        symmetry_generators = StabilizerOp.symmetry_basis(self)
        # try to reconstruct the noncontextual operator in this basis
        # not all terms can be decomposed in this basis, so check which can
        reconstructed_indices, succesfully_reconstructed = self.basis_reconstruction(symmetry_generators)
        # extract the universally commuting noncontextual terms (i.e. those which may be constructed from symmetry generators)
        universal_operator = PauliwordOp(self.symp_matrix[succesfully_reconstructed],
                                         self.coeff_vec[succesfully_reconstructed])
        self.decomposed['symmetry'] = universal_operator
        # identify the anticommuting cliques
        clique_union = self - universal_operator
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
                clique_reps.append(Ci_operator.symp_matrix[0])
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
                [self.n_terms, self.symmetry_generators.n_terms + self.n_cliques]
            )
            # Cannot simultaneously know eigenvalues of cliques so zero rows with more than one clique
            # therefore, we decompose the noncontextual terms in the respective independent bases
            for index, Ci in enumerate(self.clique_operator):
                clique_column_index = self.symmetry_generators.n_terms+index
                col_mask_inds = np.append(
                    np.arange(self.symmetry_generators.n_terms), clique_column_index
                )
                GuCi_symp = np.vstack([self.symmetry_generators.symp_matrix, Ci.symp_matrix])
                GuCi = StabilizerOp(GuCi_symp)
                reconstructed, row_mask_inds = self.basis_reconstruction(GuCi)
                row_col_mask = np.ix_(row_mask_inds, col_mask_inds)
                reconstruction_ind_matrix[row_col_mask] = reconstructed[row_mask_inds]
        else:
            (
                reconstruction_ind_matrix, 
                succesfully_reconstructed
            ) = self.basis_reconstruction(self.symmetry_generators)
        
        G_part = reconstruction_ind_matrix[:,:self.symmetry_generators.n_terms]
        r_part = reconstruction_ind_matrix[:,self.symmetry_generators.n_terms:]
        # individual elements of r_part commute with all of G_part - taking products over G_part with
        # a single element of r_part will therefore never produce a complex phase, but might result in
        # a sign slip that must be accounted for in the basis reconstruction TODO: add to basis_reconstruction!
        pauli_mult_signs = np.ones(self.n_terms)
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
        self.G_indices, self.r_indices, self.pauli_mult_signs = G_part, r_part, pauli_mult_signs

    def noncontextual_objective_function(self, 
            nu: np.array, 
            r: np.array
        ) -> float:
        """ The classical objective function that encodes the noncontextual energies
        """
        nu = np.asarray(nu) # must be an array!
        G_prod = (-1)**np.count_nonzero(np.logical_and(self.G_indices==1, nu == -1), axis=1)
        r_part = np.sum(self.r_indices*r, axis=1)
        r_part[np.where(r_part==0)]=1
        return np.sum(self.coeff_vec*G_prod*r_part*self.pauli_mult_signs).real

    def _convex_problem(self, nu):
        """ given +/-1 value assignments nu, solve for the clique operator coefficients.
        Note that, with nu fixed, the optimization problem is now convex.
        """
        if self.n_cliques==0:
            optimized_energy = self.noncontextual_objective_function(nu=nu, r=None)
            r_optimal = None
        else:
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

    def _energy_via_ref_state(self, ref_state):
        """
        """
        # update the symmetry generator G coefficients w.r.t. the reference state
        self.symmetry_generators.update_sector(ref_state=ref_state)
        fix_nu = self.symmetry_generators.coeff_vec
        energy, r_optimal = self._convex_problem(fix_nu)
        return energy, fix_nu, r_optimal

    def _energy_via_relaxation(self):
        """
        """
        # optimize discrete value assignments nu by relaxation to continuous variables
        nu_bounds = [(0, np.pi)]*self.symmetry_generators.n_terms
        optimizer_output = shgo(func=lambda angles:self._convex_problem(np.cos(angles))[0], bounds=nu_bounds)
        # if optimization was successful the optimal angles should consist of 0 and pi
        fix_nu = np.sign(np.array(np.cos(optimizer_output['x']))).astype(int)
        self.symmetry_generators.coeff_vec = fix_nu 
        energy, r_optimal = self._convex_problem(fix_nu)
        return energy, fix_nu, r_optimal

    def _energy_via_brute_force(self):
        # optimize over all discrete value assignments of nu
        tracker = []
        for nu in itertools.product([-1,1],repeat=self.symmetry_generators.n_terms):
            energy, r = self._convex_problem(np.array(nu))
            tracker.append((energy, r, np.array(nu)))
        energy, r_optimal, fix_nu = min(tracker, key=lambda x:x[0])
        return energy, fix_nu, r_optimal

    def solve(self, strategy='binary_relaxation', ref_state: np.array = None) -> None:
        """ Minimize the classical objective function, yielding the noncontextual ground state
        """
        if ref_state is not None:
            self.energy, nu, r = self._energy_via_ref_state(ref_state)
        elif strategy=='binary_relaxation' :
            self.energy, nu, r = self._energy_via_relaxation()
        elif strategy=='brute_force' :
            self.energy, nu, r = self._energy_via_brute_force()
        else:
            raise ValueError(f'unknown optimization strategy: {strategy}')
        
        # optimize the clique operator coefficients
        self.symmetry_generators.coeff_vec = nu
        if r is not None:
            self.clique_operator.coeff_vec = r