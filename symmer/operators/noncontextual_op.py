import warnings

import numpy as np
from time import time
from functools import reduce
from typing import Optional, Union, Tuple
import multiprocessing as mp
from scipy.optimize import differential_evolution, shgo
from symmer.operators import PauliwordOp, IndependentOp, AntiCommutingOp, QuantumState
from symmer.operators.utils import unit_n_sphere_cartesian_coords, check_adjmat_noncontextual
import itertools
import qubovert as qv

class NoncontextualOp(PauliwordOp):
    """ Class for representing noncontextual Hamiltonians

    Noncontextual Hamiltonians are precisely those whose terms may be reconstructed 
    under the Jordan product (AB = {A, B}/2) from a generating set of the form 
    G ∪ {C_1, ..., C_M} where {C_i, C_j}=0 for i != j and G commutes universally.
    Refer to https://arxiv.org/abs/1904.02260 for further details. 
    
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
        """ for convenience, initialize from an existing PauliwordOp
        """
        noncontextual_operator = cls(
            H.symp_matrix,
            H.coeff_vec
        )
        return noncontextual_operator

    @classmethod
    def from_hamiltonian(cls, 
            H: PauliwordOp, 
            strategy: str = 'diag', 
            basis: PauliwordOp = None, 
            DFS_runtime: int = 10,
            override_noncontextuality_check: bool = True
        ) -> "NoncontextualOp":
        """ Given a PauliwordOp, extract from it a noncontextual sub-Hamiltonian by the specified strategy
        """
        if not override_noncontextuality_check:
            if H.is_noncontextual:
                warnings.warn('input H is already noncontextual ignoring strategy')
                return cls.from_PauliwordOp(H)
        
        if strategy == 'diag':
            return cls._diag_noncontextual_op(H)
        elif strategy == 'basis':
            return cls._from_basis_noncontextual_op(H, basis)
        elif strategy.find('DFS') != -1:
            _, strategy = strategy.split('_')
            return cls._dfs_noncontextual_op(H, strategy=strategy, runtime=DFS_runtime)
        elif strategy.find('SingleSweep') != -1:
            _, strategy = strategy.split('_')
            return cls._single_sweep_noncontextual_operator(H, strategy=strategy)
        else:
            raise ValueError(f'Unrecognised noncontextual operator strategy {strategy}')

    @classmethod
    def _diag_noncontextual_op(cls, H: PauliwordOp):
        """ Return the diagonal terms of the PauliwordOp - this is the simplest noncontextual operator
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
        """ Start from the diagonal noncontextual form and append additional off-diagonal
        contributions with respect to their coefficient magnitude.
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
        """ Order the operator by some sorting key (magnitude, random or CurrentOrder)
        and then sweep accross the terms, appending to a growing noncontextual operator
        whenever possible.
        """
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

        # initialize noncontextual operator with first element of input operator
        noncon_indices = np.array([0])
        adjmat = np.array([[True]], dtype=bool)
        for index, term in enumerate(operator[1:]):
            # pad the adjacency matrix term-by-term - avoids full construction each time
            adjmat_vector = np.append(term.commutes_termwise(operator[noncon_indices]), True)
            adjmat_padded = np.pad(adjmat, pad_width=((0, 1), (0, 1)), mode='constant')
            adjmat_padded[-1,:] = adjmat_vector; adjmat_padded[:,-1] = adjmat_vector
            # check whether the adjacency matrix has a noncontextual structure
            if check_adjmat_noncontextual(adjmat_padded):
                noncon_indices = np.append(noncon_indices, index+1)
                adjmat = adjmat_padded

        return cls.from_PauliwordOp(operator[noncon_indices])

    @classmethod
    def _from_basis_noncontextual_op(cls, H: PauliwordOp, generators: PauliwordOp):
        """ Construct a noncontextual operator given a noncontextual basis, via the Jordan product ( regular matrix product if the operators commute, and equal to zero if the operators anticommute.)
        """
        assert generators is not None, 'Must specify a noncontextual basis.'
        assert generators.is_noncontextual, 'Basis is contextual.'

        _, noncontextual_terms_mask = H.jordan_generator_reconstruction(generators)
        return cls.from_PauliwordOp(H[noncontextual_terms_mask])

    def noncontextual_basis(self) -> IndependentOp:
        """ Find an independent *generating set* for the noncontextual symmetry
        * technically not a basis!
        """
        self.decomposed = {}
        # identify a basis of universally commuting operators
        symmetry_generators = IndependentOp.symmetry_generators(self)
        # try to reconstruct the noncontextual operator in this basis
        # not all terms can be decomposed in this basis, so check which can
        reconstructed_indices, succesfully_reconstructed = self.generator_reconstruction(symmetry_generators)
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
                [self.n_terms, self.symmetry_generators.n_terms + self.n_cliques], dtype=int
            )
            # Cannot simultaneously know eigenvalues of cliques so zero rows with more than one clique
            # therefore, we decompose the noncontextual terms in the respective independent bases
            for index, Ci in enumerate(self.clique_operator):
                clique_column_index = self.symmetry_generators.n_terms+index
                col_mask_inds = np.append(
                    np.arange(self.symmetry_generators.n_terms), clique_column_index
                )
                GuCi_symp = np.vstack([self.symmetry_generators.symp_matrix, Ci.symp_matrix])
                GuCi = IndependentOp(GuCi_symp)
                reconstructed, row_mask_inds = self.generator_reconstruction(GuCi)
                row_col_mask = np.ix_(row_mask_inds, col_mask_inds)
                reconstruction_ind_matrix[row_col_mask] = reconstructed[row_mask_inds]
        else:
            (
                reconstruction_ind_matrix, 
                succesfully_reconstructed
            ) = self.generator_reconstruction(self.symmetry_generators)
        
        G_part = reconstruction_ind_matrix[:,:self.symmetry_generators.n_terms]
        r_part = reconstruction_ind_matrix[:,self.symmetry_generators.n_terms:]
        # individual elements of r_part commute with all of G_part - taking products over G_part with
        # a single element of r_part will therefore never produce a complex phase, but might result in
        # a sign slip that must be accounted for in the basis reconstruction TODO: add to generator_reconstruction!
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
        nu = np.asarray(nu, dtype=int) # must be an array!
        G_prod = (-1)**np.count_nonzero(np.logical_and(self.G_indices==1, nu == -1), axis=1)
        r_part = np.sum(self.r_indices*r, axis=1)
        r_part[~np.any(self.r_indices, axis=1)]=1
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

    def solve(self, 
            strategy: str = 'brute_force', 
            ref_state: np.array = None, 
            num_anneals:int = 1_000,
            discrete_optimization_order = 'first'
        ) -> None:
        """ Minimize the classical objective function, yielding the noncontextual ground state

        Note most QUSO functions/methods work faster than their PUSO counterparts.
        """
        
        if ref_state is not None:
            # update the symmetry generator G coefficients w.r.t. the reference state
            self.symmetry_generators.update_sector(ref_state)
            ev_assignment = self.symmetry_generators.coeff_vec
            fixed_ev_mask = ev_assignment!=0
            fixed_eigvals = (ev_assignment[fixed_ev_mask]).astype(int)
            NC_solver = NoncontextualSolver(self, fixed_ev_mask, fixed_eigvals)
            # any remaining unfixed symmetry generators are solved via other means:
        else:
            NC_solver = NoncontextualSolver(self)

        NC_solver.num_anneals = num_anneals
        NC_solver.discrete_optimization_order = discrete_optimization_order

        if strategy=='brute_force':
            self.energy, nu, r = NC_solver.energy_via_brute_force()

        elif strategy=='binary_relaxation':
            self.energy, nu, r = NC_solver.energy_via_relaxation()
        
        else:
            #### qubovert strategies below this point ####
            # PUSO = Polynomial unconstrained spin Optimization
            # QUSO: Quadratic Unconstrained Spin Optimization
            if strategy == 'brute_force_PUSO':
                NC_solver.method = 'brute_force'
                NC_solver.x = 'P'   
            elif strategy == 'brute_force_QUSO':  
                NC_solver.method = 'brute_force'
                NC_solver.x = 'Q'
            elif strategy == 'annealing_PUSO':
                NC_solver.method = 'annealing'
                NC_solver.x = 'P'
            elif strategy == 'annealing_QUSO':
                NC_solver.method = 'annealing'
                NC_solver.x = 'Q'
            else:
                raise ValueError(f'Unknown optimization strategy: {strategy}')
        
            self.energy, nu, r = NC_solver.energy_xUSO()

        # optimize the clique operator coefficients
        self.symmetry_generators.coeff_vec = nu.astype(int)
        if r is not None:
            self.clique_operator.coeff_vec = r

    def get_qaoa(self, ref_state:QuantumState=None, type='qubo') -> dict:
        """
        For a given PUBO / QUBO problem make the following replacement:

         𝑥_𝑖 <--> (𝐼−𝑍_𝑖) / 2

         This defined the QAOA Hamiltonian
        Args:
            ref_state (optional): optional QuantumState to fix symmetry generators with
        Returns:
            QAOA_dict (dict): Dictionary of different QAOA Hamiltonians from discrete r_vectors

        """
        assert type in ['qubo', 'pubo']

        # fix symm generators if reference state given
        if ref_state is not None:
            # update the symmetry generator G coefficients w.r.t. the reference state
            self.symmetry_generators.update_sector(ref_state)
            ev_assignment = self.symmetry_generators.coeff_vec
            fixed_ev_mask = ev_assignment!=0
            fixed_eigvals = (ev_assignment[fixed_ev_mask]).astype(int)
        else:
            fixed_ev_mask = np.zeros(self.symmetry_generators.n_terms, dtype=bool)
            fixed_eigvals = np.array([], dtype=int)


        fixed_indices = np.where(fixed_ev_mask)[0]  # bool to indices
        fixed_assignments = dict(zip(fixed_indices, fixed_eigvals))

        r_vec_size = self.clique_operator.n_terms
        QAOA_dict = {}
        for j in range(r_vec_size):

            ## set extreme values of r_vec
            r_vec = np.zeros(r_vec_size)
            r_vec[j]=1

            r_part = np.sum(self.r_indices * r_vec, axis=1)
            r_part[~np.any(self.r_indices, axis=1)] = 1  # set all zero terms to 1 (aka multiply be value of 1)

            # setup spin
            q_vec_SPIN = {}
            for ind in range(self.symmetry_generators.n_terms):
                if ind in fixed_assignments.keys():
                    q_vec_SPIN[ind] = fixed_assignments[ind]
                else:
                    q_vec_SPIN[ind] = qv.spin_var('x%d' % ind)


            ## setup cost function
            COST = 0
            for P_index, term in enumerate(self.G_indices):
                non_zero_inds = term.nonzero()[0]
                # collect all the spin terms
                G_term = 1
                for i in non_zero_inds:
                    G_term *= q_vec_SPIN[i]

                COST += G_term * self.coeff_vec[P_index].real * self.pauli_mult_signs[P_index] * r_part[P_index].real

            if not isinstance(COST, qv._pcso.PCSO):
                # no degrees of freedom... Cost function is just Identity term
                QAOA_dict[j] = {
                    'r_vec': r_vec,
                    'H':PauliwordOp.from_dictionary({'I'*self.n_qubits: COST}),
                    'qubo': None
                }
            else:

                # make a spin/binary problem
                if type == 'qubo':
                    QUBO_problem = COST.to_qubo()
                elif type == 'pubo':
                    QUBO_problem = COST.to_pubo()
                else:
                    raise ValueError(f'unknown tspin problem: {type}')

                # note mapping often requires more qubits!
                QAOA_n_qubits = QUBO_problem.max_index+1

                I_string = 'I' * QAOA_n_qubits
                QAOA_H = PauliwordOp.empty(QAOA_n_qubits)
                for spin_inds, coeff in QUBO_problem.items():

                    if len(spin_inds) == 0:
                        QAOA_H += PauliwordOp.from_dictionary({I_string: coeff})
                    else:
                        temp = PauliwordOp.from_dictionary({I_string: coeff})
                        for q_ind in spin_inds:
                            op = list(I_string)
                            op[q_ind] = 'Z'
                            op_str = ''.join(op)
                            Qop = PauliwordOp.from_dictionary({I_string: 0.5,
                                                               op_str: -0.5})
                            temp *= Qop

                        QAOA_H += temp

                QAOA_dict[j] = {
                    'r_vec': r_vec,
                    'H': QAOA_H.copy(),
                    'qubo': dict(QUBO_problem.items())
                }

        return QAOA_dict

###############################################################################
################### NONCONTEXTUAL SOLVERS BELOW ###############################
###############################################################################

class NoncontextualSolver:

    # xUSO settings
    method:str = 'brute_force'
    x:str = 'P'
    num_anneals:int = 1_000,
    discrete_optimization_order:str = 'first'
    reoptimize_r_vec:bool = False
    _nu = None

    def __init__(
        self,
        NC_op: NoncontextualOp,
        fixed_ev_mask: np.array = None,
        fixed_eigvals: np.array = None
        ) -> None:
        self.NC_op = NC_op
        
        if fixed_ev_mask is not None:
            assert fixed_eigvals is not None, 'Must specify the fixed eigenvalues'
            assert np.sum(fixed_ev_mask) == len(fixed_eigvals), 'Number of non-zero elements in mask does not match the number of fixed eigenvalues'
            self.fixed_ev_mask = fixed_ev_mask
            self.fixed_eigvals = fixed_eigvals
        else:
            self.fixed_ev_mask = np.zeros(NC_op.symmetry_generators.n_terms, dtype=bool)
            self.fixed_eigvals = np.array([], dtype=int)
    
    #################################################################
    ########################## BRUTE FORCE ##########################
    #################################################################

    def energy_via_brute_force(self) -> Tuple[float, np.array, np.array]:
        """ Does what is says on the tin! Try every single eigenvalue assignment in parallel
        and return the minimizing noncontextual configuration. This scales exponentially in 
        the number of qubits.
        """
        if np.all(self.fixed_ev_mask):
            nu_list = self.fixed_eigvals.reshape([1,-1])
        else:
            search_size = 2**np.sum(~self.fixed_ev_mask)
            nu_list = np.ones([search_size, self.NC_op.symmetry_generators.n_terms], dtype=int)
            nu_list[:,self.fixed_ev_mask] = np.tile(self.fixed_eigvals, [search_size,1])
            nu_list[:,~self.fixed_ev_mask] = np.array(list(itertools.product([-1,1],repeat=np.sum(~self.fixed_ev_mask))))
        
        # optimize over all discrete value assignments of nu in parallel
        with mp.Pool(mp.cpu_count()) as pool:    
            tracker = pool.map(self.NC_op._convex_problem, nu_list)
        
        # find the lowest energy eigenvalue assignment from the full list
        full_search_results = zip(tracker, nu_list)
        (energy, r_optimal), fixed_nu = min(full_search_results, key=lambda x:x[0][0])

        return energy, fixed_nu, r_optimal

    #################################################################
    ###################### BINARY RELAXATION ########################
    #################################################################

    def energy_via_relaxation(self) -> Tuple[float, np.array, np.array]:
        """ Relax the binary value assignment of symmetry generators to continuous variables
        """
        # optimize discrete value assignments nu by relaxation to continuous variables
        nu_bounds = [(0, np.pi)]*(self.NC_op.symmetry_generators.n_terms-np.sum(self.fixed_ev_mask))

        def get_nu(angles):
            """ Build nu vector given fixed values
            """
            nu = np.ones(self.NC_op.symmetry_generators.n_terms)
            nu[self.fixed_ev_mask] = self.fixed_eigvals
            nu[~self.fixed_ev_mask] = np.cos(angles)
            return nu

        optimizer_output = shgo(func=lambda angles:self.NC_op._convex_problem(get_nu(angles))[0], bounds=nu_bounds)
        # if optimization was successful the optimal angles should consist of 0 and pi
        fix_nu = np.sign(np.array(get_nu(np.cos(optimizer_output['x'])))).astype(int)
        self.NC_op.symmetry_generators.coeff_vec = fix_nu 
        energy, r_optimal = self.NC_op._convex_problem(fix_nu)
        return energy, fix_nu, r_optimal
    
    #################################################################
    ################ UNCONSTRAINED SPIN OPTIMIZATION ################
    #################################################################    
    
    def _energy_xUSO(self, r_vec: np.array) -> Tuple[float, np.array, np.array]:
        """
        Get energy via either: Polynomial unconstrained spin Optimization (x=P)
                                    or
                                Quadratic Unconstrained Spin Optimization  (x=Q)

        via a brute force search over q_vector or via simulated annealing

        Note in this method the r_vector is fixed upon input! (aka just does binary optimization)

        Args:
            NC_op (NoncontextualOp): noncontextual operator
            r_vec (np.array): array of clique expectation values <r_i>
            fixed_ev_mask (np.array): bool list of where eigenvalues in nu vector are fixed
            fixed_eigvals (np.array): list of nu eigenvalues that are fixed
            method (str): brute force or annealing optimization
            x (str): Whether method is Polynomial or Quadratic optimization
            num_anneals (optional): number of simulated anneals to do

        Returns:
            energy (float): noncontextual energy

        """
        assert self.x in ['P', 'Q']
        assert self.method in ['brute_force', 'annealing']
        
        r_part = np.sum(self.NC_op.r_indices * r_vec, axis=1)
        r_part[~np.any(self.NC_op.r_indices, axis=1)] = 1  # set all zero terms to 1 (aka multiply be value of 1)
        
        # setup spin variables
        fixed_indices = np.where(self.fixed_ev_mask)[0] # bool to indices
        fixed_assignments = dict(zip(fixed_indices, self.fixed_eigvals))
        q_vec_SPIN={}
        for ind in range(self.NC_op.symmetry_generators.n_terms):
            if ind in fixed_assignments.keys():
                q_vec_SPIN[ind] = fixed_assignments[ind]
            else:
                q_vec_SPIN[ind] = qv.spin_var('x%d' % ind)

        COST = 0
        for P_index, term in enumerate(self.NC_op.G_indices):
            non_zero_inds = term.nonzero()[0]
            # collect all the spin terms
            G_term = 1
            for i in non_zero_inds:
                G_term *= q_vec_SPIN[i]

            # cost function
            COST += G_term * self.NC_op.coeff_vec[P_index].real * self.NC_op.pauli_mult_signs[P_index] * r_part[P_index].real

        if np.all(self.fixed_ev_mask):
            # if no degrees of freedom over nu vector, COST is a number
            self._nu = self.fixed_eigvals
            return COST, self.fixed_eigvals, r_vec

        if self.x =='P':
            spin_problem = COST.to_puso()
        else:
            spin_problem = COST.to_quso()

        if self.method=='brute_force':
            sol = spin_problem.solve_bruteforce()
        elif self.method == 'annealing':
            if self.x == 'P':
                puso_res = qv.sim.anneal_puso(spin_problem, num_anneals=self.num_anneals)
            elif self.x == 'Q':
                puso_res= qv.sim.anneal_quso(spin_problem, num_anneals=self.num_anneals)
                assert COST.is_solution_valid(puso_res.best.state) is True
            sol = puso_res.best.state

        solution = COST.convert_solution(sol)
        energy = COST.value(solution)
        nu_vec = np.ones(self.NC_op.symmetry_generators.n_terms, dtype=int)
        nu_vec[self.fixed_ev_mask] = self.fixed_eigvals
        nu_vec[~self.fixed_ev_mask] = np.array(list(solution.values()))
        self._nu = nu_vec # so nu accessible during the _convex_then_xUSO optimization 

        if self.reoptimize_r_vec:
            opt_energy, opt_r_vec = self.NC_op._convex_problem(nu_vec)
            return opt_energy, nu_vec, opt_r_vec
        else:
            return energy, nu_vec, r_vec
    
    def _xUSO_then_convex(self) -> Tuple[float, np.array, np.array]:
        """
        """
        self.reoptimize_r_vec = True
        
        extreme_r_vecs = np.eye(self.NC_op.n_cliques, dtype=int)
        extreme_r_vecs = np.vstack([extreme_r_vecs, -extreme_r_vecs])
        
        with mp.Pool(mp.cpu_count()) as pool:    
            tracker = pool.map(self._energy_xUSO, extreme_r_vecs)

        return sorted(tracker, key=lambda x:x[0])[0]
    
    def _convex_then_xUSO(self) -> Tuple[float, np.array, np.array]:
        """
        """
        self.reoptimize_r_vec = False

        r_bounds = [(0, np.pi)]*(self.NC_op.n_cliques-2)+[(0, 2*np.pi)]
    
        optimizer_output = differential_evolution(
            func=lambda angles:self._energy_xUSO(
                unit_n_sphere_cartesian_coords(angles)
            )[0],
            bounds=r_bounds
        )
        optimized_energy = optimizer_output['fun']
        optimized_angles = optimizer_output['x']
        r_optimal = unit_n_sphere_cartesian_coords(optimized_angles)
        
        return optimized_energy, self._nu, r_optimal
    
    def energy_xUSO(self) -> Tuple[float, np.array, np.array]:
        """
        """
        if self.NC_op.n_cliques == 0:
            return self._energy_xUSO(None)
        elif self.discrete_optimization_order == 'first':
            return self._xUSO_then_convex()
        elif self.discrete_optimization_order == 'last':
            return self._convex_then_xUSO()
        else:
            raise ValueError('Unrecognised discrete optimization order, must be first or last')