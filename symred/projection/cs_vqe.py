import numpy as np
from scipy.optimize import shgo, differential_evolution
from symred.symplectic import PauliwordOp, StabilizerOp, symplectic_to_string
from symred.projection import S3_projection
from symred.utils import unit_n_sphere_cartesian_coords
from typing import Tuple, Dict, List
from copy import deepcopy
from functools import reduce, cached_property

def unitary_partitioning_rotations(AC_op: PauliwordOp) -> List[Tuple[str,float]]:
    """ Perform unitary partitioning as per https://doi.org/10.1103/PhysRevA.101.062322 (Section A)
    Note unitary paritioning only works when the terms are mutually anticommuting
    """
    # check the terms are mutually anticommuting to avoid an infinite loop:
    assert(
        np.all(
            np.array(AC_op.adjacency_matrix, dtype=int)
                      ==np.eye(AC_op.n_terms, AC_op.n_terms)
        )
    ), 'Operator terms are not mutually anticommuting'
    
    rotations = []
    
    def _recursive_unitary_partitioning(AC_op: PauliwordOp) -> None:
        """ Always retains the first term of the operator, deletes the second 
        term at each level of recursion and reads out the necessary rotations
        """
        if AC_op.n_terms == 1:
            return None
        else:
            op_for_rotation = AC_op.copy()
            A0, A1 = op_for_rotation[0], op_for_rotation[1]
            angle = np.arctan(A1.coeff_vec / A0.coeff_vec)
            # set coefficients to 1 since we only want to track sign flip from here
            A0.coeff_vec, A1.coeff_vec = [1], [1]
            pauli_rot = (A0 * A1).multiply_by_constant(-1j)
            angle*=pauli_rot.coeff_vec
            # perform the rotation, thus deleting a single term from the input operator
            AC_op_rotated = op_for_rotation._rotate_by_single_Pword(pauli_rot, angle).cleanup_zeros()
            
            # append the rotation to list
            rotations.append((symplectic_to_string(pauli_rot.symp_matrix[0]), angle.real[0]))
            
            return _recursive_unitary_partitioning(AC_op_rotated)
    
    _recursive_unitary_partitioning(AC_op)
    
    return rotations

class CS_VQE(S3_projection):
    """
    """
    def __init__(self,
            operator: PauliwordOp,
            ref_state: np.array = None,
            target_sqp: str = 'Z',
            basis_weighting_operator: PauliwordOp = None
        ) -> None:
        """ 
        """
        self.operator = operator
        self.ref_state = ref_state
        self.target_sqp = target_sqp
        if basis_weighting_operator is not None:
            self.basis_weighting_operator = basis_weighting_operator
        else:
            self.basis_weighting_operator = operator
        self.contextual_operator = (operator-self.noncontextual_operator).cleanup_zeros()
        # decompose the noncontextual set into a dictionary of its 
        # universally commuting elements and anticommuting cliques
        self.noncontextual_reconstruction = (
            self.noncontextual_operator.basis_reconstruction(self.noncontextual_basis)
        )
        self.r_indices = self.noncontextual_reconstruction[:,:self.n_cliques]
        self.G_indices = self.noncontextual_reconstruction[:,self.n_cliques:]
        self.clique_operator = (self.noncontextual_basis[:self.n_cliques]).sort(key='Z')
        symmetry_generators_symp = self.noncontextual_basis.symp_matrix[self.n_cliques:]
        self.symmetry_generators = StabilizerOp(
            symmetry_generators_symp,
            np.ones(symmetry_generators_symp.shape[0])
        )
        # determine the noncontextual ground state - this updates the coefficients of the clique 
        # representative operator C(r) and symmetry generators G with the optimal configuration
        self.solve_noncontextual(ref_state)

    @cached_property
    def noncontextual_operator(self) -> PauliwordOp:
        """ Extract a noncontextual set of Pauli terms from the operator

        Implementation of the algorithm in https://doi.org/10.1103/PhysRevLett.123.200501
        Does a single pass over the Hamiltonian and appends terms to noncontextual_operator 
        that do not make it contextual - easy to do multiple passes although this does not 
        seem to yield better results from experimentation.

        TODO graph-based approach, currently uses legacy implementation
        """
        # order the operator terms by coefficient magnitude
        check_ops = self.operator.sort(key='magnitude')
        # initialise as identity with 0 coefficient
        I_symp = np.zeros(2*self.operator.n_qubits, dtype=int)
        noncontextual_operator = PauliwordOp(I_symp, [0])
        for i in range(check_ops.n_terms):
            if (noncontextual_operator+check_ops[i]).is_noncontextual:
                noncontextual_operator+=check_ops[i]
        return noncontextual_operator

    @cached_property
    def noncontextual_basis(self) -> StabilizerOp:
        """ Find an independent basis for the noncontextual symmetry
        """
        ham_nc_dict = {op:float(coeff) for op,coeff in self.noncontextual_operator.to_dictionary.items()}
        ham_nc_dict = dict(sorted(ham_nc_dict.items(), key=lambda x:-abs(x[1])))
        noncon_model = quasi_model(ham_nc_dict)
        G, clique_reps = noncon_model[0], noncon_model[1]

        basis = PauliwordOp(G, np.ones(len(G)))
        basis = basis + PauliwordOp(clique_reps, np.ones(len(clique_reps)))
        basis_order = np.lexsort(basis.adjacency_matrix)
        basis = StabilizerOp(basis.symp_matrix[basis_order],np.ones(basis.n_terms))
        self.n_cliques = np.count_nonzero(~np.all(basis.adjacency_matrix, axis=1))
        
        return basis

        self.decomposed = {}
        # extract the universally commuting noncontextual terms
        universal_mask = np.where(np.all(self.noncontextual_operator.adjacency_matrix, axis=1))
        universal_operator = PauliwordOp(self.noncontextual_operator.symp_matrix[universal_mask],
                                         self.noncontextual_operator.coeff_vec[universal_mask])
        self.decomposed['symmetry'] = universal_operator
        # identify the anticommuting cliques
        clique_union = (self.noncontextual_operator - universal_operator).cleanup_zeros()
        # order lexicographically and take difference between adjacent rows
        clique_grouping_order = np.lexsort(clique_union.adjacency_matrix.T)
        diff_adjacent = np.diff(clique_union.adjacency_matrix[clique_grouping_order], axis=0)
        # the unique cliques are the non-zero rows in diff_adjacent
        mask_unique_cliques = np.append(True, ~np.all(diff_adjacent==0, axis=1))
        # determine the inverse mapping so terms of the same clique have the same index
        inverse_index = np.zeros_like(clique_grouping_order)
        inverse_index[clique_grouping_order] = np.cumsum(mask_unique_cliques) - 1
        mask_cliques = np.stack([np.where(inverse_index==i)[0] for i in np.unique(inverse_index)])
        # mask each clique and select a class represetative for its contribution in the noncontextual basis
        clique_reps = []
        for i, (Ci_symp, Ci_coef) in enumerate(
            zip(
                clique_union.symp_matrix[mask_cliques],
                clique_union.coeff_vec[mask_cliques]
            )
        ):
            Ci_operator = PauliwordOp(Ci_symp, Ci_coef)
            self.decomposed[f'clique_{i}'] = Ci_operator
            # choose cliques representative that maximises basis_score
            rep_scores = [(Ci_operator[i], self.basis_score(Ci_operator[i])) for i in range(len(Ci_coef))]
            clique_reps.append(sorted(rep_scores, key=lambda x:-x[1])[0][0].symp_matrix)

        # now we are ready to build the noncontextual basis...
        # perform partial Gaussian elimination on the symmetry terms - the resulting
        # symmetry generators are heavy in the sense that they have large support
        reduced_universal = heavy_gaussian_elimination(universal_operator.symp_matrix)
        basis = PauliwordOp(reduced_universal, np.ones(reduced_universal.shape[0]))
        
        #for i in range(basis.n_terms-1):
        #    trials=[]
        #    for j in range(i+1, basis.n_terms):
        #        test_basis = basis[:i] + basis[i+1:]
        #        test_basis += basis[i] * basis[j]
        #        trials.append([test_basis, self.basis_score(test_basis)])
        #    basis = sorted(trials, key=lambda x:-x[1])[0][0]
        
        # combine the symmetry generators and clique representatives to form the noncontextual basis
        #basis_symp = np.vstack(clique_reps+[reduced_universal])
        #basis = PauliwordOp(basis_symp, np.ones(basis_symp.shape[0]))
        clique_reps = np.vstack(clique_reps)
        basis = basis + PauliwordOp(clique_reps, np.ones(clique_reps.shape[0]))
        basis_order = np.lexsort(basis.adjacency_matrix)
        basis = StabilizerOp(basis.symp_matrix[basis_order],np.ones(basis.n_terms))
        self.n_cliques = np.count_nonzero(~np.all(basis.adjacency_matrix, axis=1))
        
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
        
        # optimize the clique operator coefficients
        fix_nu = self.symmetry_generators.coeff_vec
        self.noncontextual_energy, r = convex_problem(fix_nu)
        self.clique_operator.coeff_vec = r    
        
    def contextual_subspace_projection(self,
            stabilizer_indices: List[int],
            aux_operator: PauliwordOp = None
        ) -> PauliwordOp:
        """ input a list indexing the stabilizers one wishes to enforce
        index 0 always corresponds to the clique operator C(r)
        """
        if aux_operator is not None:
            operator_to_project = aux_operator.copy()
        else:
            operator_to_project = self.operator.copy()

        # from the supplied indices determine the corresponding stabilizers to enforce
        stab_indices = deepcopy(stabilizer_indices)
        if 0 in stab_indices:
            # in this case apply the unitary partitioning rotations
            stab_indices.pop(stab_indices.index(0))
            stab_indices = [i-1 for i in stab_indices]
            UP_rotations = unitary_partitioning_rotations(self.clique_operator)
            rotated_clique_op = self.clique_operator.recursive_rotate_by_Pword(UP_rotations).cleanup_zeros(zero_threshold=1e-15)
            rotated_clique_op.coeff_vec=np.round(rotated_clique_op.coeff_vec)
            fix_stabilizers = reduce(lambda x,y: x+y,
                [rotated_clique_op]+[self.symmetry_generators[i] for i in stab_indices])
            insert_rotations = UP_rotations
        else:
            stab_indices = [i-1 for i in stab_indices]
            fix_stabilizers = reduce(lambda x,y: x+y,
                [self.symmetry_generators[i] for i in stab_indices])
            insert_rotations = []

        # instantiate as StabilizerOp to ensure algebraic independence and coefficients are +/-1
        fix_stabilizers = StabilizerOp(
            fix_stabilizers.symp_matrix, 
            np.array(fix_stabilizers.coeff_vec, dtype=int),
            target_sqp=self.target_sqp
        )
        # instantiate the parent S3_projection classwith the stabilizers we are enforcing
        super().__init__(fix_stabilizers, target_sqp=self.target_sqp)

        return self.perform_projection(
            operator=operator_to_project,
            insert_rotations=insert_rotations
        )

    def basis_score(self, 
            basis: StabilizerOp
        ) -> float:
        """ Evaluate the score of an input basis according 
        to the basis weighting operator, for example:
            - set Hamiltonian cofficients to 1 for unweighted number of commuting terms
            - specify as the SOR Hamiltonian to weight according to second-order response
            - if None given then weights by Hamiltonian coefficient magnitude
        """
        return np.sqrt(
            np.sum(
                np.square(
                    self.basis_weighting_operator.coeff_vec[
                        np.all(self.basis_weighting_operator.commutes_termwise(basis), axis=1)
                        ]
                )
            )
        )
        
class CheatS_VQE(S3_projection):
    """ A lightweight CS-VQE implementation in which we choose an arbitrary
    Pauli Z-basis and project in accordance with it. Can be interpretted as
    CS-VQE with the noncontextual set taken as the diagonal Hamiltonian terms.

    Second-order-response-corrected VQE... identify the independent Z-basis
    that maxises the SOR objective function
    """
    def __init__(self, 
            operator: PauliwordOp,
            ref_state: np.array,
            target_sqp: str = 'Z'):
        self.operator = operator
        self.ref_state = ref_state
        self.target_sqp = target_sqp        

    def project_onto_subspace(self, basis: StabilizerOp):
        """ Project the operator in accordance with the supplied basis
        """
        basis = StabilizerOp(
            basis.symp_matrix, 
            np.ones(basis.n_terms, dtype=int), 
            target_sqp=self.target_sqp
        )
        basis.update_sector(ref_state=self.ref_state)
        super().__init__(basis, target_sqp=self.target_sqp)
        
        return self.perform_projection(
            operator=self.operator.copy()
        )

    def weighted_objective(self, 
            basis: StabilizerOp, 
            aux_operator: PauliwordOp = None
        ) -> float:
        """ Evaluate the score of the input basis according to some operator
        e.g. specify the aux_operator as the SOR Hamiltonian
        """
        if aux_operator is not None:
            weighted_op = aux_operator
        else:
            weighted_op = self.operator
        return np.sqrt(
            np.sum(
                np.square(
                    weighted_op.coeff_vec[np.all(weighted_op.commutes_termwise(basis), axis=1)]
                        )
                    )
                )
