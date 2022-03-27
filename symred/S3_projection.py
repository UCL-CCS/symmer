# general imports
import numpy as np
from scipy.optimize import shgo, differential_evolution
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from functools import reduce
from cached_property import cached_property
# specialized imports
from symred.symplectic_form import PauliwordOp, StabilizerOp, symplectic_to_string
from symred.utils import (
    gf2_gaus_elim, 
    gf2_basis_for_gf2_rref,
    heavy_gaussian_elimination,
    unit_n_sphere_cartesian_coords,
    quasi_model
    )

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

class S3_projection:
    """ Base class for enabling qubit reduction techniques derived from
    the Stabilizer SubSpace (S3) projection framework, such as tapering
    and Contextual-Subspace VQE. The methods defined herein serve the 
    following purposes:

    - _perform_projection
        Assuming the input operator has been rotated via the Clifford operations 
        found in the above stabilizer_rotations method, this will effect the 
        projection onto the corresponding stabilizer subspace. This involves
        droping any operator terms that do not commute with the rotated generators
        and fixing the eigenvalues of those that do consistently.
    - perform_projection
        This method wraps _perform_projection but provides the facility to insert
        auxiliary rotations (that need not be Clifford). This is used in CS-VQE
        to implement unitary partitioning where necessary. 
    """
    rotated_flag = False

    def __init__(self,
                stabilizers: StabilizerOp, 
                target_sqp: str = 'Z'
                ) -> None:
        """
        - stabilizers
            a list of stabilizers that should be enforced, given as Pauli strings
        - eigenvalues
            the list of eigenvalue assignments to complement the stabilizers
        - target_sqp
            the target single-qubit Pauli (X or Z) that we wish to rotate onto
        - fix_qubits
            Manually overrides the qubit positions selected in stabilizer_rotations, 
            although the rotation procedure can be a bit unpredictable so take care!
        """
        self.stabilizers = stabilizers
        self.target_sqp = target_sqp
    
    def _perform_projection(self, 
            operator: PauliwordOp,
            #sym_sector: Union[List[int], np.array]
        ) -> PauliwordOp:
        """ method for projecting an operator over fixed qubit positions 
        stabilized by single Pauli operators (obtained via Clifford operations)
        """
        assert(operator.n_qubits == self.stabilizers.n_qubits), 'The input operator does not have the same number of qubits as the stabilizers'
        assert(self.rotated_flag), 'The operator has not been rotated - intended for use with perform_projection method'
        self.rotated_flag = False
        
        # remove terms that do not commute with the rotated stabilizers
        commutes_with_all_stabilizers = np.all(operator.commutes_termwise(self.rotated_stabilizers), axis=1)
        op_anticommuting_removed = operator.symp_matrix[commutes_with_all_stabilizers]
        cf_anticommuting_removed = operator.coeff_vec[commutes_with_all_stabilizers]

        # determine sign flipping from eigenvalue assignment
        # currently ill-defined for single-qubit Y stabilizers
        stab_symp_indices  = np.where(self.rotated_stabilizers.symp_matrix)[1]
        eigval_assignment = op_anticommuting_removed[:,stab_symp_indices]*self.rotated_stabilizers.coeff_vec
        eigval_assignment[eigval_assignment==0]=1 # 0 entries are identity, so fix as 1 in product
        coeff_sign_flip = cf_anticommuting_removed*(np.prod(eigval_assignment, axis=1)).T

        # the projected Pauli terms:
        unfixed_XZ_indices = np.hstack([self.free_qubit_indices,
                                        self.free_qubit_indices+operator.n_qubits])
        projected_symplectic = op_anticommuting_removed[:,unfixed_XZ_indices]

        # there may be duplicate rows in op_projected - these are identified and
        # the corresponding coefficients collected in the cleanup method
        projected_operator = PauliwordOp(projected_symplectic, coeff_sign_flip).cleanup()
        
        return projected_operator
            
    def perform_projection(self,
            operator: PauliwordOp,
            ref_state: Union[List[int], np.array]=None,
            sector: Union[List[int], np.array]=None,
            insert_rotations:List[Tuple[str, float]]=[]
        ) -> PauliwordOp:
        """ Input a PauliwordOp and returns the reduced operator corresponding 
        with the specified stabilizers and eigenvalues.
        
        insert_rotation allows one to include supplementary Pauli rotations
        to be performed prior to the stabilizer rotations, for example 
        unitary partitioning in CS-VQE
        """
        if sector is None and ref_state is not None:
            #assert(ref_state is not None), 'If no sector is provided then a reference state must be given instead'
            self.stabilizers.update_sector(ref_state)
        elif sector is not None:
            self.stabilizers.coeff_vec = np.array(sector, dtype=int)

        self.rotated_stabilizers = self.stabilizers.rotate_onto_single_qubit_paulis()
        self.stab_qubit_indices  = np.where(self.rotated_stabilizers.symp_matrix)[1] % operator.n_qubits
        self.free_qubit_indices  = np.setdiff1d(np.arange(operator.n_qubits),self.stab_qubit_indices)

        # insert any supplementary rotations coming from the child class
        stab_rotations = insert_rotations + self.stabilizers.stabilizer_rotations

        # perform the full list of rotations on the input operator...
        if stab_rotations != []:
            op_rotated = operator.recursive_rotate_by_Pword(stab_rotations)
        else:
            op_rotated = operator
        
        self.rotated_flag = True
        # ...and finally perform the stabilizer subspace projection
        return self._perform_projection(operator=op_rotated)

class QubitTapering(S3_projection):
    """ Class for performing qubit tapering as per https://arxiv.org/abs/1701.08213.
    Reduces the number of qubits in the problem whilst preserving its energy spectrum by:

    1. identifying a symmetry of the Hamiltonian,
    2. finding an independent basis therein,
    3. rotating each basis operator onto a single Pauli X, 
    4. dropping the corresponding qubits from the Hamiltonian whilst
    5. fixing the +/-1 eigenvalues

    Steps 1-2 are handled in this class whereas we defer to the parent S3_projection for 3-5.

    """
    def __init__(self,
            operator: PauliwordOp, 
            target_sqp: str = 'X'
        ) -> None:
        """ Input the PauliwordOp we wish to taper.
        There is freedom over the choice of single-qubit Pauli operator we wish to rotate onto, 
        however this is set to X by default (in line with the original tapering paper).
        """
        self.operator = operator
        self.n_taper = self.symmetry_generators.n_terms
        super().__init__(self.symmetry_generators, target_sqp=target_sqp)
        
    @cached_property
    def symmetry_generators(self) -> StabilizerOp:
        """ Find an independent basis for the input operator symmetry
        """
        # swap order of XZ blocks in symplectic matrix to ZX
        ZX_symp = np.hstack([self.operator.Z_block, self.operator.X_block])
        reduced = gf2_gaus_elim(ZX_symp)
        kernel  = gf2_basis_for_gf2_rref(reduced)

        return StabilizerOp(kernel, np.ones(kernel.shape[0]))

    def taper_it(self,
            ref_state: Union[List[int], np.array]=None,
            sector: Union[List[int], np.array]=None,
            aux_operator: PauliwordOp = None
        ) -> PauliwordOp:
        """ Finally, once the symmetry generators and sector have been identified, 
        we may perform a projection onto the corresponding stabilizer subspace via 
        the parent S3_projection class.

        This method allows one to input an auxiliary operator other than the internal
        operator itself to be tapered consistently with the identified symmetry. This is 
        especially useful when considering an Ansatz defined over the full system that 
        one wishes to restrict to the same stabilizer subspace as the Hamiltonian for 
        use in VQE, for example.
        """
        # allow an auxiliary operator (e.g. an Ansatz) to be tapered
        if aux_operator is not None:
            operator_to_taper = aux_operator.copy()
        else:
            operator_to_taper = self.operator.copy()

        # taper the operator via S3_projection.perform_projection
        tapered_operator = self.perform_projection(
            operator=operator_to_taper,
            ref_state=ref_state,
            sector=sector
        )

        # if a reference state was supplied, taper it by dropping any
        # qubit positions fixed during the perform_projection method
        if ref_state is not None:
            ref_state = np.array(ref_state)
            self.tapered_ref_state = ref_state[self.free_qubit_indices]

        return tapered_operator

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



