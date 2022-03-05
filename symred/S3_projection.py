from functools import cached_property, reduce
from shutil import ExecError
from symred.symplectic_form import PauliwordOp, StabilizerOp, symplectic_to_string
from symred.utils import gf2_gaus_elim, gf2_basis_for_gf2_rref
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Union
import numpy as np

class S3_projection:
    """ Base class for enabling qubit reduction techniques derived from
    the Stabilizer SubSpace (S3) projection framework, such as tapering
    and Contextual-Subspace VQE. The methods defined herein serve the 
    following purposes:

    - stabilizer_rotations
        This method determines a sequence of Clifford rotations mapping the
        provided stabilizers onto single-qubit Paulis (sqp), either X or Z
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
        # store stabilizers and their assignments as PauliwordOp object
        # this facilitates various manipulations such as Pauli rotations
      
        self.stabilizers = stabilizers
        self.target_sqp = target_sqp
        try:
            self.rotations, self.angles = zip(*self.stabilizer_rotations)
            self.rotated_stabilizers = self.stabilizers.recursive_rotate_by_Pword(
                pauli_rot_list=self.rotations,
                angles=self.angles
            )
        except:
            # in the case that the input stabilizers are already single-qubit Pauli operators
            self.rotations, self.angles = [], []
            self.rotated_stabilizers = self.stabilizers
    

    @cached_property
    def stabilizer_rotations(self) -> List[Tuple[str, float]]:
        """ 
        Implementation of procedure described in https://doi.org/10.22331/q-2021-05-14-456 (Lemma A.2)
        
        Returns 
        - a dictionary of stabilizers with the rotations mapping each to a 
          single Pauli in the formList[Tuple[rotation, angle, gen_rot]], 
        
        - a dictionary of qubit positions that we have rotated onto and 
          the eigenvalues post-rotation
        """
        stabilizer_ref = self.stabilizers.copy()
        rotations=[]

        def append_rotation(base_pauli: np.array, index: int) -> str:
            """ force the indexed qubit to a Pauli Y in the base Pauli
            """
            X_index = index % self.stabilizers.n_qubits # index in the X block
            base_pauli[np.array([X_index, X_index+self.stabilizers.n_qubits])]=1
            base_pauli = symplectic_to_string(base_pauli)
            # None angle defaults to pi/2 for Clifford rotation
            rotations.append((base_pauli, None))
            # return the pauli rotation to update stabilizer_ref as we go
            return base_pauli

        # This part produces rotations onto single-qubit Paulis (sqp) - might be a combination of X and Z
        # while loop active until each row of symplectic matrix contains a single non-zero element
        while np.any(~(np.count_nonzero(stabilizer_ref.symp_matrix, axis=1)==1)):
            unique_position = np.where(np.count_nonzero(stabilizer_ref.symp_matrix, axis=0)==1)[0]
            reduced = stabilizer_ref.symp_matrix[:,unique_position]
            unique_stabilizer = np.where(np.any(reduced, axis=1))
            for row in stabilizer_ref.symp_matrix[unique_stabilizer]:
                if np.count_nonzero(row) != 1:
                    # find the free indices and pick one (there is some freedom over this)
                    available_positions = np.intersect1d(unique_position, np.where(row))
                    pauli_rotation = PauliwordOp([append_rotation(row.copy(), available_positions[0])], [1])
                    # update the stabilizers by performing the rotation
                    stabilizer_ref = stabilizer_ref._rotate_by_single_Pword(pauli_rotation)

        # This part produces rotations onto the target sqp
        for row in stabilizer_ref.symp_matrix:
            sqp_index = np.where(row)[0]
            if ((self.target_sqp == 'Z' and sqp_index< self.stabilizers.n_qubits) or 
                (self.target_sqp == 'X' and sqp_index>=self.stabilizers.n_qubits)):
                pauli_rotation = append_rotation(np.zeros(2*self.stabilizers.n_qubits, dtype=int), sqp_index)

        return rotations

    def _perform_projection(self, 
            operator: PauliwordOp,
            sym_sector: Union[List[int], np.array]
        ) -> PauliwordOp:
        """ method for projecting an operator over fixed qubit positions 
        stabilized by single Pauli operators (obtained via Clifford operations)
        """
        assert(operator.n_qubits == self.stabilizers.n_qubits), 'The input operator does not have the same number of qubits as the stabilizers'
        if not self.rotated_flag:
            raise ExecError('The operator has not been rotated - intended for use with perform_projection method')
        self.rotated_flag = False

        # overwrite the coefficient vector to the assigned eigenvalues defined by the symmetry sector
        rotated_stabilizers = self.rotated_stabilizers.copy()
        rotated_stabilizers.coeff_vec*=np.array(sym_sector, dtype=int)
        stab_positions = np.einsum("ij->j",rotated_stabilizers.symp_matrix)
        stab_q_indices = np.where(stab_positions)[0]
        assert(len(stab_q_indices)== rotated_stabilizers.n_terms), 'unique indices and stabilizers do not match'

        # remove terms that do not commute with the rotated stabilizers
        commutes_with_all_stabilizers = np.all(operator.commutes_termwise(rotated_stabilizers), axis=1)
        op_anticommuting_removed = operator.symp_matrix[commutes_with_all_stabilizers]
        cf_anticommuting_removed = operator.coeff_vec[commutes_with_all_stabilizers]

        # determine sign flipping from eigenvalue assignment
        # currently ill-defined for single-qubit Y stabilizers
        eigval_assignment = op_anticommuting_removed[:,stab_q_indices]*rotated_stabilizers.coeff_vec
        eigval_assignment[eigval_assignment==0]=1 # 0 entries are identity, so fix as 1 in product
        coeff_sign_flip = cf_anticommuting_removed*(np.prod(eigval_assignment, axis=1)).T

        # the projected Pauli terms:
        all_qubits = np.arange(operator.n_qubits)
        unfixed_positions = np.setdiff1d(all_qubits,stab_q_indices % operator.n_qubits)
        unfixed_positions = np.hstack([ unfixed_positions,
                                        unfixed_positions+operator.n_qubits])
        project_symplectic = op_anticommuting_removed[:,unfixed_positions]

        # there may be duplicate rows in op_projected - these are identified and
        # the corresponding coefficients collected in the cleanup method
        project_operator = PauliwordOp(project_symplectic, coeff_sign_flip).cleanup()
        
        return project_operator
            
    def perform_projection(self,
            operator: PauliwordOp,
            sym_sector: Union[List[int], np.array],
            insert_rotation:Tuple[str,float]=None
        ) -> PauliwordOp:
        """ Input a PauliwordOp and returns the reduced operator corresponding 
        with the specified stabilizers and eigenvalues.
        
        insert_rotation allows one to include supplementary Pauli rotations
        to be performed prior to the stabilizer rotations, for example 
        unitary partitioning in CS-VQE
        """
        stab_rotations = list(self.rotations)
        angles = list(self.angles)
        # ...and insert any supplementary ones coming from the child class
        if insert_rotation is not None:
            stab_rotations.insert(0, insert_rotation[0])
            angles.insert(0, insert_rotation[1])

        # perform the full list of rotations on the input operator...
        if stab_rotations != []:
            op_rotated = operator.recursive_rotate_by_Pword(pauli_rot_list=stab_rotations, angles=angles)
        else:
            op_rotated = operator
        
        self.rotated_flag = True
        # ...and finally perform the stabilizer subspace projection
        op_project = self._perform_projection(operator=op_rotated, sym_sector=sym_sector)
    
        return op_project

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

    def identify_symmetry_sector(self, 
            ref_state: Union[List[int], np.array]
        ) -> np.array:
        """ Given the specified reference state, e.g. Hartree-Fock |1...10...0>, 
        determine the correspinding sector by measuring the symmetry generators

        TODO: currently only supports single basis vector reference - should accept a linear combination
        """
        ref_state = np.array(ref_state)
        return (-1)**np.count_nonzero(self.symmetry_generators.Z_block & ref_state, axis=1)

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
        if sector is None:
            assert(ref_state is not None), 'If no sector is provided then a reference state must be given instead'
            sector = self.identify_symmetry_sector(ref_state)
        
        if aux_operator is not None:
            operator_to_taper = aux_operator.copy()
        else:
            operator_to_taper = self.operator.copy()

        return self.perform_projection(operator_to_taper, sector)

    def taper_reference_state(self, 
            ref_state: Union[List[int], np.array],
            sector: Union[List[int], np.array]=None, 
        ) -> np.array:
        """ taper the reference state by dropping any qubit positions
        projected during the perform_projection method
        """
        if sector is None:
            sector = self.identify_symmetry_sector(ref_state)
        # ensure the stabilizer subspace projection has been called
        self.taper_it(sector=sector)
        # find the non-stabilized qubit positions by combining X+Z blocks and 
        # summing down columns to find the non-identity (stabilized) qubit positions
        non_identity = self.rotated_stabilizers.X_block + self.rotated_stabilizers.Z_block
        free_qubit_positions = np.where(np.sum(non_identity, axis=0)==0)
        
        return ref_state[free_qubit_positions]

#########################################################################
#### For now uses the legacy code for identifying noncontextual sets ####
###################### TODO graph techniques! ###########################
#########################################################################

from datetime import datetime
from datetime import timedelta

# Takes two Pauli operators specified as strings (e.g., 'XIZYZ') and determines whether they commute:
def commute(x,y):
    assert len(x)==len(y), print(x,y)
    s = 1
    for i in range(len(x)):
        if x[i]!='I' and y[i]!='I' and x[i]!=y[i]:
            s = s*(-1)
    if s==1:
        return 1
    else:
        return 0

# Input: S, a list of Pauli operators specified as strings.
# Output: a boolean indicating whether S is contextual or not.
def contextualQ(S,verbose=False):
    # Store T all elements of S that anticommute with at least one other element in S (takes O(|S|**2) time).
    T=[]
    Z=[] # complement of T
    for i in range(len(S)):
        if any(not commute(S[i],S[j]) for j in range(len(S))):
            T.append(S[i])
        else:
            Z.append(S[i])
    # Search in T for triples in which exactly one pair anticommutes; if any exist, S is contextual.
    for i in range(len(T)): # WLOG, i indexes the operator that commutes with both others.
        for j in range(len(T)):
            for k in range(j,len(T)): # Ordering of j, k does not matter.
                if i!=j and i!=k and commute(T[i],T[j]) and commute(T[i],T[k]) and not commute(T[j],T[k]):
                    if verbose:
                        return [True,None,None]
                    else:
                        return True
    if verbose:
        return [False,Z,T]
    else:
        return False

def greedy_dfs(ham,cutoff,criterion='weight'):
    
    weight = {k:abs(ham[k]) for k in ham.keys()}
    possibilities = [k for k, v in sorted(weight.items(), key=lambda item: -item[1])] # sort in decreasing order of weight
    
    best_guesses = [[]]
    stack = [[[],0]]
    start_time = datetime.now()
    delta = timedelta(seconds=cutoff)
    
    i = 0
    
    while datetime.now()-start_time < delta and stack:
        
        while i < len(possibilities):
#             print(i)
            next_set = stack[-1][0]+[possibilities[i]]
#             print(next_set)
#             iscontextual = contextualQ(next_set)
#             print('  ',iscontextual,'\n')
            if not contextualQ(next_set):
                stack.append([next_set,i+1])
            i += 1
        
        if criterion == 'weight':
            new_weight = sum([abs(ham[p]) for p in stack[-1][0]])
            old_weight = sum([abs(ham[p]) for p in best_guesses[-1]])
            if new_weight > old_weight:
                best_guesses.append(stack[-1][0])
                # print(len(stack[-1][0]))
                # print(stack[-1][0],'\n')
            
        if criterion == 'size' and len(stack[-1][0]) > len(best_guesses[-1]):
            best_guesses.append(stack[-1][0])
            # print(len(stack[-1][0]))
            # print(stack[-1][0],'\n')
            
        top = stack.pop()
        i = top[1]
    
    return best_guesses

class CS_VQE(S3_projection):
    """
    """
    def __init__(self,
            operator: PauliwordOp,
            ref_state: np.array = None,
            target_sqp: str = 'Z'
        ) -> None:
        """ 
        """
        self.operator = operator
        self.ref_state = ref_state
        self.target_sqp = target_sqp
        self.contextual_operator = (operator-self.noncontextual_operator).cleanup_zeros()
        # decompose the noncontextual set into a dictionary of its 
        # universally commuting elements and anticommuting cliques
        self.decompose_noncontextual
        self.r_indices = self.noncontextual_reconstruction[:,:self.n_cliques]
        self.G_indices = self.noncontextual_reconstruction[:,self.n_cliques:]
        self.clique_operator = self.noncontextual_basis[:self.n_cliques]
        self.symmetry_generators = self.noncontextual_basis[self.n_cliques:]
        # determine the noncontextual ground state - this updates the coefficients of the clique 
        # representative operator C(r) and symmetry generators G with the optimal configuration
        self.solve_noncontextual(ref_state)

    @cached_property
    def noncontextual_operator(self):
        """ Extract a noncontextual set of Pauli terms from the operator
        TODO graph-based approach, currently uses legacy implementation
        """
        op_dict = self.operator.to_dictionary
        noncontextual_set = greedy_dfs(op_dict, cutoff=1)[-1]
        return PauliwordOp({op:op_dict[op] for op in noncontextual_set})

    @cached_property
    def noncontextual_basis(self) -> StabilizerOp:
        """ Find an independent basis for the noncontextual symmetry
        """
        # mask universally commuting terms
        adj_mat = self.noncontextual_operator.adjacency_matrix
        mask_universal = np.where(np.all(adj_mat, axis=1))
        # swap order of XZ blocks in symplectic matrix to ZX
        ZX_symp = np.hstack([self.noncontextual_operator.Z_block[mask_universal], 
                             self.noncontextual_operator.X_block[mask_universal]])
        reduced = gf2_gaus_elim(ZX_symp)
        kernel  = gf2_basis_for_gf2_rref(reduced)

        basis = StabilizerOp(kernel, np.ones(kernel.shape[0]))
        # order the basis so clique representatives appear at the beginning
        basis_order = np.lexsort(basis.adjacency_matrix)
        basis = StabilizerOp(basis.symp_matrix[basis_order],np.ones(basis.n_terms))
        self.n_cliques = np.count_nonzero(~np.all(basis.adjacency_matrix, axis=1))

        return basis

    @cached_property
    def decompose_noncontextual(self):
        """
        """
        # note the first two columns will never both be 1... definition of noncontextual set!
        # where the 1 appears determines which clique the term is in
        self.noncontextual_reconstruction = self.noncontextual_operator.basis_reconstruction(self.noncontextual_basis)
        mask_non_universal, clique_index = np.where(self.noncontextual_reconstruction[:,0:self.n_cliques])
        mask_universal = np.where(np.all(self.noncontextual_reconstruction[:,0:self.n_cliques]==0, axis=1)) 

        decomposed = {}
        univ_symp = self.noncontextual_operator.symp_matrix[mask_universal]
        univ_coef = self.noncontextual_operator.coeff_vec[mask_universal]
        decomposed['symmetry'] = PauliwordOp(univ_symp, univ_coef)

        for i in np.unique(clique_index):
            mask_clique = clique_index==i
            Ci_symp = self.noncontextual_operator.symp_matrix[mask_non_universal][mask_clique]
            Ci_coef = self.noncontextual_operator.coeff_vec[mask_non_universal][mask_clique]
            decomposed[f'clique_{i}'] = PauliwordOp(Ci_symp, Ci_coef)

        return decomposed

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
        TODO allow any number of cliques and implement effective discrete optimizer (such as hypermapper)
        """
        if ref_state is not None:
            fix_nu = np.prod((-1) ** (self.symmetry_generators.Z_block & ref_state), axis=1)
            # update the symmetry generator G coefficients
            self.symmetry_generators.coeff_vec*=fix_nu
            if self.n_cliques==2:
                def convex_problem(theta):
                    r = np.array([np.cos(theta), np.sin(theta)])
                    return self.noncontextual_objective_function(fix_nu, r)
                opt_out = minimize_scalar(convex_problem)
                self.noncontextual_energy = opt_out['fun']
                theta = opt_out['x']
                r = np.array([np.cos(theta), np.sin(theta)])
                # update the C(r) operator coefficients
                self.clique_operator.coeff_vec*=r
            else:
                raise NotImplementedError('Currently only works for two cliques')
        else:
            raise NotImplementedError('Currently only works provided a reference state')
            # Allow discrete optimization over generator value assignment here, e.g. with hypermapper
    
    @cached_property
    def unitary_partitioning(self):
        """ Implementation of the unitary partitiong procedure 
        described in https://doi.org/10.1103/PhysRevA.101.062322 (Section A)
        
        TODO Currently works only when number of cliques M=2
        """
        C0 = self.clique_operator[0]
        C1 = self.clique_operator[1]

        pauli_rotation = (C0*C1).multiply_by_constant(-1j)
        angle = np.arctan(C0.coeff_vec/C1.coeff_vec)
        #if you wish to rotate onto +1 eigenstate:
        if abs(C1.coeff_vec+np.cos(angle)) < 1e-15:
            angle += np.pi
            
        return pauli_rotation, angle

    def contextual_subspace_projection(self,
            stabilizer_indices: List[int],
            aux_operator: PauliwordOp = None
        ) -> PauliwordOp:
        """ input a list indexing the stabilizers one wishes to enforce
        index 0 always corresponds to the clique operator C(r)
        """
        # this is set in fix_stabilizers for now
        sector = np.ones(len(stabilizer_indices), dtype=int)

        if aux_operator is not None:
            operator_to_project = aux_operator.copy()
        else:
            operator_to_project = self.operator.copy()

        if 0 in stabilizer_indices:
            stabilizer_indices.pop(stabilizer_indices.index(0))
            stabilizer_indices = [i-1 for i in stabilizer_indices]
            UP_rot, UP_angle = self.unitary_partitioning
            rotated_clique_op = self.clique_operator._rotate_by_single_Pword(
                UP_rot, UP_angle
                ).cleanup_zeros()
            fix_stabilizers = reduce(lambda x,y: x+y,
                [rotated_clique_op]+[self.symmetry_generators[i] for i in stabilizer_indices])
            insert_rotation = [list(UP_rot.to_dictionary.keys())[0], UP_angle]
        else:
            stabilizer_indices = [i-1 for i in stabilizer_indices]
            fix_stabilizers = reduce(lambda x,y: x+y,
                [self.symmetry_generators[i] for i in stabilizer_indices])
            insert_rotation = None

        super().__init__(fix_stabilizers, target_sqp=self.target_sqp)

        return self.perform_projection(
            operator_to_project, 
            sector, 
            insert_rotation=insert_rotation
        )
        

