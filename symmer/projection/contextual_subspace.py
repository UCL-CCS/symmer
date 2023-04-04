import numpy as np
from symmer.operators import PauliwordOp, IndependentOp, NoncontextualOp
from symmer.projection.utils import (
    update_eigenvalues, StabilizerIdentification, ObservableBiasing, stabilizer_walk
)
from symmer.projection import S3_projection
from typing import List, Union, Optional

class ContextualSubspace(S3_projection):
    """ Class for performing contextual subspace methods as per https://quantum-journal.org/papers/q-2021-05-14-456/.
    Reduces the number of qubits in the problem while aiming to control the systematic error incurred along the way.

    This class handles the following:
    1. Identify a set of operators one wishes to enforce as stabilizers over the contextual subspace,
        one might think ofthese as 'pseudo-symmetries', as opposed to the true, physical symmetries of qubit tapering.
    2. Construct a noncontextual Hamiltoinian that respects the stabilizers selecting in (1),
        the NoncontextualOp class handles the decomposition into a generating set and classical optimization over the noncontextual objective function 
    NOTE: the order in which (1) and (2) are performed depends on the noncontextual strategy specified
    3. Apply unitary partitioning (either sequence of rotations or linear combination of unitaries) to collapse noncontextual cliques
    
    The remaining steps are handled by the parent S3_projection class:
    4. rotate each stabilizer onto a single-qubit Pauli operator, 
    5. drop the corresponding qubits from the Hamiltonian whilst
    6. fixing the +/-1 eigenvalues
    """
    def __init__(self,
            operator: PauliwordOp,
            noncontextual_strategy: str = 'diag',
            noncontextual_solver: str = 'brute_force',
            num_anneals:Optional[int] = 1000,
            discrete_optimization_order = 'first',
            unitary_partitioning_method: str = 'LCU',
            reference_state: np.array = None,
            noncontextual_operator: NoncontextualOp = None,
        ):
        """
        """
        # noncontextual startegy will have the form x_y, where x is the actual strategy
        # and y is some supplementary method indicating a sorting key such as magnitude
        self.ref_state = reference_state
        extract_noncon_strat = noncontextual_strategy.split('_')
        self.nc_strategy = extract_noncon_strat[0]
        self.noncontextual_solver = noncontextual_solver
        self.num_anneals = num_anneals
        self.discrete_optimization_order = discrete_optimization_order
        if self.nc_strategy=='StabilizeFirst':
            self.stabilize_first_method = extract_noncon_strat[1]
        # With the exception of the StabilizeFirst noncontextual strategy, here we build
        # the noncontextual Hamiltonian in line with the specified strategy
        self.operator = operator
        if noncontextual_operator is None and self.nc_strategy != 'StabilizeFirst':
            self.noncontextual_operator = NoncontextualOp.from_hamiltonian(
                operator, strategy=noncontextual_strategy
            )
        else:
            self.noncontextual_operator = noncontextual_operator
        self._noncontextual_update()
        self.unitary_partitioning_method = unitary_partitioning_method
    
    def _noncontextual_update(self):
        """ To be executed each time the noncontextual operator is updated.
        """
        if self.noncontextual_operator is not None:
            self.contextual_operator = self.operator - self.noncontextual_operator
            if self.contextual_operator.n_terms == 0:
                raise ValueError('The Hamiltonian is noncontextual, the contextual subspace is empty.')
            self.noncontextual_operator.solve(
                strategy=self.noncontextual_solver, 
                ref_state=self.ref_state, 
                num_anneals=self.num_anneals,
                discrete_optimization_order=self.discrete_optimization_order
            )
            self.n_cliques = self.noncontextual_operator.n_cliques

    def manual_stabilizers(self, S: Union[List[str], IndependentOp]) -> None:
        """ Specify a set of operators to enforce manually
        """
        if isinstance(S, list):
            S = IndependentOp.from_list(S)
        self.n_qubits_in_subspace = self.operator.n_qubits - S.n_terms
        self.stabilizers = S

    def update_stabilizers(self, 
            n_qubits: int, 
            strategy: str = 'aux_preserving',
            aux_operator: PauliwordOp = None,
            depth: int = 2,
            n_cliques: int = 2,
            n_stabilizers_in_clique: int = 1,
            HF_array: np.array = None,
            use_X_only: bool = True
        ) -> None:
        """ Update the stabilizers that will be used for the subspace projection
        """
        assert(n_qubits<=self.operator.n_qubits), (
            'Cannot define a contextual subspace larger than the base Hamiltonian'
        )
        # will ensure one too few stabilizers will be selected, with the 
        # additional one identified via the following clique expansion step
        if self.nc_strategy=='StabilizeFirst':
            if self.stabilize_first_method=='magnitude':
                n_qubits += 1

        if strategy == 'aux_preserving':
            S = self._aux_operator_preserving_stabilizer_search(
                n_qubits=n_qubits, aux_operator=aux_operator, use_X_only=use_X_only
            )
        elif strategy == 'greedy_search':
            S = self._greedy_stabilizer_search(
                n_qubits=n_qubits, depth=depth
            )
        elif strategy == 'random':
            S = self._random_stabilizers(
                n_qubits=n_qubits
            )
        elif strategy == 'HOMO_LUMO_biasing':
            S = self._HOMO_LUMO_biasing(
                n_qubits=n_qubits, HF_array=HF_array, 
                weighting_operator=aux_operator, use_X_only=use_X_only
            )
        else:
            raise ValueError('Unrecognised stabilizer search strategy.')

        self.n_qubits_in_subspace = self.operator.n_qubits - S.n_terms
        self.stabilizers = S

        # the StabilizeFirst strategy differs from the others in that the noncontextual
        # Hamiltonian is constructed AFTER selecting stabilizers, which is what we do next:
        if self.nc_strategy == 'StabilizeFirst':
            if self.stabilize_first_method == 'commuting':
                assert n_stabilizers_in_clique < self.stabilizers.n_terms, 'At least one stabilizer must be assigned to the symmetry generating set.'
                # move stabilizers into a clique by increasing commutativity with full Hamiltonian
                stabilizer_commutativity_count = np.count_nonzero(
                    self.stabilizers.commutes_termwise(self.operator), axis=1
                )
                order_by_commutativity = np.argsort(stabilizer_commutativity_count)
                force_clique_rep = self.stabilizers[order_by_commutativity][:n_stabilizers_in_clique]
                force_symmetries = self.stabilizers[order_by_commutativity][1+n_stabilizers_in_clique:]
                sum_clique_reps = self._get_clique_representatives(
                    symmetry_terms=force_symmetries, n_cliques=n_cliques, clique_reps=[force_clique_rep]
                )
            elif self.stabilize_first_method == 'magnitude':
                # find list of anticommuting operators that commute with the stabilizers, selected by coefficient magnitude
                sum_clique_reps = self._get_clique_representatives(n_cliques=n_cliques, clique_reps=[])
                # choose the dominant term to be enforced in noncontextual solution
                extra_stabilizer = sum_clique_reps.sort(by='magnitude')[0]
                extra_stabilizer.coeff_vec[0]=1
                self.stabilizers += extra_stabilizer

            # find symmetry generators given a sum of anticommuting operators
            symgen = IndependentOp.symmetry_generators(sum_clique_reps+self.stabilizers)
            # this forms a noncontextual generating set under the Jordan product
            noncon_basis = symgen*1 + sum_clique_reps
            self.noncontextual_operator = NoncontextualOp.from_hamiltonian(strategy='basis', H=self.operator, basis=noncon_basis)
            # finally, solve the noncontextual optimization problem
            self._noncontextual_update()

    def _greedy_stabilizer_search(self,
            n_qubits: int, 
            depth: int=2
        ) -> IndependentOp:
        """
        """
        raise NotImplementedError

    def _aux_operator_preserving_stabilizer_search(self,
            n_qubits: int,
            aux_operator: PauliwordOp,
            use_X_only: bool = True
        ) -> IndependentOp:
        """ Choose stabilizers that preserve some auxiliary operator.
        This could be an Ansatz operator such as UCCSD, for example.
        """
        if aux_operator is None:
            if self.nc_strategy == 'StabilizeFirst':
                aux_operator = self.operator
            else:
                aux_operator = self.contextual_operator

        SI = StabilizerIdentification(aux_operator, use_X_only=use_X_only)
        S = SI.symmetry_generators_by_subspace_dimension(n_qubits)

        return S

    def _HOMO_LUMO_biasing(self,
            n_qubits: int,
            HF_array: np.array,
            weighting_operator: PauliwordOp = None,
            use_X_only:bool = True
        ) -> IndependentOp:
        """ Bias the Hamiltonian with respect to the HOMO-LUMO gap 
        and preserve terms in the resulting operator as above.
        """
        assert(HF_array is not None), 'Must supply the Hartree-Fock state for this strategy'
        
        OB = ObservableBiasing(
            base_operator=self.operator, 
            HOMO_LUMO_gap=np.where(np.asarray(HF_array==0).reshape(-1))[0][0]-.5 # currently assumes JW mapping!
        )
        S = stabilizer_walk(
            n_sim_qubits=n_qubits, 
            biasing_operator=OB, 
            weighting_operator=weighting_operator,
            use_X_only=use_X_only
        )
        return S

    def _random_stabilizers(self, 
            n_qubits: int
        )  -> IndependentOp:
        """ Generate a random set of stabilizers
        """
        # TODO better approach that does not rely on this *potentially infinite* while loop!
        found_stabilizers=False
        while not found_stabilizers:
            try:
                S = PauliwordOp.random(
                    self.operator.n_qubits,
                    self.operator.n_qubits-n_qubits,
                    diagonal=True
                )
                S.coeff_vec[:] = 1
                S = IndependentOp.from_PauliwordOp(S)
                found_stabilizers = True
            except:
                pass
        
        return S

    def _get_clique_representatives(self, 
            symmetry_terms: IndependentOp = None, 
            n_cliques: int = 2, 
            clique_reps: List[PauliwordOp] = []
        ) -> PauliwordOp:
        """" For use with the StabilizeFirst noncontextual strategy. Given a set of terms we wish
        to ensure are symmetries and potentially some initial clique representatives, grow the clique_reps
        until we achieve the desired number n_cliques.
        """
        assert n_cliques > 1, 'Must specify more than one clique.'
        if symmetry_terms is None:
            symmetry_terms = self.stabilizers
        non_identity = self.operator[np.any(self.operator.symp_matrix, axis=1)]
        commutes_with_stabilizers_mask = np.all(symmetry_terms.commutes_termwise(non_identity), axis=0)
        non_symmetry_mask = ~non_identity.generator_reconstruction(symmetry_terms)[1]
        valid_terms = non_identity[non_symmetry_mask & commutes_with_stabilizers_mask]
        if clique_reps != []:
            clique_elements = sum(clique_reps, PauliwordOp.empty(self.operator.n_qubits))
            anticom_with_existing_clique_reps_mask = (
                ~np.any(clique_elements.commutes_termwise(valid_terms), axis=0)
            )
            valid_terms = valid_terms[anticom_with_existing_clique_reps_mask]
        
        if len(clique_reps)==n_cliques:
            return sum(clique_reps)
        elif valid_terms.n_terms == 0:
            raise RuntimeError(f'Cannot identify {n_cliques} cliques, try lowering n_cliques.')
        else:
            clique_reps.append(valid_terms.sort()[0])
            return self._get_clique_representatives(symmetry_terms, n_cliques, clique_reps)

    def _prepare_stabilizers(self) -> None:
        """ Prepare the chosen stabilizers for projection into the contextual subspace.
        This includes eigenvalue assignment (obtained from the solution of the noncontextual Hamiltonian),
        and application of unitary partitioning if enforcing a clique element.
        """
        if self.noncontextual_operator.n_cliques > 0:
            # mask stabilizers that lie within one of the noncontextual cliques
            clique_commutation = self.stabilizers.commutes_termwise(self.noncontextual_operator.clique_operator)
            mask_which_clique = np.all(clique_commutation, axis=0)
        else:
            mask_which_clique = []

        if ~np.all(mask_which_clique):
            # we may only enforce stabilizers that live within the same clique, not accross them:
            assert(sum(mask_which_clique)==1), (
                'Cannot enforce stabilizers from different cliques since '+
                'unitary partitioning collapses onto just one of them.'
            )
            # generate the unitary partitioning rotations that map onto the 
            # clique representative correpsonding with the given stabilizers
            (
                self.mapped_clique_rep, 
                self.unitary_partitioning_rotations,        
                clique_normalizaion, # always normalized in contextual subspace...
                normalized_clique # therefore will be the same as the clique_operator

            ) = self.noncontextual_operator.clique_operator.unitary_partitioning(
                up_method=self.unitary_partitioning_method, s_index=int(np.where(mask_which_clique)[0][0])
            )
            # add the clique representative to the noncontextual basis in order to 
            # update the eigenvalue assignments of the chosen stablizers so they are 
            # consistent with the noncontextual ground state configuration - this is 
            # G U {RARdag} in the original CS-VQE notation. 
            augmented_basis = (
                IndependentOp.from_PauliwordOp(self.mapped_clique_rep) + 
                self.noncontextual_operator.symmetry_generators
            )
            # given this new basis, we reconstruct the given stabilizers to identify
            # the correct subspace corresponding with the noncontextual ground state (nu, r)
            update_eigenvalues(basis=augmented_basis, stabilizers=self.stabilizers)
            self.perform_unitary_partitioning = True
        else:
            update_eigenvalues(
                basis=self.noncontextual_operator.symmetry_generators, 
                stabilizers=self.stabilizers
            )
            self.perform_unitary_partitioning = False

    def project_onto_subspace(self, operator_to_project:PauliwordOp=None) -> PauliwordOp:
        """ Projects with respect to the current stabilizers; these are 
        updated using the ContextualSubspace.update_stabilizers method.
        """
        # if not supplied with an alternative operator for projection, use the internal operator 
        if operator_to_project is None:
            operator_to_project = self.operator.copy()    
        # first prepare the stabilizers, which is particularly relevant when 
        # one wishes to enforce stabilizer(s) lying within a noncontextual clique
        self._prepare_stabilizers()
        # instantiate the parent S3_projection class that handles the subspace projection
        super().__init__(self.stabilizers)
        # perform unitary partitioning
        if self.perform_unitary_partitioning:
            # the rotation is implemented differently depending on the choice of LCU or seq_rot
            if self.unitary_partitioning_method=='LCU':
                # linear-combination-of-unitaries approach
                rotated_op = (self.unitary_partitioning_rotations * operator_to_project
                        * self.unitary_partitioning_rotations.dagger).cleanup()
            elif self.unitary_partitioning_method=='seq_rot':
                # sequence-of-rotations approach
                rotated_op = operator_to_project.perform_rotations(self.unitary_partitioning_rotations)
            else:
                raise ValueError('Unrecognised unitary partitioning rotation method, must be one of LCU or seq_rot.')
        else:
            rotated_op = operator_to_project
        # finally, project the operator before returning
        cs_operator = self.perform_projection(rotated_op)

        return cs_operator

    def hamiltonian(self, 
            n_qubits: int, 
            strategy: str = 'aux_preserving',
            aux_operator: PauliwordOp = None,
            depth: int = 2,
            HF_array: np.array = None
        ) -> PauliwordOp:
        """ Wraps all the above methods for ease of use
        """
        self.update_stabilizers(
            n_qubits=n_qubits, 
            strategy=strategy, 
            aux_operator=aux_operator, 
            depth=depth,
            HF_array = HF_array
        )
        cs_operator = self.project_onto_subspace()
        return cs_operator