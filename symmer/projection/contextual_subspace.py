import numpy as np
from symmer.operators import PauliwordOp, IndependentOp, NoncontextualOp, QuantumState
from symmer.projection.utils import (
    update_eigenvalues, StabilizerIdentification, ObservableBiasing, stabilizer_walk
)
from symmer.projection import S3_projection
from symmer.evolution import trotter
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
            unitary_partitioning_method: str = 'seq_rot',
            reference_state: Union[np.array, QuantumState] = None,
            noncontextual_operator: NoncontextualOp = None,
            noncontextual_expansion_order: int = 1
        ):
        """
        """
        # noncontextual startegy will have the form x_y, where x is the actual strategy
        # and y is some supplementary method indicating a sorting key such as magnitude
        if reference_state is None or isinstance(reference_state, QuantumState):
            self.ref_state = reference_state
        else:
            self.ref_state = QuantumState(reference_state)            
        extract_noncon_strat = noncontextual_strategy.split('_')
        self.nc_strategy = extract_noncon_strat[0]
        self.noncontextual_solver = noncontextual_solver
        self.num_anneals = num_anneals
        self.noncontextual_expansion_order = noncontextual_expansion_order
        # With the exception of the StabilizeFirst noncontextual strategy, here we build
        # the noncontextual Hamiltonian in line with the specified strategy
        self.operator = operator
        if noncontextual_operator is None and self.nc_strategy != 'StabilizeFirst':
            self.noncontextual_operator = NoncontextualOp.from_hamiltonian(
                operator, strategy=noncontextual_strategy
            )
        else:
            self.noncontextual_operator = noncontextual_operator
        self.unitary_partitioning_method = unitary_partitioning_method
        self._noncontextual_update()

    def manual_stabilizers(self, S: Union[List[str], IndependentOp]) -> None:
        """ Specify a set of operators to enforce manually
        """
        if isinstance(S, list):
            S = IndependentOp.from_list(S)
        self.n_qubits_in_subspace = self.operator.n_qubits - S.n_terms
        if self.n_qubits_in_subspace == 0:
            self.return_NC = True
        else:
            self.return_NC = False
        self.stabilizers = S
        self._prepare_stabilizers()

    def update_stabilizers(self, 
            n_qubits: int, 
            strategy: str = 'aux_preserving',
            aux_operator: PauliwordOp = None,
            HF_array: np.array = None,
            use_X_only: bool = True
        ) -> None:
        """ Update the stabilizers that will be used for the subspace projection
        """
        assert(n_qubits<=self.operator.n_qubits), (
            'Cannot define a contextual subspace larger than the base Hamiltonian'
        )
        if n_qubits == 0:
            n_qubits = 1
            self.return_NC = True
        else:
            self.return_NC = False

        if n_qubits == self.operator.n_qubits:
            self.stabilizers = None
        else:
            if strategy == 'aux_preserving':
                S = self._aux_operator_preserving_stabilizer_search(
                    n_qubits=n_qubits, aux_operator=aux_operator, use_X_only=use_X_only
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
            self._prepare_stabilizers()

    def _noncontextual_update(self):
        """ To be executed each time the noncontextual operator is updated.
        """
        if self.noncontextual_operator is not None:
            self.noncontextual_operator.up_method = self.unitary_partitioning_method
            self.contextual_operator = self.operator - self.noncontextual_operator
            if self.contextual_operator.n_terms == 0:
                raise ValueError('The Hamiltonian is noncontextual, the contextual subspace is empty.')
            self.noncontextual_operator.solve(
                strategy=self.noncontextual_solver, 
                ref_state=self.ref_state, 
                num_anneals=self.num_anneals,
                expansion_order=self.noncontextual_expansion_order
            )
            self.n_cliques = self.noncontextual_operator.n_cliques
        
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

    def _prepare_stabilizers(self) -> None:
        """ Prepare the chosen stabilizers for projection into the contextual subspace.
        This includes eigenvalue assignment (obtained from the solution of the noncontextual Hamiltonian),
        and application of unitary partitioning if enforcing a clique element.
        """
        self.S3_initialized = False
        #the StabilizeFirst strategy differs from the others in that the noncontextual
        #Hamiltonian is constructed AFTER selecting stabilizers, which is what we do here:
        if self.nc_strategy == 'StabilizeFirst':
            self.noncontextual_operator = NoncontextualOp._from_stabilizers_noncontextual_op(
                H=self.operator, stabilizers=self.stabilizers, use_jordan_product=False
            )
            self._noncontextual_update()

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
            self.noncontextual_operator.update_clique_representative_operator(
                clique_index=int(np.where(mask_which_clique)[0][0])
            )
            # add the clique representative to the noncontextual generators in order to 
            # update the eigenvalue assignments of the chosen stablizers so they are 
            # consistent with the noncontextual ground state configuration - this is 
            # G U {RARdag} in the original CS-VQE notation. 
            augmented_generators = (
                IndependentOp(self.noncontextual_operator.mapped_clique_rep.symp_matrix, [-1]) + 
                self.noncontextual_operator.symmetry_generators
            )
            # given these new generators, we reconstruct the given stabilizers to identify
            # the correct subspace corresponding with the noncontextual ground state (nu, r)
            update_eigenvalues(generators=augmented_generators, stabilizers=self.stabilizers)
            self.perform_unitary_partitioning = True
        else:
            update_eigenvalues(
                generators=self.noncontextual_operator.symmetry_generators, 
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
        # if there are no stabilizers, return the input operator
        if self.stabilizers is None:
            return operator_to_project   
        # instantiate the parent S3_projection class that handles the subspace projection
        super().__init__(self.stabilizers)
        self.S3_initialized = True
        # perform unitary partitioning
        if self.perform_unitary_partitioning:
            # the rotation is implemented differently depending on the choice of LCU or seq_rot
            if self.noncontextual_operator.up_method=='LCU':
                # linear-combination-of-unitaries approach
                rotated_op = (self.noncontextual_operator.unitary_partitioning_rotations * operator_to_project
                        * self.noncontextual_operator.unitary_partitioning_rotations.dagger).cleanup()
            elif self.noncontextual_operator.up_method=='seq_rot':
                # sequence-of-rotations approach
                rotated_op = operator_to_project.perform_rotations(self.noncontextual_operator.unitary_partitioning_rotations)
            else:
                raise ValueError('Unrecognised unitary partitioning rotation method, must be one of LCU or seq_rot.')
        else:
            rotated_op = operator_to_project
        # finally, project the operator before returning
        cs_operator = self.perform_projection(rotated_op)

        if self.return_NC:
            assert cs_operator.n_qubits == 1, 'Projected operator consists of more than one qubit.'
            cs_operator = NoncontextualOp.from_PauliwordOp(cs_operator)
            cs_operator.solve()
            return cs_operator.energy
        else:
            return cs_operator

    def project_state_onto_subspace(self, 
            state_to_project: QuantumState = None
        ) -> QuantumState:
        """ Project a QuantumState into the contextual subspace
        """
        # if there are no stabilizers, return the input QuantumState
        if self.stabilizers is None:
            return state_to_project
        
        assert self.S3_initialized, 'Must first project an operator into the contextual subspace via the project_onto_subspace method'
        # can provide an auxiliary state to project, although not in general scalable
        if state_to_project is None:
            assert self.ref_state is not None, 'Must provide a state to project into the contextual subspace'
            state_to_project = self.ref_state

        if self.perform_unitary_partitioning:
            # behaviour is different whether using the LCU or seq_rot UP methods
            if self.noncontextual_operator.up_method == 'LCU':
                rotation = self.noncontextual_operator.unitary_partitioning_rotations
            elif self.noncontextual_operator.up_method == 'seq_rot':
                rotation_generator = sum([R*angle*.5*1j for R,angle in self.noncontextual_operator.unitary_partitioning_rotations])
                rotation = trotter(rotation_generator)
            return self.project_state(rotation * state_to_project)
        else:
            return self.project_state(state_to_project)