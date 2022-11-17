import numpy as np
from symmer.symplectic import PauliwordOp, StabilizerOp, NoncontextualOp
from symmer.projection.utils import (
    update_eigenvalues, StabilizerIdentification, ObservableBiasing, stabilizer_walk
)
from symmer.projection import S3_projection
from typing import List, Union

class ContextualSubspace(S3_projection):
    """
    """
    def __init__(self,
            operator: PauliwordOp,
            noncontextual_strategy: str = 'diag',
            unitary_partitioning_method: str = 'LCU',
            reference_state: np.array = None,
            noncontextual_operator: NoncontextualOp = None
        ):
        """
        """
        self.operator = operator
        if noncontextual_operator is None:
            self.noncontextual_operator = NoncontextualOp.from_hamiltonian(
                operator, strategy=noncontextual_strategy
            )
        else:
            self.noncontextual_operator = noncontextual_operator
        self.noncontextual_operator.solve(strategy='brute_force', ref_state=reference_state)
        self.contextual_operator = self.noncontextual_operator - self.operator
        self.unitary_partitioning_method = unitary_partitioning_method
    
    def manual_stabilizers(self, S: Union[List[str], StabilizerOp]) -> None:
        """
        """
        if isinstance(S, list):
            S = StabilizerOp.from_list(S)
        self.n_qubits_in_subspace = self.operator.n_qubits - S.n_terms
        self.stabilizers = S

    def update_stabilizers(self, 
            n_qubits: int, 
            strategy: str = 'aux_preserving',
            aux_operator: PauliwordOp = None,
            depth: int = 2,
            HF_array: np.array = None
        ) -> None:
        """ Update the stabilizers that will be used for the subspace projection
        """
        assert(n_qubits<=self.operator.n_qubits), (
            'Cannot define a contextual subspace larger than the base Hamiltonian'
        )

        if strategy == 'aux_preserving':
            S = self._aux_operator_preserving_stabilizer_search(
                n_qubits=n_qubits, aux_operator=aux_operator
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
                n_qubits=n_qubits, HF_array=HF_array, weighting_operator=aux_operator
            )
        else:
            raise ValueError('Unrecognised stabilizer search strategy.')

        self.n_qubits_in_subspace = self.operator.n_qubits - S.n_terms
        self.stabilizers = S

    def _greedy_stabilizer_search(self,
            n_qubits: int, 
            depth: int=2
        ) -> StabilizerOp:
        """
        """
        raise NotImplementedError

    def _aux_operator_preserving_stabilizer_search(self,
            n_qubits: int,
            aux_operator: PauliwordOp
        ) -> StabilizerOp:
        """
        """
        if aux_operator is None:
            aux_operator = self.contextual_operator

        SI = StabilizerIdentification(aux_operator)
        S = SI.symmetry_basis_by_subspace_dimension(n_qubits)

        return S

    def _HOMO_LUMO_biasing(self,
            n_qubits: int,
            HF_array: np.array,
            weighting_operator: PauliwordOp = None
        ) -> StabilizerOp:
        """
        """
        assert(HF_array is not None), 'Must supply the Hartree-Fock state for this strategy'
        
        OB = ObservableBiasing(
            base_operator=self.contextual_operator, 
            HOMO_LUMO_gap=np.where(HF_array==0)[0][0]-.5 # currently assumes JW mapping!
        )
        S = stabilizer_walk(
            n_sim_qubits=n_qubits, 
            biasing_operator=OB, 
            weighting_operator=weighting_operator,
        )
        return S

    def _random_stabilizers(self, 
            n_qubits: int
        )  -> StabilizerOp:
        """
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
                S = StabilizerOp.from_PauliwordOp(S)
                found_stabilizers = True
            except:
                pass
        
        return S

    def _prepare_stabilizers(self):
        """
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
                StabilizerOp.from_PauliwordOp(self.mapped_clique_rep) + 
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

    def project_onto_subspace(self, operator_to_project=None) -> PauliwordOp:
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