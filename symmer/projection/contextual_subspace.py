import numpy as np
from symmer.symplectic import PauliwordOp, StabilizerOp, NoncontextualOp
from symmer.projection.utils import lp_norm
from symmer.projection import S3_projection, ObservableBiasing, StabilizerIdentification, stabilizer_walk

class ContextualSubspace(S3_projection):
    def __init__(self,
            operator: PauliwordOp,
            noncontextual_strategy: str = 'diag'
        ):
        self.noncontextual_operator = NoncontextualOp.from_hamiltonian(
            operator, strategy=noncontextual_strategy
        )
        self.contextual_operator = operator - self.noncontextual_operator

    def update_stabilizers(self, 
            n_qubits: int, 
            strategy: str = 'aux_preserving',
            aux_operator: PauliwordOp = None,
            depth: int = 2,
            HOMO_LUMO_index: int = None
        ) -> StabilizerOp:
        """
        """
        assert(n_qubits<=self.noncontextual_operator.n_qubits), 'Cannot define a contextual subspace larger than the base Hamiltonian'
        
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
            S = self._chemistry_HOMO_LUMO_biasing(
                n_qubits=n_qubits, HOMO_LUMO_index=HOMO_LUMO_index
            )
        else:
            raise ValueError('Unrecognised stabilizer search strategy.')

        self.stabilizers = S

    def _greedy_stabilizer_search(self,
            n_qubits: int, 
            depth: int=2
        ) -> StabilizerOp:
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

        # if any of the stabilizers anticommute with a clique...
        clique_commutation = S.commutes_termwise(self.noncontextual_operator.clique_operator)
        if np.any(~clique_commutation):
            S_index = int(np.where(np.any(~clique_commutation, axis=1))[0][0])
            self.clique_index = int(np.where(clique_commutation[S_index])[0][0])
            S_clique = S[S_index]
            
            force_symmetry = self.noncontextual_operator.clique_operator.copy()
            force_symmetry.coeff_vec[:] = 2*np.max(abs(aux_operator.coeff_vec))
            
            SI = StabilizerIdentification(aux_operator+force_symmetry)
            S_symmetry = SI.symmetry_basis_by_subspace_dimension(n_qubits+1)
            S = StabilizerOp.from_PauliwordOp(S_clique+S_symmetry)

        return S

    def _chemistry_HOMO_LUMO_biasing(self,
            n_qubits: int,
            HOMO_LUMO_index: int
        ) -> StabilizerOp:
        raise NotImplementedError

    def _random_stabilizers(self, 
            n_qubits: int
        )  -> StabilizerOp:
        raise NotImplementedError

    def project_onto_subspace(self) -> PauliwordOp:
        """ Projects with respect to the current stabilizers; these are 
        updated using the ContextualSubspace.update_stabilizers method.
        """
        super().__init__(self.stabilizers)
        #TODO s3 projection

    def hamiltonian(self, 
            n_qubits: int, 
            strategy: str = '',
            aux_operator = None
        ) -> PauliwordOp:

        self.get_stabilizers(n_qubits, strategy, aux_operator)
        cs_operator = self.project_onto_subspace()

        return cs_operator