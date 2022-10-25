import numpy as np
from symmer.symplectic import PauliwordOp, NoncontextualOp, StabilizerOp
from symmer.projection.utils import lp_norm
from symmer.projection import S3_projection, ObservableBiasing, StabilizerIdentification, stabilizer_walk

class ContextualSubspace(S3_projection):
    def __init__(self,
            operator: PauliwordOp,
            noncontextual_strategy: str = 'diag'
        ):
        self.H_noncon = NoncontextualOp.from_hamiltonian(
            operator, strategy=noncontextual_strategy
        )
        self.H_context = operator - self.H_noncon

    def update_stabilizers(self, 
            n_qubits: int, 
            strategy: str = '',
            aux_operator = None
        ) -> StabilizerOp:
        """
        """
        assert(n_qubits<=self.H_noncon.n_qubits), 'Cannot define a contextual subspace larger than the base Hamiltonian'
        
        self.stabilizers = None

    def _greedy_stabilizer_search(self,
            n_qubits: int, 
            depth: int=2
        ) -> StabilizerOp:
        pass

    def _aux_operator_preserving_stabilizer_search(self,
            n_qubits: int,
            aux_operator: PauliwordOp
        ) -> StabilizerOp:
        pass

    def _chemistry_HOMO_LUMO_biasing(self,
            n_qubits: int,
            HOMO_LUMO_index: int
        ) -> StabilizerOp:
        pass

    def _random_stabilizers(self, 
            n_qubits: int
        )  -> StabilizerOp:
        pass

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