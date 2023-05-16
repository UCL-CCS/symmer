from symmer.operators import PauliwordOp, QuantumState
from symmer.projection import QubitTapering, ContextualSubspace
from symmer.utils import exact_gs_energy
from symmer.approximate import get_MPO, find_groundstate_quimb
from typing import Union, List
import numpy as np
import warnings

class QubitSubspaceManager:

    CS_ready = False

    def __init__(self,
            hamiltonian: PauliwordOp,
            ref_state: Union[np.ndarray, List[int], QuantumState] = None,
            run_qubit_tapering: bool = True,
            run_contextual_subspace: bool = True
        ) -> None:
        """
        """
        self.hamiltonian = hamiltonian
        self.ref_state = self.prepare_ref_state(ref_state)
        self.run_qubit_tapering = run_qubit_tapering
        self.run_contextual_subspace = run_contextual_subspace
        self.build_subspace_objects()
        
    def prepare_ref_state(self, ref_state=None):
        """
        """
        if ref_state is not None:
            if isinstance(ref_state, list):
                ref_state = np.array(ref_state).reshape(-1)
            if isinstance(ref_state, np.ndarray):
                ref_state = QuantumState(ref_state, [1])
        else:
            warnings.warn('No reference state supplied - trying to identify one via alternative means.')
            if self.hamiltonian.n_qubits <= 12:
                _, ref_state = exact_gs_energy(self.hamiltonian.to_sparse_matrix)
            else:
                warnings.warn(
                    'Results are currently unstable for reference state '+ 
                    'generation via tensor network techniques'
                )
                mpo = get_MPO(self.hamiltonian, max_bond_dimension=10)
                ref_state = find_groundstate_quimb(mpo)

        return ref_state.cleanup(zero_threshold=1e-4).normalize

    def build_subspace_objects(self):
        """
        """
        if self.run_qubit_tapering:
            self.QT           = QubitTapering(operator=self.hamiltonian)
            self._hamiltonian = self.QT.taper_it(ref_state=self.ref_state)
            self._ref_state   = self.QT.tapered_ref_state.normalize
        else:
            self._hamiltonian = self.hamiltonian.copy()
            self._ref_state   = self.ref_state.copy()

        if self.run_contextual_subspace:
            self.CS = ContextualSubspace(
                operator=self._hamiltonian,
                reference_state=self._ref_state,
                noncontextual_strategy='StabilizeFirst',
                noncontextual_solver='brute_force'
            )

    def get_reduced_hamiltonian(self, 
            n_qubits:int=None, aux_operator:PauliwordOp=None
        ) -> PauliwordOp:
        """
        """
        if self.run_qubit_tapering:
            if not self.run_contextual_subspace and n_qubits is not None:
                warnings.warn('The n_qubits parameter is redundant when contextual subspace is not run.')
            operator_out = self._hamiltonian
            aux_operator = self.QT.taper_it(aux_operator=aux_operator)

        if self.run_contextual_subspace:
            assert n_qubits is not None, 'Must supply the desired number of qubits for the contextual subspace.'
            self.CS.update_stabilizers(
                n_qubits=n_qubits, aux_operator=aux_operator, strategy='aux_preserving'
            )
            self.CS_ready = True
            operator_out = self.CS.project_onto_subspace()

        return operator_out
        
    def project_auxiliary_operator(self, operator):
        """
        """
        if self.run_qubit_tapering:
            operator = self.QT.taper_it(aux_operator=operator)
        
        if self.run_contextual_subspace:
            assert self.CS_ready, 'Have not yet projected the Hamiltonian into the contextual subspace'
            operator = self.CS.project_onto_subspace(
                operator_to_project=operator
            )

        return operator



    
    

