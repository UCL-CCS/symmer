from symmer.operators import PauliwordOp, QuantumState
from symmer.projection import QubitTapering, ContextualSubspace
from symmer.utils import exact_gs_energy
from symmer.approximate import get_MPO, find_groundstate_quimb
from typing import Union, List
import numpy as np
import warnings

class QubitSubspaceManager:
    """ Class for automating the following qubit subspace techqniques:

    *** QubitTapering ***
    - Maps each Z2-symmetry onto distinct qubits and projects over them
    - Yields an exact reduction, given that the correct sector is identifed

    *** ContextualSubspace ***
    - Partitions the Hamiltonian into noncontextual and contextual components
    - The noncontextual problem may be solved via classical optimization
    - We then impose noncontextual symmetries over the contextual portion of
      the Hamiltonian, constrained by the noncontextual solution
    - Allows one to define a reduced Hamiltonian of any size but incurs error
    
    It is recommended that the user should specify a reference state,
    such as Hartree-Fock. Otherwise, Symmer will try to identify an
    alternative refernce, either via direct diagonalization  if the 
    Hamiltonian is sufficiently small, or using a DMRG calculation.
    """

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
        
    def prepare_ref_state(self, ref_state=None) -> QuantumState:
        """ If no reference state is provided, then try to generate one.
        If the Hamiltonian contains fewer than 12 qubits, we will diagonalise
        and select the true ground state. Otherwise, a cheap DMRG calculation
        will be performed to generate an approximate ground state.
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

    def build_subspace_objects(self) -> None:
        """ Initialize the relevant qubit subspace classes.
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
        """ Project the Hamiltonian in line with the desired qubit subspace techqniques
        and, in the case of ContextualSubspace, the desired number of qubits.
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
        
    def project_auxiliary_operator(self, operator: PauliwordOp) -> PauliwordOp:
        """ Project additional operators consistently with respect to the Hamiltonian.
        """
        if self.run_qubit_tapering:
            operator = self.QT.taper_it(aux_operator=operator)
        
        if self.run_contextual_subspace:
            assert self.CS_ready, 'Have not yet projected the Hamiltonian into the contextual subspace'
            operator = self.CS.project_onto_subspace(
                operator_to_project=operator
            )

        return operator



    
    

