from symmer.operators import PauliwordOp, QuantumState
from symmer.projection import QubitTapering, ContextualSubspace
from symmer.utils import exact_gs_energy
from symmer.approximate import get_MPO, find_groundstate_quimb
from typing import Union, List
import numpy as np
import warnings

class QubitSubspaceManager:
    """ 
    Class for automating the following qubit subspace techqniques:

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

    Attributes:
        _projection_ready (bool): If True, Hamiltonian is ready for projection. By default, it is set to False.
    """

    _projection_ready = False

    def __init__(self,
            hamiltonian: PauliwordOp,
            ref_state: Union[np.ndarray, List[int], QuantumState] = None,
            run_qubit_tapering: bool = True,
            run_contextual_subspace: bool = True
        ) -> None:
        """
        Args:
            hamiltonian (PauliwordOp): Hamiltonian which is to be projected.
            ref_state (QuantumState): Reference State. If no reference state is provided, then try to generate one. By default, it is set to None.
            run_qubit_tapering (bool): If True, Qubit Tapering is performed. By default, it is set to True. 
            run_contextual_subspace (bool): If True, Contextual Subspace Method is used to solve the problem. By default, it is set to True.
        """
        self.hamiltonian = hamiltonian
        self.ref_state = self.prepare_ref_state(ref_state)
        self.run_qubit_tapering = run_qubit_tapering
        self.run_contextual_subspace = run_contextual_subspace
        self.build_subspace_objects()
        
    def prepare_ref_state(self, ref_state=None) -> QuantumState:
        """ 
        If no reference state is provided, then try to generate one.
        If the Hamiltonian contains fewer than 12 qubits, we will diagonalise
        and select the true ground state. Otherwise, a cheap DMRG calculation
        will be performed to generate an approximate ground state.

        Args: 
            ref_state (QuantumState): Reference State. If no reference state is provided, then try to generate one. By default, it is set to None.

        Returns:
            ref_state (QuantumState): Generated Reference State.
        """
        if ref_state is not None:
            if isinstance(ref_state, list):
                ref_state = np.array(ref_state).reshape(-1)
            if isinstance(ref_state, np.ndarray):
                ref_state = QuantumState(ref_state, [1])
            self._aux_operator = None
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
            self._aux_operator = ref_state.state_op

        return ref_state.cleanup(zero_threshold=1e-4).normalize

    def build_subspace_objects(self) -> None:
        """ 
        Initialize the relevant qubit subspace classes.
        """
        if self.run_qubit_tapering:
            self.QT             = QubitTapering(operator=self.hamiltonian)
            self._hamiltonian   = self.QT.taper_it(ref_state=self.ref_state)
            self._ref_state     = self.QT.tapered_ref_state.normalize
            self._Z2_symmetries = self.QT.symmetry_generators.copy()
        else:
            self._hamiltonian   = self.hamiltonian.copy()
            self._ref_state     = self.ref_state.copy()
            self._Z2_symmetries = None

        if self.run_contextual_subspace:
            self.CS = ContextualSubspace(
                operator=self._hamiltonian,
                reference_state=self._ref_state,
                noncontextual_strategy='StabilizeFirst',
                noncontextual_solver  ='brute_force'
            )

    def get_reduced_hamiltonian(self, 
            n_qubits:int=None, aux_operator:PauliwordOp=None
        ) -> PauliwordOp:
        """ 
        Project the Hamiltonian in line with the desired qubit subspace techqniques
        and, in the case of ContextualSubspace, the desired number of qubits.

        Args:
            n_qubits (int): Number of Qubits. By default, it is set to None.
            aux_operator (PauliwordOp): Auxiliary operator. By default, it is set to None.

        Returns:
            operator_out (PauliwordOp): Projection of Hamiltonian.
        """
        self._projection_ready = True
        self._n_qubits = n_qubits
        if aux_operator is None:
            aux_operator = self._aux_operator

        if n_qubits >= self.hamiltonian.n_qubits:
            warnings.warn(
                'Specified at least as many qubits as are present in the Hamiltonian - '+
                f'returning the full {self.hamiltonian.n_qubits} operator.')
            operator_out = self.hamiltonian

        elif n_qubits > self._hamiltonian.n_qubits:
            # if one wishes not to taper all the available Z2 symmetries
            assert self.run_qubit_tapering, ''
            self.QT.symmetry_generators = self._Z2_symmetries[:self.hamiltonian.n_qubits-n_qubits]
            operator_out = self.QT.taper_it(ref_state=self.ref_state)

        else:
            if self.run_qubit_tapering:
                if not self.run_contextual_subspace and n_qubits < self._hamiltonian.n_qubits:
                    warnings.warn(
                        'When contextual subspace is not run we may only reduce '+ 
                        'the Hamiltonian by the number of Z2 symmetries present. '+
                        f'The reduced Hamiltonian will contain {self._hamiltonian.n_qubits} qubits.'
                    )
                self.QT.symmetry_generators = self._Z2_symmetries
                aux_operator = self.QT.taper_it(aux_operator=aux_operator)
                operator_out = self._hamiltonian

            if self.run_contextual_subspace:
                assert n_qubits is not None, 'Must supply the desired number of qubits for the contextual subspace.'
                self.CS.update_stabilizers(
                    n_qubits=n_qubits, aux_operator=aux_operator, strategy='aux_preserving'
                )
                operator_out = self.CS.project_onto_subspace()

            if not self.run_qubit_tapering and not self.run_contextual_subspace:
                warnings.warn('Not running any subspace methods - returning the original Hamiltonian')
                operator_out = self.hamiltonian

        return operator_out
        
    def project_auxiliary_operator(self, operator: PauliwordOp) -> PauliwordOp:
        """ 
        Project additional operators consistently with respect to the Hamiltonian.

        Args:
            operator (PauliwordOp): Additional operator which has to be projected consistently with respect to the Hamiltonian.

        Returns:
            operator (PauliwordOp): Projection of additional operator.
        """
        assert self._projection_ready, 'Have not yet projected the Hamiltonian into the contextual subspace'

        if self._n_qubits < self.hamiltonian.n_qubits:

            if self.run_qubit_tapering:
                operator = self.QT.taper_it(aux_operator=operator)
            
            if self.run_contextual_subspace:
                operator = self.CS.project_onto_subspace(operator_to_project=operator)

        return operator
    
    def project_auxiliary_state(self, state: QuantumState) -> QuantumState:
        """ 
        Project quantum state consistently with respect to the Hamiltonian.

        Args:
            operator (QuantumState): Quantum State which has to be projected consistently with respect to the Hamiltonian.
        
        Returns:
            operator (PauliwordOp): Projection of Quantum State.
        """
        assert self._projection_ready, 'Have not yet projected the Hamiltonian into the contextual subspace'

        if self._n_qubits < self.hamiltonian.n_qubits:

            if self.run_qubit_tapering:
                state = self.QT.project_state(state_to_project=state)
            if self.run_contextual_subspace:
                state = self.CS.project_state(state_to_project=state)

        return state



    
    

