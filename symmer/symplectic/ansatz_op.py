import numpy as np
from typing import Dict, List, Tuple, Union
from functools import reduce
from cached_property import cached_property
from symmer.symplectic import PauliwordOp
from qiskit.circuit import QuantumCircuit, ParameterVector

class AnsatzOp(PauliwordOp):
    """ Based on PauliwordOp and introduces functionality for converting operators to quantum circuits
    """
    def __init__(self,
            operator:   Union[List[str], Dict[str, float], np.array],
            coeff_vec: Union[List[complex], np.array] = None
        ) -> None:
        """
        """
        super().__init__(operator, coeff_vec)
        assert(np.all(self.coeff_vec.imag==0)), 'Coefficients must have zero imaginary component'
        self.coeff_vec = self.coeff_vec.real
    
    def cleanup(self):
        return super()._cleanup()

    def exponentiate(self):
        """
        Returns:
            exp_T (PauliwordOp): exponentiated form of the ansatz operator
        """
        exp_bin = []
        for term, angle in zip(self.symp_matrix, self.coeff_vec):
            exp_bin.append(
                PauliwordOp(
                    np.vstack([np.zeros_like(term), term]), 
                    [np.cos(angle), 1j*np.sin(angle)]
                )
            )
        exp_T = reduce(lambda x,y:x*y, exp_bin)

        return exp_T

    @cached_property
    def to_instructions(self) -> Dict[int, Dict[str, List[int]]]:
        """ Stores a dictionary of gate instructions at each step, where each value
        is a dictionary indicating the indices on which to apply each H,S,CNOT and RZ gate
        """
        circuit_instructions = {}
        for step, (X,Z) in enumerate(zip(self.X_block, self.Z_block)):
            # locations for H and S gates to transform into Pauli Z basis
            H_indices = np.where(X)[0][::-1]
            S_indices = np.where(X & Z)[0][::-1]
            # CNOT cascade indices
            CNOT_indices = np.where(X | Z)[0][::-1]
            circuit_instructions[step] = {'H_indices':H_indices, 
                                        'S_indices':S_indices, 
                                        'CNOT_indices':CNOT_indices,
                                        'RZ_index':CNOT_indices[-1]}
        return circuit_instructions

    def to_QuantumCircuit(self, 
        ref_state: np.array = None,
        basis_change_indices: Dict[str, List[int]] = {'X_indices':[],'Y_indices':[]},
        trotter_number: int = 1, 
        bind_params: bool = True
        ) -> str:
        """
        Convert the operator to a QASM circuit string for input 
        into quantum computing packages such as Qiskit and Cirq

        basis_change_indices in form [X_indices, Y_indices]
        """
        def qiskit_ordering(indices):
            """ we index from left to right - in Qiskit this ordering is reversed
            """
            return self.n_qubits - 1 - indices

        qc = QuantumCircuit(self.n_qubits)
        for i in qiskit_ordering(np.where(ref_state==1)[0]):
            qc.x(i)

        def CNOT_cascade(cascade_indices, reverse=False):
            index_pairs = list(zip(cascade_indices[:-1], cascade_indices[1:]))
            if reverse:
                index_pairs = index_pairs[::-1]
            for source, target in index_pairs:
                qc.cx(source, target)

        def circuit_from_step(angle, H_indices, S_indices, CNOT_indices, RZ_index):
            # to Pauli X basis
            for i in S_indices:
                qc.sdg(i)
            # to Pauli Z basis
            for i in H_indices:
                qc.h(i)
            # compute parity
            CNOT_cascade(CNOT_indices)
            qc.rz(-2*angle, RZ_index)
            CNOT_cascade(CNOT_indices, reverse=True)
            for i in H_indices:
                qc.h(i)
            for i in S_indices:
                qc.s(i)

        if bind_params:
            angles = self.coeff_vec.real/trotter_number
        else:
            angles = np.array(ParameterVector('P', self.n_terms))/trotter_number

        assert(len(angles)==len(self.to_instructions)), 'Number of parameters does not match the circuit instructions'
        for trot_step in range(trotter_number):
            for step, gate_indices in self.to_instructions.items():
                qiskit_gate_indices = [qiskit_ordering(indices) for indices in gate_indices.values()]
                qc.barrier()
                circuit_from_step(angles[step], *qiskit_gate_indices)

        qc.barrier()
        for i in basis_change_indices['Y_indices']:
            qc.s(qiskit_ordering(i))
        for i in basis_change_indices['X_indices']:
            qc.h(qiskit_ordering(i))
            
        return qc