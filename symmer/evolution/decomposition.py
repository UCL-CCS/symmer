from functools import reduce
from typing import Dict, List, Union
from symmer.operators import PauliwordOp, QuantumState
from symmer.evolution.gate_library import *
from qiskit.circuit import QuantumCircuit, ParameterVector
from networkx import Graph, draw_spring
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

##############################################
# Decompose any QASM file into a PauliwordOp #
##############################################

def qasm_to_PauliwordOp(qasm: str, reverse=False, combine=True) -> PauliwordOp:
    """ 
    Decompose an QASM circuit into a linear combination of Pauli 
    operators via the gate definitions in evolution.gate_library.

    Args:
        qasm (str): QASM circuit which you want to convert into  a linear combination of Pauli operators.
        reverse (bool): If 'True' then flip qubit ordering and negate angles (for consistency with Qiskit). By default it's set to 'False'.
        combine (bool): If 'True' then take product over gateset - obscures gate contributions in resulting PauliwordOp. By default it's set to 'True'.

    Returns:
        A linear combination of Pauli operators representing QASM circuit.
    """
    gate_map = {
        'x':X, 'y':Y, 'z':Z, 'h':Had, 'rx':RX, 'ry':RY, 
        'rz':RZ, 'u1':U1, 'cz':CZ, 'cx':CX, 's':S
    } # for conversion from qiskit to PauliwordOp definitions
    gateset = []
    for gate in qasm.split(';\n')[:-1]:
        name, qubits = gate.split(' ')
        # identify number of qubits in circuit
        if name=='qreg':
            num_qubits = int(qubits[2:-1])
        if name in ['barrier', 'include', 'OPENQASM', 'qreg']:
            pass
        else:
            # extract angle
            if name.find('(')!=-1:
                name, angle = name.split('(')
                angle = angle[:-1]
                if angle=='pi/2':
                    angle = np.pi/2
                elif angle=='-pi/2':
                    angle = -np.pi/2
                else:
                    angle = float(angle)
            else:
                angle = None
            #extract qubits
            if qubits.find(',')!=-1:
                control, target = qubits.split(',')
                control, target = int(control[2:-1]), int(target[2:-1])
            else:
                control, target = -1, int(qubits[2:-1])
            # if reverse then flip qubit ordering and negate angles (for consistency with Qiskit)
            flip=1
            if reverse:
                flip=-1
                control, target = num_qubits-1-control, num_qubits-1-target
            # generate relevant gate and append to list
            if name in ['x', 'y', 'z', 'h', 's', 'sdg']:
                G = gate_map[name](num_qubits, target)
            elif name in ['cz', 'cx']:
                G = gate_map[name](num_qubits, control, target)
            elif name in ['rx', 'ry', 'rz', 'u1']:
                G = gate_map[name](num_qubits, target, angle=flip*angle)
            else:
                raise ValueError(f'Gate decomposition {name} not defined')
            gateset.append(G)

    # if combine then take product over gateset - obscures gate contributions in resulting PauliwordOp
    if combine:
        qc_decomposition = reduce(lambda x,y:x*y, gateset[::-1])
        return qc_decomposition.cleanup()
    else:
        return gateset

####################################################
# Trotterized circuit of exponentiated PauliwordOp #
####################################################

def PauliwordOp_to_instructions(PwordOp) -> Dict[int, Dict[str, List[int]]]:
        """ 
        Stores a dictionary of gate instructions at each step, where each value
        is a dictionary indicating the indices on which to apply each H,S,CNOT and RZ gate.

        Args:
            PwordOp (PauliWordOp): PauliwordOp which has to be converted into Trotterized circuit instructions.

        Returns:
            circuit_instructions (dict): Trotterized circuit
        """
        circuit_instructions = {}
        for step, (X,Z) in enumerate(zip(PwordOp.X_block, PwordOp.Z_block)):
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

def PauliwordOp_to_QuantumCircuit(
    PwordOp: PauliwordOp, 
    ref_state: np.array  = None,
    basis_change_indices: Dict[str, List[int]] = {'X_indices':[],'Y_indices':[]},
    trotter_number: int = 1, 
    bind_params: bool = True,
    include_barriers:bool = True,
    parameter_label: str = 'P'
    ) -> QuantumCircuit:
    """
    Convert the operator to a QASM circuit string for input 
    into quantum computing packages such as Qiskit and Cirq.

    Args:
        PwordOp (PauliwordOp): PauliwordOp which has to be converted into Quantum Circuit.
        ref_state (np.array): Reference State. By default, it is set to None.
        basis_change_indices (Dict[str, List[int]]): Basis change indices in form [X_indices, Y_indices].
        trotter_number (int): Trotter Number. By default, it's set to 1.
        bind_params (bool): If 'True', parameters are binded. By default it's set to 'True'.
        include_barriers (bool): If 'True', the quantum circuit includes barriers. By default it's set to 'True'.
        parameter_label (str): Parameter label. By default it's set to 'P'.

    Returns:
        Quantum Circuit representation of Operator
    """
    if isinstance(ref_state, QuantumState):
        assert ref_state.n_terms == 1
        ref_state = ref_state.state_matrix[0]

    def qiskit_ordering(indices):
        """ 
        We index from left to right - in Qiskit this ordering is reversed.

        Args:
            indices (int): Indices
        """
        return PwordOp.n_qubits - 1 - indices

    qc = QuantumCircuit(PwordOp.n_qubits)
    for i in qiskit_ordering(np.where(ref_state==1)[0]):
        qc.x(i)

    non_identity = PwordOp[np.any(PwordOp.symp_matrix, axis=1)]

    if non_identity.n_terms > 0:

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
            angles = non_identity.coeff_vec.real/trotter_number
        else:
            angles = np.array(ParameterVector(parameter_label, non_identity.n_terms))/trotter_number

        instructions = PauliwordOp_to_instructions(non_identity)
        assert(len(angles)==len(instructions)), 'Number of parameters does not match the circuit instructions'
        for trot_step in range(trotter_number):
            for step, gate_indices in instructions.items():
                qiskit_gate_indices = [qiskit_ordering(indices) for indices in gate_indices.values()]

                if include_barriers:
                    qc.barrier()

                circuit_from_step(angles[step], *qiskit_gate_indices)

    if include_barriers:
        qc.barrier()

    for i in basis_change_indices['Y_indices']:
        qc.s(qiskit_ordering(i))
    for i in basis_change_indices['X_indices']:
        qc.h(qiskit_ordering(i))
        
    return qc