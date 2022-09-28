from qiskit import QuantumCircuit
from functools import reduce
from symmer.symplectic import PauliwordOp
from symmer.unitaries.gate_library import *

###################################################
# Decompose any QuantumCircuit into a PauliwordOp #
###################################################

def decompose_circuit(qc: QuantumCircuit) -> PauliwordOp:
    """ Decompose an QuantumCircuit into a linear combination 
    of Pauli operators via the gate definitions above.
    """
    gate_map = {
        'x':X, 'y':Y, 'z':Z, 'h':Had, 'rx':RX, 'ry':RY, 
        'rz':RZ, 'u1':U1, 'cz':CZ, 'cx':CX, 's':S
    } # for conversion from qiskit to PauliwordOp definitions
    gateset = []
    for gate in qc:
        name = gate.operation.name
        qubits = gate.qubits
        target = qc.num_qubits - 1 - qubits[0].index
        if len(qubits) != 1:
            control, target = target, qc.num_qubits - 1 - qubits[1].index
            
        if name == 'barrier':
            pass
        else:
            if name in ['x', 'y', 'z', 'h', 's', 'sdg']:
                G = gate_map[name](qc.num_qubits, target)
            elif name in ['cz', 'cx']:
                G = gate_map[name](qc.num_qubits, control, target)
            elif name in ['rx', 'ry', 'rz', 'u1']:
                angle = -float(gate.operation.params[0]) # minus due to differing qiskit definition
                G = gate_map[name](qc.num_qubits, target, angle=angle)
            else:
                raise ValueError(f'Gate decomposition {name} not defined')
            gateset.append(G)

    qc_decomposition = reduce(lambda x,y:x*y, gateset[::-1])
    
    return qc_decomposition.cleanup()