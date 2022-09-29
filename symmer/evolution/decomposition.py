from functools import reduce
from symmer.symplectic import PauliwordOp
from symmer.evolution.gate_library import *

##############################################
# Decompose any QASM file into a PauliwordOp #
##############################################

def qasm_to_PauliwordOp(qasm: str, reverse=False, combine=True) -> PauliwordOp:
    """ Decompose an QASM circuit into a linear combination of Pauli 
    operators via the gate definitions in evolution.gate_library.
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