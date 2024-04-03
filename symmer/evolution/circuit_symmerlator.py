from typing import List
from symmer import PauliwordOp, QuantumState
from qiskit import QuantumCircuit
import numpy as np
from qiskit import qasm3
import re

class CircuitSymmerlator:
    """
    Symmer circuit simulator

    Each Clifford gate is expanded as a sequence of pi/2 rotations
    (correct up to some global phase that cancels in the inner product calculation).

    Non-Clifford gates also included, although note the use of these is inefficient.
    
    """
    def __init__(self, n_qubits: int) -> None:
        """
        """
        self.n_qubits = n_qubits
        self.sequence = []
        self.gate_map = {
            'x':self.X,'y':self.Y,'z':self.Z,
            'rx':self.RX,'ry':self.RY,'rz':self.RZ,
            'sx':self.sqrtX,'sy':self.sqrtY,'sz':self.sqrtZ,
            'cx':self.CX,'cy':self.CY,'cz':self.CZ,
            'h':self.H,'s':self.S,'sdg':self.Sdag,
            '':self.R,'t':self.T,'ccx':self.Toffoli,'swap':self.SWAP
        }

    def get_rotation_string(self, pauli: str, indices: List[int]):
        """ Given a Pauli string and list of corresponding qubit indices, return the PauliwordOp
        """
        pauli = list(pauli)
        assert len(pauli)==len(indices), 'Number of Paulis and indices do not match'
        assert set(pauli).issubset({'I','X','Y','Z'}), 'Pauli operators are either I, X, Y or Z.'
        R = ['I']*self.n_qubits
        for i,P in zip(indices, pauli):
            R[i] = P
        return PauliwordOp.from_list([''.join(R)])
    
    def pi_2_multiple(self, multiple: int):
        """ 
        For multiple % 4 = 0,1,2,3 and rotation operator R we have the following behaviour:
            0: +I
            1: +R
            2: -I
            3: -R
        multiplied by the anticommuting component of the target operator.
        """
        return np.pi/2*multiple
    
    #################################################################
    ######################  CLIFFORD GATES  #########################
    #################################################################
    
    def X(self, index: int) -> None:
        """ Pauli iX gate """
        self.sequence.append( (self.get_rotation_string('X', [index]), self.pi_2_multiple(+2)) )
    
    def Y(self, index: int) -> None:
        """ Pauli iY gate """
        self.sequence.append( (self.get_rotation_string('Y', [index]), self.pi_2_multiple(+2)) )
    
    def Z(self, index: int) -> None:
        """ Pauli iZ gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+2)) )

    def H(self, index: int) -> None:
        """ iH Hadamard gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+2)) )
        self.sequence.append( (self.get_rotation_string('Y', [index]), self.pi_2_multiple(+1)) )
        
    def S(self, index: int) -> None:
        """ √(+i)S gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+1)) )

    def Sdag(self, index: int) -> None:
        """ -√(-i)S† gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+3)) )

    def sqrtX(self, index: int) -> None:
        """ √(iX) gate """
        self.sequence.append( (self.get_rotation_string('X', [index]), self.pi_2_multiple(+1)) )

    def sqrtY(self, index: int) -> None:
        """ √(iY) gate """
        self.sequence.append( (self.get_rotation_string('Y', [index]), self.pi_2_multiple(+1)) )

    def sqrtZ(self, index: int) -> None:
        """ √(iZ) gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+1)) )

    def CX(self, control: int, target: int) -> None:
        """ √(-i)CX gate """
        self.sequence.append( (self.get_rotation_string('ZX', [control, target]), self.pi_2_multiple(+1)) )
        self.sequence.append( (self.get_rotation_string('ZI', [control, target]), self.pi_2_multiple(+3)) )
        self.sequence.append( (self.get_rotation_string('IX', [control, target]), self.pi_2_multiple(+3)) )
    
    def CY(self, control: int, target: int) -> None:
        """ √(-i)CY gate """
        self.sequence.append( (self.get_rotation_string('ZY', [control, target]), self.pi_2_multiple(+1)) )
        self.sequence.append( (self.get_rotation_string('ZI', [control, target]), self.pi_2_multiple(+3)) )
        self.sequence.append( (self.get_rotation_string('IY', [control, target]), self.pi_2_multiple(+3)) )

    def CZ(self, control: int, target: int) -> None:
        """ √(-i)CZ gate """
        self.sequence.append( (self.get_rotation_string('ZZ', [control, target]), self.pi_2_multiple(+1)) )
        self.sequence.append( (self.get_rotation_string('ZI', [control, target]), self.pi_2_multiple(+3)) )
        self.sequence.append( (self.get_rotation_string('IZ', [control, target]), self.pi_2_multiple(+3)) )

    def SWAP(self, qubit_1: int, qubit_2: int) -> None:
        """ Swap qubits 1 and 2 """
        self.CX(qubit_1,qubit_2)
        self.CX(qubit_2,qubit_1)
        self.CX(qubit_1,qubit_2)
        
    #################################################################
    ####################  NON-CLIFFORD GATES  #######################
    #################### WARNING: INEFFICIENT #######################
    #################################################################
        
    def R(self, pauli: str, indices: List[int], angle: float) -> None:
        """ Arbitrary rotation gate """
        self.sequence.append( (self.get_rotation_string(pauli, indices), -angle) )
    
    def RX(self, index: int, angle: float) -> None:
        """ Pauli X rotation """
        self.R('X', [index], angle)

    def RY(self, index: int, angle: float) -> None:
        """ Pauli Y rotation """
        self.R('Y', [index], angle)

    def RZ(self, index: int, angle: float) -> None:
        """ Pauli Z rotation """
        self.R('Z', [index], angle)

    def T(self, index: int, angle: float) -> None:
        """ T gate """
        raise NotImplementedError()
    
    def Toffoli(self, control_1: int, control_2: int, target: int) -> None:
        """ Doubly-controlled X gate """
        raise NotImplementedError()

    #################################################################
    ######################  GATE EXECUTION  #########################
    #################################################################

    def apply_sequence(self, operator: PauliwordOp) -> PauliwordOp:
        """ Apply the stored sequence of rotations on the input operator
        """
        assert operator.n_qubits == self.n_qubits, 'The operator is defined over a different number of qubits'
        return operator.perform_rotations(self.sequence[::-1])
    
    def evaluate(self, operator: PauliwordOp) -> float:
        """ Evaluate the stored rotations on the input operator
        """
        rotated_op = self.apply_sequence(operator)
        expval = 0
        for rotated_str, coeff in rotated_op.to_dictionary.items():
            if set(rotated_str).issubset({'I','Z'}):
                expval += coeff
        return expval
        
    @classmethod
    def from_qasm(cls, qasm: str, angle_factor: int=1) -> "CircuitSymmerlator":
        """ Initialize the simulator from a QASM circuit
        """
        instructions = qasm.split(';\n')[:-1]
        qasm_version = instructions.pop(0)
        inclusions   = instructions.pop(0)
        registers    = instructions.pop(0)
        # n_qubits     = int(registers.split(' ')[1][2:-1])
        n_qubits =   int(re.findall(r'\d+', registers)[0])

        self = cls(n_qubits)
        pi = np.pi # for evaluating strings like '3*pi/2'
        for step in instructions:
            gate_qubits  = step.split(' ')
            gate = gate_qubits[0]
            qubits = ''.join(gate_qubits[1:])
            qubits = [int(q[2:-1]) for q in qubits.split(',')]
            extract_angle = gate.split('(')
            if len(extract_angle) == 1:
                gate  = extract_angle[0]
                angle = None
            else:
                gate, angle = extract_angle
                angle = eval(angle[:-1])
            if angle is not None:
                self.gate_map[gate](*qubits, angle=angle_factor*angle)
            else:
                self.gate_map[gate](*qubits)
        return self
    
    @classmethod
    def from_qiskit(cls, circuit: QuantumCircuit) -> "CircuitSymmerlator":
        """ Initialize the simulator from a Qiskit QuantumCircuit
        """
        return cls.from_qasm(qasm3.dumps(circuit.reverse_bits()), angle_factor=-1)
