from symmer import PauliwordOp, QuantumState
import numpy as np

class CircuitSymmerlator:
    """
    Symmer circuit simulator

    Each Clifford gate is expanded as a sequence of pi/2 rotations
    (correct up to some global phase that cancels in the inner product calculation).

    Non-Clifford gates also included, although note the use of these is inefficient.
    
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.sequence = []

    def get_rotation_string(self, pauli, indices):
        """
        """
        pauli = list(pauli)
        assert len(pauli)==len(indices), 'Number of Paulis and indices do not match'
        assert set(pauli).issubset({'I','X','Y','Z'}), 'Pauli operators are either I, X, Y or Z.'
        R = ['I']*self.n_qubits
        for i,P in zip(indices, pauli):
            R[i] = P
        return PauliwordOp.from_list([''.join(R)])
    
    def pi_2_multiple(self, multiple):
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
    
    def X(self, index):
        """ Pauli iX gate """
        self.sequence.append( (self.get_rotation_string('X', [index]), self.pi_2_multiple(+2)) )
    
    def Y(self, index):
        """ Pauli iY gate """
        self.sequence.append( (self.get_rotation_string('Y', [index]), self.pi_2_multiple(+2)) )
    
    def Z(self, index):
        """ Pauli iZ gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+2)) )

    def H(self, index):
        """ iH Hadamard gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+2)) )
        self.sequence.append( (self.get_rotation_string('Y', [index]), self.pi_2_multiple(+1)) )
        
    def S(self, index):
        """ √(+i)S gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+1)) )

    def Sdag(self, index):
        """ -√(-i)S† gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+3)) )

    def sqrtX(self, index):
        """ √(iX) gate """
        self.sequence.append( (self.get_rotation_string('X', [index]), self.pi_2_multiple(+1)) )

    def sqrtY(self, index):
        """ √(iY) gate """
        self.sequence.append( (self.get_rotation_string('Y', [index]), self.pi_2_multiple(+1)) )

    def sqrtZ(self, index):
        """ √(iZ) gate """
        self.sequence.append( (self.get_rotation_string('Z', [index]), self.pi_2_multiple(+1)) )

    def CX(self, control, target):
        """ √(-i)CX gate """
        self.sequence.append( (self.get_rotation_string('ZX', [control, target]), self.pi_2_multiple(+1)) )
        self.sequence.append( (self.get_rotation_string('ZI', [control, target]), self.pi_2_multiple(+3)) )
        self.sequence.append( (self.get_rotation_string('IX', [control, target]), self.pi_2_multiple(+3)) )
    
    def CY(self, control, target):
        """ √(-i)CY gate """
        self.sequence.append( (self.get_rotation_string('ZY', [control, target]), self.pi_2_multiple(+1)) )
        self.sequence.append( (self.get_rotation_string('ZI', [control, target]), self.pi_2_multiple(+3)) )
        self.sequence.append( (self.get_rotation_string('IY', [control, target]), self.pi_2_multiple(+3)) )

    def CZ(self, control, target):
        """ √(-i)CZ gate """
        self.sequence.append( (self.get_rotation_string('ZZ', [control, target]), self.pi_2_multiple(+1)) )
        self.sequence.append( (self.get_rotation_string('ZI', [control, target]), self.pi_2_multiple(+3)) )
        self.sequence.append( (self.get_rotation_string('IZ', [control, target]), self.pi_2_multiple(+3)) )
    
    #################################################################
    ####################  NON-CLIFFORD GATES  #######################
    #################### WARNING: INEFFICIENT #######################
    #################################################################
        
    def R(self, pauli, indices, angle):
        """ Arbitrary rotation gate """
        self.sequence.append( (self.get_rotation_string(pauli, indices), -angle) )
    
    def RX(self, index, angle):
        """ Pauli X rotation """
        self.R('X', [index], angle)

    def RY(self, index, angle):
        """ Pauli Y rotation """
        self.R('Y', [index], angle)

    def RZ(self, index, angle):
        """ Pauli Z rotation """
        self.R('Z', [index], angle)

    def T(self, index, angle):
        """ T gate """
        raise NotImplementedError()
    
    def Toffoli(self, control_1, control_2, target):
        """ Doubly-controlled X gate """
        raise NotImplementedError()

    #################################################################
    ######################  GATE EXECUTION  #########################
    #################################################################

    def apply_sequence(self, operator):
        """
        """
        assert operator.n_qubits == self.n_qubits, 'The operator is defined over a different number of qubits'
        return operator.perform_rotations(self.sequence[::-1])
    
    def evaluate(self, operator):
        rotated_op = self.apply_sequence(operator)
        zero_state = QuantumState([0]*self.n_qubits)
        return rotated_op.expval(zero_state)
    
    def parse_qasm(self, qasm):
        """ Convert qasm circuit into sequence of rotations
        """
        raise NotImplementedError()