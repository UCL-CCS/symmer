import numpy as np
from symmer.operators import PauliwordOp
from symmer.evolution import trotter

#############################################
# Gate library decomposed into PauliwordOps #
#############################################

def I(n_qubits:int) -> PauliwordOp:
    return PauliwordOp.from_dictionary({'I'*n_qubits:1})

def X(n_qubits:int, index:int) -> PauliwordOp:
    X_str = ['I']*n_qubits; X_str[index] = 'X'
    return PauliwordOp.from_dictionary({''.join(X_str):1})

def Y(n_qubits:int, index:int) -> PauliwordOp:
    Y_str = ['I']*n_qubits; Y_str[index] = 'Y'
    return PauliwordOp.from_dictionary({''.join(Y_str):1})

def Z(n_qubits:int, index:int) -> PauliwordOp:
    Z_str = ['I']*n_qubits; Z_str[index] = 'Z'
    return PauliwordOp.from_dictionary({''.join(Z_str):1})

def Had(n_qubits:int, index:int) -> PauliwordOp:
    """ Hadamard gate
    """
    return (
        Z(n_qubits, index).multiply_by_constant(1/np.sqrt(2))+
        X(n_qubits, index).multiply_by_constant(1/np.sqrt(2))
    )

def CZ(n_qubits:int, control:int, target:int) -> PauliwordOp:
    """ Controlled Z gate
    """
    ZI = Z(n_qubits, control)
    IZ = Z(n_qubits, target)
    ZZ = ZI * IZ
    
    CZ_exp = (ZZ - IZ - ZI).multiply_by_constant(np.pi/4)
    CZ = trotter(CZ_exp.multiply_by_constant(1j), trotnum=1).multiply_by_constant(np.sqrt(1j))
    return CZ

def CX(n_qubits:int, control:int, target:int) -> PauliwordOp:
    """ Controlled X gate
    """
    _Had = Had(n_qubits, target)
    return _Had * CZ(n_qubits, control, target) * _Had

def RX(n_qubits:int, index:int, angle:float) -> PauliwordOp:
    return trotter(X(n_qubits, index).multiply_by_constant(1j*angle/2))

def RY(n_qubits:int, index:int, angle:float) -> PauliwordOp:
    return trotter(Y(n_qubits, index).multiply_by_constant(1j*angle/2))

def RZ(n_qubits:int, index:int, angle:float) -> PauliwordOp:
    return trotter(Z(n_qubits, index).multiply_by_constant(1j*angle/2))

def U1(n_qubits:int, index:int, angle:float) -> PauliwordOp:
    return RZ(n_qubits, index, angle).multiply_by_constant(np.exp(1j*angle/2))

def S(n_qubits:int, index:int) -> PauliwordOp:
    return RZ(n_qubits,index,-np.pi/2).multiply_by_constant(np.sqrt(1j))
