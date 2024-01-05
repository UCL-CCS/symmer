import numpy as np
from symmer.operators import PauliwordOp
from symmer.evolution import trotter

#############################################
# Gate library decomposed into PauliwordOps #
#############################################

def I(n_qubits:int) -> PauliwordOp:
    """
    Identity gate

    Args:
        n_qubits (int): Number of qubits

    Returns:
        PauliwordOp representing the identity operation ('I') applied to a system of 'n_qubits'.
    """
    return PauliwordOp.from_dictionary({'I'*n_qubits:1})

def X(n_qubits:int, index:int) -> PauliwordOp:
    """
    Pauli X gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the operation 'X' has to be applied.

    Returns:
        PauliwordOp representing the Pauli 'X' operator applied to the specified qubit index while leaving the other qubits unchanged.
    """
    X_str = ['I']*n_qubits; X_str[index] = 'X'
    return PauliwordOp.from_dictionary({''.join(X_str):1})

def Y(n_qubits:int, index:int) -> PauliwordOp:
    """
    Pauli Y gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the operation 'Y' has to be applied.

    Returns:
        PauliwordOp representing the Pauli 'Y' operator applied to the specified qubit index while leaving the other qubits unchanged.
    """
    Y_str = ['I']*n_qubits; Y_str[index] = 'Y'
    return PauliwordOp.from_dictionary({''.join(Y_str):1})

def Z(n_qubits:int, index:int) -> PauliwordOp:
    """
    Pauli Z gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the operation 'Z' has to be applied.

    Returns:
        PauliwordOp representing the Pauli 'Z' operator applied to the specified qubit index while leaving the other qubits unchanged.
    """
    Z_str = ['I']*n_qubits; Z_str[index] = 'Z'
    return PauliwordOp.from_dictionary({''.join(Z_str):1})

def Had(n_qubits:int, index:int) -> PauliwordOp:
    """ 
    Hadamard gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the operation 'H' has to be applied.

    Returns:
        PauliwordOp representing the 'H' operator applied to the specified qubit index while leaving the other qubits unchanged.
    """
    return (
        Z(n_qubits, index).multiply_by_constant(1/np.sqrt(2))+
        X(n_qubits, index).multiply_by_constant(1/np.sqrt(2))
    )

def CZ(n_qubits:int, control:int, target:int) -> PauliwordOp:
    """ 
    Controlled Z gate

    Args:
        n_qubits (int): Number of qubits.
        control (int): Qubit index at which will act as a control qubit.
        target (int): Qubit index at which the operation 'Z' has to be applied if the control qubit is in |1> state.

    Returns:
        PauliwordOp representing the Controlled-Z (CZ) gate applied to the specified control and target qubits in a system of 'n_qubits' qubits.
    """
    ZI = Z(n_qubits, control)
    IZ = Z(n_qubits, target)
    ZZ = ZI * IZ
    
    CZ_exp = (ZZ - IZ - ZI).multiply_by_constant(np.pi/4)
    CZ = trotter(CZ_exp.multiply_by_constant(1j), trotnum=1).multiply_by_constant(np.sqrt(1j))
    return CZ

def CX(n_qubits:int, control:int, target:int) -> PauliwordOp:
    """ 
    Controlled X gate

    Args:
        n_qubits (int): Number of qubits.
        control (int): Qubit index at which will act as a control qubit.
        target (int): Qubit index at which the operation 'X' has to be applied if the control qubit is in |1> state.

    Returns:
        PauliwordOp representing the Controlled-X (CX) gate applied to the specified control and target qubits in a system of 'n_qubits' qubits.
    """
    _Had = Had(n_qubits, target)
    return _Had * CZ(n_qubits, control, target) * _Had

def CY(n_qubits:int, control:int, target:int) -> PauliwordOp:
    """ 
    Controlled Y gate

    Args:
        n_qubits (int): Number of qubits.
        control (int): Qubit index at which will act as a control qubit.
        target (int): Qubit index at which the operation 'X' has to be applied if the control qubit is in |1> state.

    Returns:
        PauliwordOp representing the Controlled-X (CX) gate applied to the specified control and target qubits in a system of 'n_qubits' qubits.
    """
    _Had = Had(n_qubits, target)
    _S   = S(n_qubits, target)
    return _S * _Had * CZ(n_qubits, control, target) * _Had * _S.dagger

def RX(n_qubits:int, index:int, angle:float) -> PauliwordOp:
    """ 
    Rotation-X gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the operation 'RX' has to be applied.
        angle (float): The angle by which the qubit state has to be rotated around the X-axis.

    Returns:
        PauliwordOp representing the rotation around the X-axis (RX) gate by the specified angle applied to a specific qubit in a system of 'n_qubits'.
    """
    return trotter(X(n_qubits, index).multiply_by_constant(1j*angle/2))

def RY(n_qubits:int, index:int, angle:float) -> PauliwordOp:
    """ 
    Rotation-Y gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the operation 'RY' has to be applied.
        angle (float): The angle by which the qubit state has to be rotated around the Y-axis.

    Returns:
        PauliwordOp representing the rotation around the Y-axis (RY) gate by the specified angle applied to a specific qubit in a system of 'n_qubits'.
    """
    return trotter(Y(n_qubits, index).multiply_by_constant(1j*angle/2))

def RZ(n_qubits:int, index:int, angle:float) -> PauliwordOp:
    """ 
    Rotation-Z gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the operation 'RZ' has to be applied.
        angle (float): The angle by which the qubit state has to be rotated around the Z-axis.

    Returns:
        PauliwordOp representing the rotation around the Z-axis (RZ) gate by the specified angle applied to a specific qubit in a system of 'n_qubits'.
    """
    return trotter(Z(n_qubits, index).multiply_by_constant(1j*angle/2))

def U1(n_qubits:int, index:int, angle:float) -> PauliwordOp:
    """ 
    U1 gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the phase shift is to be introduced to the qubit state.
        angle (float): Phase angle.

    Returns:
        PauliwordOp representing the U1 gate with phase 'angle' applied to a specific qubit in a system of 'n_qubits'.
    """
    return RZ(n_qubits, index, angle).multiply_by_constant(np.exp(1j*angle/2))

def S(n_qubits:int, index:int) -> PauliwordOp:
    """
    S gate

    Args:
        n_qubits (int): Number of qubits.
        index (int): Qubit index at which the operation 'S' has to be applied.

    Returns:
        PauliwordOp representing the 'S' operator applied to the specified qubit index while leaving the other qubits unchanged.
    """
    return RZ(n_qubits,index,-np.pi/2).multiply_by_constant(np.sqrt(1j))
