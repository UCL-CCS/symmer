import pytest
import numpy as np
from symmer import PauliwordOp
from symmer.evolution.gate_library import *
from symmer.evolution.circuit_symmerlator import CircuitSymmerlator
from qiskit.quantum_info import Statevector, random_clifford

def test_random_cliffords():
    """ Test the simulator on lots of random 5-qubit Clifford circuits
    """
    for _ in range(50):
        n_q = 5
        # generate random Clifford circuit
        qc = random_clifford(n_q).to_circuit()
        sv = Statevector(qc).data.reshape(-1,1)
        # ... and a random observable
        observable = PauliwordOp.random(n_q, 100, complex_coeffs=False)
        observable_matrix = observable.to_sparse_matrix.toarray() # reverse order because qiskit
        # Than, intialize the circuit simulator and check it matches direct statevector calculation.
        CS = CircuitSymmerlator.from_qiskit(qc)
        assert np.isclose(
            (sv.conj().T @ observable_matrix @ sv)[0,0],
            CS.evaluate(observable)
        )

#### INDIVIDUAL GATE TESTS BELOW ####

one_qubit_paulis = [
    'I', 'X', 'Y', 'Z'
]
two_qubit_paulis = [
    'IZ','ZI','ZZ','IX','XI','XX','IY','YI','YY',
    'XY','XZ','YX','YZ','ZX','ZY','II'
]

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_X_gate(P):
    CS = CircuitSymmerlator(1)
    CS.X(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == X(1,0)*P*X(1,0)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_Y_gate(P):
    CS = CircuitSymmerlator(1)
    CS.Y(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == Y(1,0)*P*Y(1,0)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_Z_gate(P):
    CS = CircuitSymmerlator(1)
    CS.Z(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == Z(1,0)*P*Z(1,0)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_RX_gate(P):
    CS = CircuitSymmerlator(1)
    angle = np.random.random()
    CS.RX(0, angle)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == RX(1,0,-angle)*P*RX(1,0,angle)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_RY_gate(P):
    CS = CircuitSymmerlator(1)
    angle = np.random.random()
    CS.RY(0, angle)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == RY(1,0,-angle)*P*RY(1,0,angle)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_RZ_gate(P):
    CS = CircuitSymmerlator(1)
    angle = np.random.random()
    CS.RZ(0, angle)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == RZ(1,0,-angle)*P*RZ(1,0,angle)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_H_gate(P):
    CS = CircuitSymmerlator(1)
    CS.H(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == Had(1,0)*P*Had(1,0)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_S_gate(P):
    CS = CircuitSymmerlator(1)
    CS.S(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == S(1,0).dagger * P * S(1,0)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_Sdag_gate(P):
    CS = CircuitSymmerlator(1)
    CS.Sdag(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == S(1,0) * P * S(1,0).dagger

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_sqrtX_gate(P):
    CS = CircuitSymmerlator(1)
    CS.sqrtX(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == RX(1,0,+np.pi/2)*P*RX(1,0,-np.pi/2)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_sqrtY_gate(P):
    CS = CircuitSymmerlator(1)
    CS.sqrtY(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == RY(1,0,+np.pi/2)*P*RY(1,0,-np.pi/2)

@pytest.mark.parametrize("P", one_qubit_paulis)
def test_sqrtZ_gate(P):
    CS = CircuitSymmerlator(1)
    CS.sqrtZ(0)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == RZ(1,0,+np.pi/2)*P*RZ(1,0,-np.pi/2)

@pytest.mark.parametrize("P", two_qubit_paulis)
def test_CX_gate(P):
    CS = CircuitSymmerlator(2)
    CS.CX(0,1)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == CX(2,0,1)*P*CX(2,0,1)

@pytest.mark.parametrize("P", two_qubit_paulis)
def test_CY_gate(P):
    CS = CircuitSymmerlator(2)
    CS.CY(0,1)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == CY(2,0,1)*P*CY(2,0,1)

@pytest.mark.parametrize("P", two_qubit_paulis)
def test_CZ_gate(P):
    CS = CircuitSymmerlator(2)
    CS.CZ(0,1)
    P = PauliwordOp.from_list([P])
    assert CS.apply_sequence(P) == CZ(2,0,1)*P*CZ(2,0,1)