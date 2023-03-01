import pytest
import numpy as np
from symmer.evolution import qasm_to_PauliwordOp
from symmer.symplectic import QuantumState
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import CircuitStateFn

@pytest.mark.parametrize("n_qubits", [1,2,3,4,5,6,7])
def test_against_qiskit(n_qubits):
    
    qc = EfficientSU2(n_qubits, reps=2).decompose()
    qc = qc.bind_parameters(np.random.random(qc.num_parameters))
    psi1 = CircuitStateFn(qc)
    
    qc_pwop = qasm_to_PauliwordOp(qc.qasm(), reverse=True)
    psi2 = qc_pwop * QuantumState([[0]*n_qubits])

    assert np.all(np.isclose(
            psi1.to_spmatrix().T.toarray(), psi2.to_sparse_matrix.toarray()
        )
    )