"""Smoke tests for VQE_Driver public surface."""
import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit, Parameter

from symmer.operators import PauliwordOp, QuantumState
from symmer.evolution import VQE_Driver


def test_vqe_driver_smoke_two_qubit():
    """Construct a 2-qubit VQE driver, evaluate energy/grad, run a few iterations."""
    H = PauliwordOp.from_list(['ZI', 'IZ', 'XX'], [1.0, 1.0, 0.5])
    theta = Parameter('θ')
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0, 1)

    driver = VQE_Driver(observable=H, ansatz_circuit=qc)
    driver.verbose = False
    driver.expectation_eval = 'sparse_array'

    x0 = np.array([0.1])
    energy = driver.f(x0)
    grad = driver.gradient(x0)
    assert np.isfinite(energy)
    assert grad.shape == (1,)

    # L-BFGS-B returns jac so serialize_opt_data succeeds
    opt_out, history = driver.run(x0=x0, method='L-BFGS-B', options={'maxiter': 3})
    assert 'energy' in history
    assert len(history['energy']) >= 1
    assert np.isfinite(list(history['energy'].values())[0])
    assert 'fun' in opt_out
