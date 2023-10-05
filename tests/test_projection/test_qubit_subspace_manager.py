import os
import json
import pytest
import numpy as np
from numbers import Number
from symmer import QubitSubspaceManager, PauliwordOp, QuantumState
from symmer.utils import exact_gs_energy

test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ham_data_dir = os.path.join(test_dir, "hamiltonian_data")

with open(os.path.join(ham_data_dir, "Be_STO-3G_SINGLET_JW.json"), "r") as f:
    H_data = json.load(f)
hf_energy = H_data["data"]["calculated_properties"]["HF"]["energy"]
fci_energy = H_data["data"]["calculated_properties"]["FCI"]["energy"]
H_op = PauliwordOp.from_dictionary(H_data["hamiltonian"])
CC_op = PauliwordOp.from_dictionary(
    H_data["data"]["auxiliary_operators"]["UCCSD_operator"]
)
HF_state = QuantumState(H_data["data"]["hf_array"])


def test_correct_qubit_numbers():
    QSM = QubitSubspaceManager(hamiltonian=H_op, ref_state=HF_state)
    for n_qubits in range(1, H_op.n_qubits + 1):
        H_reduced = QSM.get_reduced_hamiltonian(n_qubits, aux_operator=CC_op)
        CC_reduced = QSM.project_auxiliary_operator(CC_op)
        HF_reduced = QSM.project_auxiliary_state(HF_state)
        assert H_reduced.n_qubits == n_qubits
        assert CC_reduced.n_qubits == n_qubits
        assert HF_reduced.n_qubits == n_qubits


def test_correct_qubit_numbers_no_tapering():
    QSM = QubitSubspaceManager(
        hamiltonian=H_op, ref_state=HF_state, run_qubit_tapering=False
    )
    for n_qubits in range(1, H_op.n_qubits - 4):
        H_reduced = QSM.get_reduced_hamiltonian(n_qubits, aux_operator=CC_op)
        CC_reduced = QSM.project_auxiliary_operator(CC_op)
        HF_reduced = QSM.project_auxiliary_state(HF_state)
        assert H_reduced.n_qubits == n_qubits
        assert CC_reduced.n_qubits == n_qubits
        assert HF_reduced.n_qubits == n_qubits


def test_no_contextual_subspace():
    QSM = QubitSubspaceManager(
        hamiltonian=H_op,
        ref_state=HF_state,
        run_contextual_subspace=False,
        run_qubit_tapering=True,
    )
    with pytest.warns():
        H_reduced = QSM.get_reduced_hamiltonian(3)
    assert H_reduced.n_qubits == QSM._hamiltonian.n_qubits


def test_no_subspace_methods():
    QSM = QubitSubspaceManager(
        hamiltonian=H_op,
        ref_state=HF_state,
        run_contextual_subspace=False,
        run_qubit_tapering=False,
    )
    with pytest.warns():
        H_reduced = QSM.get_reduced_hamiltonian(3)
    assert H_reduced.n_qubits == QSM._hamiltonian.n_qubits


def test_too_many_qubits():
    QSM = QubitSubspaceManager(hamiltonian=H_op, ref_state=HF_state)
    with pytest.warns():
        H_reduced = QSM.get_reduced_hamiltonian(15)
    assert H_reduced.n_qubits == QSM.hamiltonian.n_qubits


def test_subspace_errors():
    errors = [0.031, 0.015, 0.00035, 0.0002, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]
    QSM = QubitSubspaceManager(hamiltonian=H_op, ref_state=HF_state)
    for index, n_qubits in enumerate(range(1, H_op.n_qubits + 1)):
        H_reduced = QSM.get_reduced_hamiltonian(n_qubits, aux_operator=CC_op)
        cs_error = abs(exact_gs_energy(H_reduced.to_sparse_matrix)[0] - fci_energy)
        assert cs_error < errors[index]


def test_return_noncontextual_energy():
    QSM = QubitSubspaceManager(hamiltonian=H_op, ref_state=HF_state)
    H_reduced = QSM.get_reduced_hamiltonian(0, aux_operator=CC_op)
    assert isinstance(H_reduced, Number)
    assert H_reduced < hf_energy and H_reduced > fci_energy


def test_no_input_ref_state():
    with pytest.warns():
        QSM = QubitSubspaceManager(hamiltonian=H_op)
