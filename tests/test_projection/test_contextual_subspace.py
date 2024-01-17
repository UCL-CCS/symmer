import os
import json
import pytest
import numpy as np
import multiprocessing as mp
from symmer.projection.utils import *
from symmer import QubitTapering, ContextualSubspace, QuantumState
from symmer.operators import PauliwordOp, IndependentOp, NoncontextualOp
from symmer.evolution import trotter
from symmer.utils import exact_gs_energy

test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ham_data_dir = os.path.join(test_dir, 'hamiltonian_data')

with open(os.path.join(ham_data_dir, 'Be_STO-3G_SINGLET_JW.json'), 'r') as f:
    H_data = json.load(f)
hf_energy = H_data['data']['calculated_properties']['HF']['energy']
fci_energy = H_data['data']['calculated_properties']['FCI']['energy']
H_op = PauliwordOp.from_dictionary(H_data['hamiltonian'])
CC_op = PauliwordOp.from_dictionary(H_data['data']['auxiliary_operators']['UCCSD_operator'])
QT = QubitTapering(H_op)
H_taper = QT.taper_it(ref_state=H_data['data']['hf_array'])
CC_taper = QT.taper_it(aux_operator=CC_op)

def test_noncontextual_operator():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    assert CS.noncontextual_operator.is_noncontextual
    assert not CS.contextual_operator.is_noncontextual

def test_random_stabilizers():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    CS.update_stabilizers(3, strategy='random')
    H_cs = CS.project_onto_subspace()
    assert CS.n_qubits_in_subspace == 3
    assert H_cs.n_qubits == 3

def test_noncontextual_ground_state():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    assert np.isclose(CS.noncontextual_operator.energy, hf_energy)

def test_manual_stabilizers():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    G = IndependentOp.from_list(['ZIZZZ', 'IZZZZ'])
    CS.manual_stabilizers(G)
    H_cs = CS.project_onto_subspace()
    assert CS.n_qubits_in_subspace == 3
    assert H_cs.n_qubits == 3
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.0005
    
def test_update_stabilizers_aux_preserving():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving')
    H_cs = CS.project_onto_subspace()
    assert CS.n_qubits_in_subspace == 3
    assert H_cs.n_qubits == 3
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.0005

def test_update_stabilizers_unrecognised_strategy():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    with pytest.raises(ValueError):
        CS.update_stabilizers(3, aux_operator=CC_taper, strategy='symmer')

def test_update_stabilizers_HOMO_LUMO_biasing():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='HOMO_LUMO_biasing', HF_array=QT.tapered_ref_state.state_matrix)
    H_cs = CS.project_onto_subspace()
    assert CS.n_qubits_in_subspace == 3
    assert H_cs.n_qubits == 3
    samples = []
    # HOMO-LUMO biasing non-deterministic, so run a few instances to make sure the target error is achieved.
    #####

    # global func
    # def func(i):
    #     CS.update_stabilizers(3, aux_operator=CC_taper, strategy='HOMO_LUMO_biasing', HF_array=QT.tapered_ref_state.state_matrix)
    #     return abs(exact_gs_energy(CS.project_onto_subspace().to_sparse_matrix)[0] - fci_energy)
    # with mp.Pool(mp.cpu_count()) as pool:
    #     samples = pool.map(func, range(10))

    for _ in range(10):
        CS.update_stabilizers(3, aux_operator=CC_taper, strategy='HOMO_LUMO_biasing',
                              HF_array=QT.tapered_ref_state.state_matrix)
        samples.append(abs(exact_gs_energy(CS.project_onto_subspace().to_sparse_matrix)[0] - fci_energy))

    assert min(samples) < 0.004

def test_StabilizeFirst_strategy_correct_usage():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='StabilizeFirst')
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving')
    H_cs = CS.project_onto_subspace()
    assert H_cs.n_qubits == 3
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.0005

@pytest.mark.parametrize("ref_state", [QT.tapered_ref_state, QT.tapered_ref_state.state_matrix[0]])
def test_reference_state(ref_state):
    CS = ContextualSubspace(
        H_taper, noncontextual_strategy='StabilizeFirst',
        reference_state=ref_state
    )
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving')
    H_cs = CS.project_onto_subspace()
    assert H_cs.n_qubits == 3
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.0005

def test_StabilizeFirst_strategy_correct_usage():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='StabilizeFirst')
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving')
    H_cs = CS.project_onto_subspace()
    assert H_cs.n_qubits == 3
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.0005

def test_project_auxiliary_operator():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    G = IndependentOp.from_list(['ZIZZZ', 'IZZZZ'])
    CS.manual_stabilizers(G)
    H_cs = CS.project_onto_subspace()
    CC_cs = CS.project_onto_subspace(operator_to_project=CC_taper)
    assert CC_cs.n_qubits == 3
    assert abs(H_cs.expval(trotter(CC_cs*1j, trotnum=10) * QuantumState([0,0,0])) - fci_energy) < 0.0005

def test_no_aux_operator_provided():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    CS.update_stabilizers(3, aux_operator=None, strategy='aux_preserving')

def test_StabilizeFirst_no_aux_operator_provided():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='StabilizeFirst')
    CS.update_stabilizers(3, aux_operator=None, strategy='aux_preserving')

def test_operator_already_noncontextual():
    with pytest.raises(ValueError):
        CS = ContextualSubspace(NoncontextualOp.from_hamiltonian(H_taper))

@pytest.mark.parametrize("up_method", ['LCU', 'seq_rot'])
def test_unitary_partitioning_method(up_method):
    CS = ContextualSubspace(
        H_taper, noncontextual_strategy='SingleSweep_magnitude', 
        unitary_partitioning_method=up_method
    )
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving')
    H_cs = CS.project_onto_subspace()
    assert H_cs.n_qubits == 3
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.0005

@pytest.mark.parametrize("up_method", ['LCU', 'seq_rot'])
def test_project_state_onto_subspace(up_method):
    CS = ContextualSubspace(
        H_taper, noncontextual_strategy='SingleSweep_magnitude',
        unitary_partitioning_method=up_method
    )
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving')
    CS.project_onto_subspace()
    projected_state = CS.project_state_onto_subspace(QT.tapered_ref_state)
    assert projected_state == QuantumState([[0,0,0]], [-1])

def test_project_state_onto_subspace_before_operator():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='StabilizeFirst')
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving')
    with pytest.raises(AssertionError):
        CS.project_state_onto_subspace(QT.tapered_ref_state)