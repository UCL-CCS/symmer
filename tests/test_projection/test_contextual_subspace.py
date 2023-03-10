import os
import json
import pytest
import numpy as np
from symmer.projection.utils import *
from symmer import QubitTapering, ContextualSubspace, QuantumState
from symmer.operators import PauliwordOp, IndependentOp
from symmer.evolution import trotter
from symmer.utils import exact_gs_energy

ham_data_dir = os.path.join(os.getcwd(), 'tests/hamiltonian_data')
with open(f'{ham_data_dir}/Be_STO-3G_SINGLET_JW.json', 'r') as f:
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
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.00035

def random_stabilizers():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    CS.update_stabilizers(3, strategy='random')
    H_cs = CS.project_onto_subspace()
    assert CS.n_qubits_in_subspace == 3
    assert H_cs.n_qubits == 3
    
def test_update_stabilizers_aux_preserving():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving')
    H_cs = CS.project_onto_subspace()
    assert CS.n_qubits_in_subspace == 3
    assert H_cs.n_qubits == 3
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.00035

# HOMO-LUMO biasing non-deterministic, occasionally this test fails so commented out for now.
# def test_update_stabilizers_HOMO_LUMO_biasing():
#     CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
#     CS.update_stabilizers(3, aux_operator=CC_taper, strategy='HOMO_LUMO_biasing', HF_array=QT.tapered_ref_state.state_matrix)
#     H_cs = CS.project_onto_subspace()
#     assert CS.n_qubits_in_subspace == 3
#     assert H_cs.n_qubits == 3
#     assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.00035

def test_StabilizeFirst_strategy_too_many_cliques():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='StabilizeFirst_magnitude')
    with pytest.raises(RuntimeError):
        CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving', n_cliques=5)

def test_StabilizeFirst_strategy_correct_usage():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='StabilizeFirst_magnitude')
    CS.update_stabilizers(3, aux_operator=CC_taper, strategy='aux_preserving', n_cliques=3)
    H_cs = CS.project_onto_subspace()
    assert H_cs.n_qubits == 3
    assert abs(exact_gs_energy(H_cs.to_sparse_matrix)[0] - fci_energy) < 0.00031 # the extra clique actually allows StabilizeFirst to do better!

def test_project_auxiliary_operator():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    G = IndependentOp.from_list(['ZIZZZ', 'IZZZZ'])
    CS.manual_stabilizers(G)
    H_cs = CS.project_onto_subspace()
    CC_cs = CS.project_onto_subspace(operator_to_project=CC_taper)
    assert CC_cs.n_qubits == 3
    assert abs(H_cs.expval(trotter(CC_cs*1j, trotnum=10) * QuantumState([0,0,0])) - fci_energy) < 0.0004

def test_hamiltonian():
    CS = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude')
    G = IndependentOp.from_list(['ZIZZZ', 'IZZZZ'])
    CS.manual_stabilizers(G)
    H_cs = CS.project_onto_subspace()
    assert H_cs == CS.hamiltonian(3, aux_operator=CC_taper)
    
