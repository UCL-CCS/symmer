import os
import json
import pytest
import numpy as np
from symmer.projection.utils import *
from symmer import QubitTapering
from symmer.operators import PauliwordOp, IndependentOp

test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ham_data_dir = os.path.join(test_dir, 'hamiltonian_data')

with open(os.path.join(ham_data_dir, 'Be_STO-3G_SINGLET_JW.json'), 'r') as f:
    H_data = json.load(f)
H_op = PauliwordOp.from_dictionary(H_data['hamiltonian'])
CC_op = PauliwordOp.from_dictionary(H_data['data']['auxiliary_operators']['UCCSD_operator'])
QT = QubitTapering(H_op)
H_taper = QT.taper_it(ref_state=H_data['data']['hf_array'])
CC_taper = QT.taper_it(aux_operator=CC_op)

def test_norm():
    arr = np.random.random(100)
    assert np.isclose(np.linalg.norm(arr), norm(arr))

def test_lp_norm():
    arr = np.random.random(100)
    p = np.random.randint(1, 10)
    assert np.isclose(np.linalg.norm(arr, ord=p),  lp_norm(arr, p=p))

def test_update_eigenvalues_insufficient_basis():
    G1 = IndependentOp.from_list(['IZ', 'ZI'])
    G2 = IndependentOp.from_list(['ZZ', 'XX'])
    with pytest.raises(ValueError):
        update_eigenvalues(G1, G2)

def test_update_eigenvalues_correct_usage():
    G1 = IndependentOp.from_dictionary({'ZII':-1, 'ZZI':1, 'IZZ':-1})
    G2 = IndependentOp.from_list(['ZZZ', 'IIZ', 'ZIZ'])
    update_eigenvalues(basis=G1, stabilizers=G2)
    assert np.all(G2.coeff_vec == np.array([+1, +1, -1]))

def test_basis_weighting():
    weighting_operator = PauliwordOp.from_list(['XYZX', 'YYYY', 'ZZZZ', 'IXZX', 'YXZI'])
    SI = StabilizerIdentification(weighting_operator=weighting_operator, use_X_only=True)
    assert SI.basis_weighting == PauliwordOp.from_list(
        ['XXIX', 'XXXX', 'IIII', 'IXIX', 'XXII']
    )

def test_symmetry_generators_by_term_significance():
    SI = StabilizerIdentification(weighting_operator=CC_taper, use_X_only=True)
    G = SI.symmetry_generators_by_term_significance(n_preserved=4)
    assert G == IndependentOp.from_list(['IZZZZ'])

def symmetry_generators_by_subspace_dimension():
    SI = StabilizerIdentification(weighting_operator=CC_taper, use_X_only=True)
    G = SI.symmetry_generators_by_subspace_dimension(n_sim_qubits=3)
    assert G == IndependentOp.from_list(['ZIZZZ', 'IZZZZ'])

def symmetry_generators_by_subspace_dimension_sweep():
    SI = StabilizerIdentification(weighting_operator=CC_taper)
    for n_q in range(6):
        G = SI.symmetry_generators_by_subspace_dimension(n_sim_qubits=n_q)
        assert n_q == H_taper.n_qubits - G.n_terms

