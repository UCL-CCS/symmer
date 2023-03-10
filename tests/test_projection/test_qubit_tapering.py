import pytest
import numpy as np
from symmer import PauliwordOp, QubitTapering, QuantumState
from symmer.operators import IndependentOp
from symmer.utils import exact_gs_energy
from symmer.evolution import trotter

H2_op = PauliwordOp.from_dictionary(
    {
        'IIII': (-0.05933866442819677+0j),
        'IIIZ': (-0.23676939575319134+0j),
        'IIZI': (-0.23676939575319134+0j),
        'IIZZ': (0.17571274411978302+0j),
        'IZII': (0.17579122569046912+0j),
        'IZIZ': (0.12223870791335416+0j),
        'IZZI': (0.16715312911492025+0j),
        'ZIII': (0.17579122569046912+0j),
        'ZIIZ': (0.16715312911492025+0j),
        'ZIZI': (0.12223870791335416+0j),
        'ZZII': (0.17002500620877006+0j),
        'XXYY': (-0.044914421201566114+0j),
        'XYYX': (0.044914421201566114+0j),
        'YXXY': (0.044914421201566114+0j),
        'YYXX': (-0.044914421201566114+0j)
    }
)
CC_op = PauliwordOp.from_dictionary(
    {
        'XXXX': (-0.006725473252131252+0j),
        'XXXY': 0.006725473252131252j,
        'XXYX': 0.006725473252131252j,
        'XXYY': (0.006725473252131252+0j),
        'XYXX': -0.006725473252131252j,
        'XYXY': (-0.006725473252131252+0j),
        'XYYX': (-0.006725473252131252+0j),
        'XYYY': 0.006725473252131252j,
        'YXXX': -0.006725473252131252j,
        'YXXY': (-0.006725473252131252+0j),
        'YXYX': (-0.006725473252131252+0j),
        'YXYY': 0.006725473252131252j,
        'YYXX': (0.006725473252131252+0j),
        'YYXY': -0.006725473252131252j,
        'YYYX': -0.006725473252131252j,
        'YYYY': (-0.006725473252131252+0j)
    }
)
hf_energy   = -1.117505831043514
ccsd_energy = -1.1368383583027837
fci_energy  = -1.1368382276023516
hf_state = QuantumState([1, 1, 0, 0])
ccsd_state = trotter(CC_op, trotnum=20) * hf_state
QT = QubitTapering(H2_op)

def test_init():
    assert QT.operator==H2_op
    assert QT.n_taper ==3

def test_symmetry_generators_H2():
    G1 = QT.symmetry_generators
    G2 = IndependentOp.from_list(['ZIZI', 'IZIZ', 'IIZZ'])
    assert (
        np.all(G1.basis_reconstruction(G2)[1]) and
        np.all(G2.basis_reconstruction(G1)[1])
    )

def test_taper_H2_hamiltonian():
    H2_taper = QT.taper_it(ref_state=hf_state)
    assert H2_taper.n_qubits == 1
    assert np.isclose(exact_gs_energy(H2_taper.to_sparse_matrix)[0], fci_energy)

def test_change_number_of_stabilizers():
    QT.symmetry_generators = IndependentOp.from_list(['ZIZI', 'IZIZ'])
    with pytest.warns():
        H2_taper = QT.taper_it(ref_state=hf_state)
    assert H2_taper.n_qubits == 2
    assert np.isclose(exact_gs_energy(H2_taper.to_sparse_matrix)[0], fci_energy)

def test_reference_state_projection():
    H2_taper = QT.taper_it(ref_state=hf_state)
    hf_taper = QT.project_state(hf_state)
    ccsd_taper = QT.project_state(ccsd_state)
    assert np.isclose(hf_state.dagger * H2_op    * hf_state,   hf_energy)
    assert np.isclose(hf_state.dagger * H2_op    * ccsd_state, ccsd_energy)
    assert np.isclose(hf_taper.dagger * H2_taper * hf_taper,   hf_energy)
    assert np.isclose(hf_taper.dagger * H2_taper * ccsd_taper, ccsd_energy)