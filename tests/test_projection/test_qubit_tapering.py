import pytest
import numpy as np
from symmer import PauliwordOp, QubitTapering
from symmer.symplectic import IndependentOp
from symmer.utils import exact_gs_energy

H2_op = PauliwordOp.from_dictionary(
    {
    'IIII': (-0.09706626816762845+0j),
    'IIIZ': (-0.22343153690813597+0j),
    'IIZI': (-0.22343153690813597+0j),
    'IIZZ': (0.17441287612261608+0j),
    'IZII': (0.17141282644776884+0j),
    'IZIZ': (0.12062523483390426+0j),
    'IZZI': (0.16592785033770355+0j),
    'ZIII': (0.17141282644776884+0j),
    'ZIIZ': (0.16592785033770355+0j),
    'ZIZI': (0.12062523483390426+0j),
    'ZZII': (0.16868898170361213+0j),
    'XXYY': (-0.0453026155037993+0j),
    'XYYX': (0.0453026155037993+0j),
    'YXXY': (0.0453026155037993+0j),
    'YYXX': (-0.0453026155037993+0j)
    }
)
energy = -1.1372838344885023
ref_state = np.array([1, 1, 0, 0])
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
    H2_taper = QT.taper_it(ref_state=ref_state)
    assert H2_taper.n_qubits == 1
    assert np.isclose(exact_gs_energy(H2_taper.to_sparse_matrix)[0], energy)

def test_change_number_of_stabilizers():
    QT.symmetry_generators = IndependentOp.from_list(['ZIZI', 'IZIZ'])
    with pytest.warns():
        H2_taper = QT.taper_it(ref_state=ref_state)
    assert H2_taper.n_qubits == 2
    assert np.isclose(exact_gs_energy(H2_taper.to_sparse_matrix)[0], energy)