import pytest
from symmer.operators import IndependentOp, PauliwordOp
import numpy as np

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

def test_target_sqp_invalid_value_error():
    with pytest.raises(ValueError):
        IndependentOp([[0,1]], [1], target_sqp='x')

def test_from_list():
    op1 = IndependentOp.from_list(['X', 'Z'])
    op2 = IndependentOp(
        [[0,1],[1,0]], [1,1]
    )
    assert op1 == op2

def test_from_dictionary():
    op1 = IndependentOp.from_dictionary({'X':1, 'Z':1})
    op2 = IndependentOp(
        [[0,1],[1,0]], [1,1]
    )
    assert op1 == op2

# def test_no_symmetry_generator_error():
#     op = PauliwordOp.from_list(['X', 'Y','Z'])
#     with pytest.raises(RuntimeError):
#         IndependentOp.symmetry_generators(op)

def test_no_symmetry_generators():
    op = PauliwordOp.from_list(['X', 'Y','Z'])
    ind_op = IndependentOp.symmetry_generators(op)
    assert ind_op.n_terms == 0


def test_commuting_overide_symmetry_generators():
    op = PauliwordOp.from_list(
        ['IZZ', 'ZZI', 'IXX', 'XXI', 'IYY', 'YYI']
    )
    gen_without_override = IndependentOp.symmetry_generators(op, commuting_override=False)
    gen_with_override = IndependentOp.symmetry_generators(op, commuting_override=True)
    assert gen_with_override != gen_without_override
    assert gen_without_override in [
        IndependentOp.from_list(['XXX']), IndependentOp.from_list(['ZZZ'])
    ]
    assert gen_with_override == IndependentOp.from_list(['XXX', 'ZZZ'])

def test_clique_cover_large_anticommuting_generating_set():
    op = PauliwordOp.from_list(['Z'*20])
    with pytest.warns():
        IndependentOp.symmetry_generators(op)

def test_dependent_input():
    with pytest.raises(ValueError):
        IndependentOp.from_list(['X', 'Y', 'Z'])

def test_rotations_onto_sqp_Z():
    op = PauliwordOp.from_list(['Z'*20])
    G = IndependentOp.symmetry_generators(op)
    G.target_sqp = 'Z'
    rotated = G.rotate_onto_single_qubit_paulis()
    assert (
        np.all(np.sum(rotated.Z_block, axis=1) <= 1) and
        np.all(~rotated.X_block)
    )

def test_rotations_onto_sqp_X():
    op = PauliwordOp.from_list(['Z'*20])
    G = IndependentOp.symmetry_generators(op)
    G.target_sqp = 'X'
    rotated = G.rotate_onto_single_qubit_paulis()
    assert (
        np.all(np.sum(rotated.X_block, axis=1) <= 1) and
        np.all(~rotated.Z_block)
    )

def test_symmetry_generators_H2():
    G1 = IndependentOp.symmetry_generators(H2_op)
    G2 = IndependentOp.from_list(['ZIZI', 'IZIZ', 'IIZZ'])

    assert (
        np.all(G1.basis_reconstruction(G2)[1]) and
        np.all(G2.basis_reconstruction(G1)[1])
    )

def test_value_assignment():
    G = IndependentOp.symmetry_generators(H2_op)
    G.update_sector(ref_state=ref_state)
    assert np.all(G.coeff_vec == (-1) ** np.sum(np.bitwise_and(G.Z_block,ref_state), axis=1))
    
def test_indexing():
    G = IndependentOp.from_list(['IZ', 'ZI', 'XX'])
    assert G[0] == IndependentOp.from_list(['IZ'])
    assert G[1] == IndependentOp.from_list(['ZI'])
    assert G[2] == IndependentOp.from_list(['XX'])

def test_invalid_coefficient():
    with pytest.raises(ValueError):
        IndependentOp.from_dictionary({'X':1, 'Z':2})