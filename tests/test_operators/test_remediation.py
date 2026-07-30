"""Regression tests for the stability / observability remediation work."""
import pytest
import numpy as np
from unittest.mock import patch

from symmer import process
from symmer.operators import PauliwordOp, QuantumState, IndependentOp
from symmer.operators.utils import binary_array_to_int, count1_in_int_bitstring
from symmer.projection import S3Projection


def test_commutes_multi_term_leftover():
    """commutes must inspect all leftover coefficients, not only index 0."""
    P1 = PauliwordOp.from_list(['X', 'Z'], [1, 1])
    P2 = PauliwordOp.from_list(['Y'], [1])
    assert not P1.commutes(P2)

    # Explicitly cover the old buggy pattern: first coeff zero, later nonzero
    fake = PauliwordOp.from_list(['I', 'X'], [0.0, 1.0])
    with patch.object(PauliwordOp, 'commutator', return_value=fake):
        with patch.object(fake, 'cleanup', return_value=fake):
            assert not P1.commutes(P2)


def test_hash_eq_after_near_threshold_cleanup():
    P1 = PauliwordOp.from_list(['XX', 'YY', 'XX'], [1.0, 2.0, 1e-16])
    P2 = PauliwordOp.from_list(['XX', 'YY'], [1.0, 2.0])
    assert P1 == P2
    assert hash(P1) == hash(P2)


def test_imul_mutates_self():
    P = PauliwordOp.from_list(['X', 'Z'], [1, 1])
    alias = P
    Q = PauliwordOp.from_list(['Y'], [1])
    expected = PauliwordOp.from_list(['X', 'Z'], [1, 1]) * Q
    P *= Q
    assert P is alias
    assert P == expected


def test_dagger_no_alias():
    P = PauliwordOp.from_list(['X', 'Y'], [1 + 1j, 2 - 3j])
    Pd = P.dagger
    Pd.coeff_vec[0] = 999
    assert P.coeff_vec[0] != 999
    assert not np.shares_memory(P.symp_matrix, Pd.symp_matrix)


def test_reindex_cycle():
    P = PauliwordOp.from_list(['XYZ'], [1])
    # {0:1, 1:2, 2:0}: fill index i from index map[i]
    Pref = P.reindex({0: 1, 1: 2, 2: 0})
    # X from qubit1 (Y), Y from qubit2 (Z), Z from qubit0 (X) → YZX
    assert Pref == PauliwordOp.from_list(['YZX'], [1])


def test_y_stabilizer_projection_raises():
    op = PauliwordOp.from_list(['ZI', 'IZ', 'XX'], [1, 1, 0.5])
    stabs = IndependentOp.from_list(['YI'])
    stabs.target_sqp = 'Y'
    proj = S3Projection(stabs)
    with pytest.raises(NotImplementedError, match="Y"):
        proj.perform_projection(op, sector=[1])


def test_assign_value_threshold():
    from symmer.operators.independent_op import assign_value

    S = PauliwordOp.from_list(['Z'], [1])
    # Equal superposition → ⟨Z⟩ = 0
    psi = QuantumState([[0], [1]], [1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert assign_value([S], (psi, 0.5))[0] == 0

    # Computational |0⟩ → ⟨Z⟩ = +1
    psi0 = QuantumState([[0]], [1.0])
    assert assign_value([S], (psi0, 0.5))[0] == 1
    # High threshold rejects even a perfect eigenvalue
    assert assign_value([S], (psi0, 1.5))[0] == 0


def test_random_stabilizers_eventually_raises():
    from symmer.projection.contextual_subspace import ContextualSubspace

    # Bypass full CS init — only need ``operator`` for the sampler under test
    CS = object.__new__(ContextualSubspace)
    CS.operator = PauliwordOp.from_list(['ZI', 'IZ', 'XX'], [1, 1, 1])
    with patch.object(
        IndependentOp, 'from_PauliwordOp', side_effect=ValueError('dependent')
    ):
        with pytest.raises(RuntimeError, match='Failed to sample'):
            CS._random_stabilizers(n_qubits=1)


def test_binary_array_to_int_wide():
    # 64-bit width: float path would lose low bits; object path must match Python ints
    width = 64
    row = np.zeros((1, width), dtype=int)
    row[0, -1] = 1  # least significant bit
    row[0, 0] = 1   # most significant bit
    got = binary_array_to_int(row)[0]
    expected = (1 << (width - 1)) + 1
    assert int(got) == expected


def test_count1_in_int_bitstring_wide():
    # Classic 32-bit SWAR fails above 32 bits; bit_count must not
    val = (1 << 40) | (1 << 3) | 1
    assert count1_in_int_bitstring(val) == 3


def test_generator_reconstruction_dependent():
    gens = PauliwordOp.from_list(['ZI', 'ZI'], [1, 1])  # dependent
    op = PauliwordOp.from_list(['ZI'], [1])
    with pytest.raises(AssertionError, match='dependent'):
        op.generator_reconstruction(gens)
    # With override, dependent generators are accepted
    recon, mask = op.generator_reconstruction(gens, override_independence_check=True)
    assert recon.shape[0] == 1


@pytest.mark.filterwarnings(
    "ignore:This process \\(pid=.*\\) is multi-threaded, use of fork\\(\\):DeprecationWarning"
)
def test_expval_mp_matches_single_thread():
    H = PauliwordOp.from_list(['ZI', 'IZ', 'XX'], [1.0, 0.5, 0.25])
    psi = QuantumState([[1, 0]], [1.0])
    previous = process.method
    try:
        process.method = 'single_thread'
        e_single = H.expval(psi)
        process.method = 'mp'
        e_mp = H.expval(psi)
        assert np.isclose(e_single, e_mp)
    finally:
        process.method = previous


def test_from_matrix_incomplete_basis_lossy():
    mat = PauliwordOp.from_list(['X', 'Y', 'Z'], [1, 1, 1]).to_sparse_matrix.toarray()
    incomplete = PauliwordOp.from_list(['X', 'Z'], [1, 1])
    with pytest.warns(UserWarning, match='lossy projection'):
        projected = PauliwordOp.from_matrix(mat, operator_basis=incomplete, strategy='full_basis')
    full = PauliwordOp.from_matrix(mat, strategy='projector')
    assert projected != full
