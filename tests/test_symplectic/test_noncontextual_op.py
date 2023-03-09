import pytest
import numpy as np
from symmer.symplectic import PauliwordOp, NoncontextualOp, QuantumState

noncon_problem = {
    'H_dict':  {'IIII': (-0.09706626816762845+0j),
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
 'YYXX': (-0.0453026155037993+0j)},
    'E': -1.1372838344885023,
    'reference_state': np.array([1, 1, 0, 0]),
    'partial_reference_state': QuantumState(
        np.array([[1, 1, 0, 0],
                  [1, 1, 1, 1]]),
        np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
}

H_con_dict = {'II': (0.104907),
 'IZ': (0.2038683),
 'ZI': (-0.238925),
 'ZZ': (0.2386317),
 'IX': (0.1534837),
 'IY': (0.1503439),
 'ZX': (0.0679678),
 'ZY': (0.2538080),
 'XI': (0.0994848),
 'XZ': (-0.044597),
 'YI': (-0.274103),
 'YZ': (-0.078968),
 'XX': (-0.292164),
 'XY': (-0.183966),
 'YX': (-0.058251),
 'YY': (-0.212114)
 }

def test_init_contextual_input():
    """
    Check error is thrown if a contextual operator symplectic matrix is used as input to NoncontextualOp
    Returns:

    """
    H_con = PauliwordOp.from_dictionary(H_con_dict)
    symp_matrix = H_con.symp_matrix
    coeff_vec = H_con.coeff_vec
    with pytest.raises(AssertionError):
        NoncontextualOp(symp_matrix, coeff_vec)


def test_init_noncontextual_input():
    """
    Check error is thrown if a contextual operator symplectic matrix is used as input to NoncontextualOp
    Returns:

    """
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])

    symp_matrix = H_noncon.symp_matrix
    coeff_vec = H_noncon.coeff_vec
    noncon_op = NoncontextualOp(symp_matrix, coeff_vec)
    assert noncon_op.is_noncontextual


####################################
# Testing noncontextual optimizers #
####################################

def test_solve_brute_force_discrete_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='brute_force',
                   ref_state=None,
                   discrete_optimization_order=None,
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


# def test_solve_binary_relaxation_no_ref():
#     H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
#     H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)
#
#     H_noncon_op.solve(strategy='binary_relaxation',
#                       ref_state=None,
#                       discrete_optimization_order=None,
#                       num_anneals=None)
#     assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_brute_force_PUSO_discrete_first_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='brute_force_PUSO',
                   ref_state=None,
                   discrete_optimization_order='first', # <- HERE
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_brute_force_PUSO_discrete_last_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='brute_force_PUSO',
                   ref_state=None,
                   discrete_optimization_order='last', # <- HERE
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_brute_force_QUSO_discrete_first_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='brute_force_QUSO',
                   ref_state=None,
                   discrete_optimization_order='first', # <- HERE
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_brute_force_QUSO_discrete_last_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='brute_force_QUSO',
                   ref_state=None,
                   discrete_optimization_order='last', # <- HERE
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_annealing_PUSO_discrete_first_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='annealing_PUSO',
                   ref_state=None,
                   discrete_optimization_order='first', # <- HERE
                   num_anneals=1_000)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_annealing_PUSO_discrete_last_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='annealing_PUSO',
                   ref_state=None,
                   discrete_optimization_order='last', # <- HERE
                   num_anneals=1_000)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_annealing_QUSO_discrete_first_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='annealing_QUSO',
                   ref_state=None,
                   discrete_optimization_order='first', # <- HERE
                   num_anneals=1_000)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_annealing_QUSO_discrete_last_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='annealing_QUSO',
                   ref_state=None,
                   discrete_optimization_order='last', # <- HERE
                   num_anneals=1_000)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])

###

def test_solve_full_reference_state():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)
    reference = noncon_problem['reference_state']

    H_noncon_op.solve(strategy='brute_force',
                   ref_state=reference,
                   discrete_optimization_order=None,
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])

###

def test_solve_brute_force_discrete_partial_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    partial_reference_state = noncon_problem['partial_reference_state']

    H_noncon_op.solve(strategy='brute_force',
                   ref_state=partial_reference_state,
                   discrete_optimization_order=None,
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


# def test_solve_binary_relaxation_partial_ref():
#     H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
#     H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)
#
#     partial_reference_state = noncon_problem['partial_reference_state']
#
#     H_noncon_op.solve(strategy='binary_relaxation',
#                       ref_state=partial_reference_state,
#                       discrete_optimization_order=None,
#                       num_anneals=None)
#     assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_brute_force_PUSO_discrete_partial_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    partial_reference_state = noncon_problem['partial_reference_state']

    H_noncon_op.solve(strategy='brute_force_PUSO',
                   ref_state=partial_reference_state,
                   discrete_optimization_order='first', # <- HERE
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_brute_force_QUSO_discrete_partial_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    partial_reference_state = noncon_problem['partial_reference_state']

    H_noncon_op.solve(strategy='brute_force_QUSO',
                   ref_state=partial_reference_state,
                   discrete_optimization_order='first', # <- HERE
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_annealing_PUSO_discrete_partial_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    partial_reference_state = noncon_problem['partial_reference_state']

    H_noncon_op.solve(strategy='annealing_PUSO',
                   ref_state=partial_reference_state,
                   discrete_optimization_order='first', # <- HERE
                   num_anneals=1_000)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_annealing_QUSO_discrete_partial_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    partial_reference_state = noncon_problem['partial_reference_state']

    H_noncon_op.solve(strategy='annealing_QUSO',
                   ref_state=partial_reference_state,
                   discrete_optimization_order='first', # <- HERE
                   num_anneals=1_000)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])

