import warnings

import pytest
import numpy as np
from symmer.operators import PauliwordOp, NoncontextualOp, QuantumState
from symmer.operators.noncontextual_op import NoncontextualSolver
from symmer.utils import exact_gs_energy

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


# def test_diag_noncontextual_op():
#     """ Return the diagonal terms of the PauliwordOp - this is the simplest noncontextual operator
#     """
#     H = PauliwordOp.from_dictionary(H_con_dict)
#     H_noncon_diag = NoncontextualOp._diag_noncontextual_op(H)
#     assert ~np.any(H_noncon_diag.X_block, axis=1)

def test_from_hamiltonian_diag():
    """ check noncontextual operator via diagonal approach (Z ops only!)
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_diag = NoncontextualOp.from_hamiltonian(H,
            strategy='diag',
            basis= None,
            DFS_runtime=10)
    assert np.all(np.sum(H_noncon_diag.X_block, axis=1)==0), 'some non Z operators present'
    assert H_noncon_diag.is_noncontextual


def test_from_hamiltonian_DFS_magnitude():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='DFS_magnitude',
            basis= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_DFS_largest():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='DFS_largest',
            basis= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_SingleSweep_magnitude():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='SingleSweep_magnitude',
            basis= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_SingleSweep_random():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='SingleSweep_random',
            basis= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_SingleSweep_CurrentOrder():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='SingleSweep_CurrentOrder',
            basis= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_basis():
    """
    check noncontextual op via a defined basis
    """
    ## COMMUTING BASIS
    basis1 = PauliwordOp.from_dictionary({'IZ': (1+0j),
                                         'ZI': (1+0j)})
    H = PauliwordOp.from_dictionary(H_con_dict)

    H_noncon_basis1 = NoncontextualOp.from_hamiltonian(H,
            strategy='basis',
            basis= basis1,
            DFS_runtime=10)

    assert H_noncon_basis1.is_noncontextual

    ## NON-COMMUTING BASIS
    basis2 = PauliwordOp.from_dictionary({'IZ': (1 + 0j),
                                          'ZI': (1 + 0j),
                                          'XI': (1 + 0j)})

    H_noncon_basis2 = NoncontextualOp.from_hamiltonian(H,
            strategy='basis',
            basis= basis2,
            DFS_runtime=10)

    assert H_noncon_basis2.is_noncontextual

    assert H_noncon_basis2.n_terms >= H_noncon_basis1.n_terms



####################################
# Testing noncontextual optimizers #
####################################

def test_noncontextual_objective_function():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    nu = [-1, -1,  1]
    r_vec = [0.22365807, 0.97466767]
    e_noncon = H_noncon_op.noncontextual_objective_function(nu, r_vec)
    assert np.isclose(e_noncon, noncon_problem['E'])


def test_convex_problem():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    nu = [-1, -1,  1]
    optimized_energy, r_optimal = H_noncon_op._convex_problem(nu)
    assert np.isclose(optimized_energy, noncon_problem['E'])


def test_solve_brute_force_discrete_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='brute_force',
                   ref_state=None,
                   discrete_optimization_order=None,
                   num_anneals=None)
    assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


def test_solve_binary_relaxation_no_ref():
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    H_noncon_op.solve(strategy='binary_relaxation',
                      ref_state=None,
                      discrete_optimization_order=None,
                      num_anneals=None)
    assert noncon_problem['E']<=H_noncon_op.energy
    if not np.isclose(H_noncon_op.energy, noncon_problem['E']):
        warnings.warn('binary relaxation method not finding correct energy')
    else:
        assert np.isclose(H_noncon_op.energy, noncon_problem['E'])


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


def test_get_qaoa_qubo_no_reference():
    """
    checks qaoa qubo with no reference state (compare exact brute force approach)
    """
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    QAOA_dict = H_noncon_op.get_qaoa(ref_state=None, type='qubo')

    for key in QAOA_dict.keys():
        exp = QAOA_dict[key]
        qaoa_H = exp['H']

        e_qaoq, gs_qaoa = exact_gs_energy(qaoa_H.to_sparse_matrix)

        # brute force check ground state for a fixed r_vec!!!
        NC_solver = NoncontextualSolver(H_noncon_op)
        NC_solver.metheod = 'brute_force'
        NC_solver.x = 'Q'

        energy, nu_vec, _ = NC_solver._energy_xUSO(exp['r_vec'])

        assert np.isclose(e_qaoq, energy)


def test_get_qaoa_qubo_with_full_reference():
    """
    checks qaoa qubo with full reference state (compare exact brute force approach)
    """
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)
    reference = noncon_problem['reference_state']

    H_noncon_op.symmetry_generators.update_sector(reference)
    ev_assignment = H_noncon_op.symmetry_generators.coeff_vec
    fixed_ev_mask = ev_assignment != 0
    fixed_eigvals = (ev_assignment[fixed_ev_mask]).astype(int)

    QAOA_dict = H_noncon_op.get_qaoa(ref_state=reference, type='qubo')

    for key in QAOA_dict.keys():
        exp = QAOA_dict[key]
        qaoa_H = exp['H']

        e_qaoq, gs_qaoa = exact_gs_energy(qaoa_H.to_sparse_matrix)

        # brute force check ground state for a fixed r_vec!!!
        NC_solver = NoncontextualSolver(H_noncon_op,
                                        fixed_ev_mask,
                                        fixed_eigvals)
        NC_solver.metheod = 'brute_force'
        NC_solver.x = 'Q'

        energy, nu_vec, _ = NC_solver._energy_xUSO(exp['r_vec'])

        assert np.isclose(e_qaoq, energy)


def test_get_qaoa_qubo_with_partial_reference():
    """
    checks qaoa qubo with partial reference state (compare exact brute force approach)
    """
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)
    reference = noncon_problem['partial_reference_state']

    H_noncon_op.symmetry_generators.update_sector(reference)
    ev_assignment = H_noncon_op.symmetry_generators.coeff_vec
    fixed_ev_mask = ev_assignment != 0
    fixed_eigvals = (ev_assignment[fixed_ev_mask]).astype(int)

    QAOA_dict = H_noncon_op.get_qaoa(ref_state=reference, type='qubo')

    for key in QAOA_dict.keys():
        exp = QAOA_dict[key]
        qaoa_H = exp['H']

        e_qaoq, gs_qaoa = exact_gs_energy(qaoa_H.to_sparse_matrix)

        # brute force check ground state for a fixed r_vec!!!
        NC_solver = NoncontextualSolver(H_noncon_op, fixed_ev_mask, fixed_eigvals)
        NC_solver.metheod = 'brute_force'
        NC_solver.x = 'Q'

        energy, nu_vec, _ = NC_solver._energy_xUSO(exp['r_vec'])

        assert np.isclose(e_qaoq, energy)


def test_get_qaoa_pubo_no_reference():
    """
    checks qaoa pubo with no reference state (compare exact brute force approach)
    """
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)

    QAOA_dict = H_noncon_op.get_qaoa(ref_state=None, type='pubo')

    for key in QAOA_dict.keys():
        exp = QAOA_dict[key]
        qaoa_H = exp['H']

        e_qaoq, gs_qaoa = exact_gs_energy(qaoa_H.to_sparse_matrix)

        # brute force check ground state for a fixed r_vec!!!
        NC_solver = NoncontextualSolver(H_noncon_op)
        NC_solver.metheod = 'brute_force'
        NC_solver.x = 'Q'

        energy, nu_vec, _ = NC_solver._energy_xUSO(exp['r_vec'])

        assert np.isclose(e_qaoq, energy)


def test_get_qaoa_pubo_with_full_reference():
    """
    checks qaoa pubo with full reference state (compare exact brute force approach)
    """
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)
    reference = noncon_problem['reference_state']

    H_noncon_op.symmetry_generators.update_sector(reference)
    ev_assignment = H_noncon_op.symmetry_generators.coeff_vec
    fixed_ev_mask = ev_assignment != 0
    fixed_eigvals = (ev_assignment[fixed_ev_mask]).astype(int)

    QAOA_dict = H_noncon_op.get_qaoa(ref_state=reference, type='pubo')

    for key in QAOA_dict.keys():
        exp = QAOA_dict[key]
        qaoa_H = exp['H']

        e_qaoq, gs_qaoa = exact_gs_energy(qaoa_H.to_sparse_matrix)

        # brute force check ground state for a fixed r_vec!!!
        NC_solver = NoncontextualSolver(H_noncon_op,
                                        fixed_ev_mask,
                                        fixed_eigvals)
        NC_solver.metheod = 'brute_force'
        NC_solver.x = 'Q'

        energy, nu_vec, _ = NC_solver._energy_xUSO(exp['r_vec'])

        assert np.isclose(e_qaoq, energy)


def test_get_qaoa_pubo_with_partial_reference():
    """
    checks qaoa pubo with partial reference state (compare exact brute force approach)
    """
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)
    reference = noncon_problem['partial_reference_state']

    H_noncon_op.symmetry_generators.update_sector(reference)
    ev_assignment = H_noncon_op.symmetry_generators.coeff_vec
    fixed_ev_mask = ev_assignment != 0
    fixed_eigvals = (ev_assignment[fixed_ev_mask]).astype(int)

    QAOA_dict = H_noncon_op.get_qaoa(ref_state=reference, type='pubo')

    for key in QAOA_dict.keys():
        exp = QAOA_dict[key]
        qaoa_H = exp['H']

        e_qaoq, gs_qaoa = exact_gs_energy(qaoa_H.to_sparse_matrix)

        # brute force check ground state for a fixed r_vec!!!
        NC_solver = NoncontextualSolver(H_noncon_op, fixed_ev_mask, fixed_eigvals)
        NC_solver.metheod = 'brute_force'
        NC_solver.x = 'Q'

        energy, nu_vec, _ = NC_solver._energy_xUSO(exp['r_vec'])

        assert np.isclose(e_qaoq, energy)


def test_get_qaoa_error():
    """
    checks for type if type is not pubo or qubo
    """
    H_noncon = PauliwordOp.from_dictionary(noncon_problem['H_dict'])
    H_noncon_op = NoncontextualOp.from_PauliwordOp(H_noncon)
    with pytest.raises(AssertionError):
        H_noncon_op.get_qaoa(type='INCORRECT_STRING')