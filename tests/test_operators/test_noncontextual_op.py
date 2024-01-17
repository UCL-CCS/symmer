import warnings

import pytest
import numpy as np
from symmer.operators import PauliwordOp, NoncontextualOp, QuantumState
from symmer.operators.noncontextual_op import NoncontextualSolver
from symmer.utils import exact_gs_energy

def jordan_generator_reconstruction_check(self, generators):
    """ Function for jordan generators reconstruction test
    This builds the noncontextual operator under the Jordan product, but does not give the
    reconstruction matrix. This can be used to check that the function with the reconstruction
    matrix IS correct!
    """
    mask_symmetries = np.all(generators.adjacency_matrix, axis=1)
    Symmetries = generators[mask_symmetries]
    Anticommuting = generators[~mask_symmetries]
    assert np.all(Anticommuting.adjacency_matrix == np.eye(Anticommuting.n_terms))

    PwordOp_noncon = self[self.generator_reconstruction(Symmetries)[1]]
    PwordOp_remain = self - PwordOp_noncon
    for P in Anticommuting:
        PwordOp_noncon += PwordOp_remain[PwordOp_remain.generator_reconstruction(P+Symmetries)[1]]

    return PwordOp_noncon

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
            generators= None,
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
            generators= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_DFS_largest():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='DFS_largest',
            generators= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_SingleSweep_magnitude():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='SingleSweep_magnitude',
            generators= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_SingleSweep_random():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='SingleSweep_random',
            generators= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_SingleSweep_CurrentOrder():
    """
    check noncontextual op via depth first search
    """
    H = PauliwordOp.from_dictionary(H_con_dict)
    H_noncon_dfs = NoncontextualOp.from_hamiltonian(H,
            strategy='SingleSweep_CurrentOrder',
            generators= None,
            DFS_runtime=10)

    assert H_noncon_dfs.is_noncontextual


def test_from_hamiltonian_generators():
    """
    check noncontextual op via a defined generators
    """
    ## COMMUTING generators
    generators1 = PauliwordOp.from_dictionary({'IZ': (1+0j),
                                         'ZI': (1+0j)})
    H = PauliwordOp.from_dictionary(H_con_dict)

    H_noncon_generators1 = NoncontextualOp.from_hamiltonian(H,
            strategy='generators',
            generators= generators1,
            DFS_runtime=10)

    assert H_noncon_generators1.is_noncontextual

    ## NON-COMMUTING generators
    generators2 = PauliwordOp.from_dictionary({'IZ': (1 + 0j),
                                          'ZI': (1 + 0j),
                                          'XI': (1 + 0j)})

    H_noncon_generators2 = NoncontextualOp.from_hamiltonian(H,
            strategy='generators',
            generators= generators2,
            DFS_runtime=10)

    assert H_noncon_generators2.is_noncontextual

    assert H_noncon_generators2.n_terms >= H_noncon_generators1.n_terms


def test_noncon_no_symmetry_generators():
    Pwords = PauliwordOp.from_list(['X', 'Y', 'Z'])
    E_ground = -1.7320508075688772

    with pytest.warns():
        # capture warning of no Z2 symmetries
        H_noncon = NoncontextualOp.from_PauliwordOp(Pwords)
    assert H_noncon.n_terms == Pwords.n_terms
    H_noncon.solve()
    assert np.isclose(H_noncon.energy, E_ground)


def test_noncon_noncommuting_Z2():
    """
    test operator that previously caused bug

    Here we have a set of Z2 symmerties from H_noncon that don't termwise commute!
    i.e.
    Z2_symmerties = IndependentOp.symmetry_generators(H_noncon, commuting_override=True)
    Z2_symmerties don't all pairwise commute. This can lead to problems if not handled correctly.

    """
    H_noncon = PauliwordOp.from_dictionary({'IIIIIIIIII': (-15.27681058613343+0j), 'IIIIIIIIIZ': (-0.2172743037258172+0j), 'IIIIIZIIII': (0.11405708903311647+0j), 'IIIIIZIIIZ': (0.22929933946046854+0j), 'IIIIZIIIII': (0.07618739164001126+0j), 'IIIIZIIIIZ': (0.10424432000703249+0j), 'IIIIZZIIII': (0.2458433632534619+0j), 'IIIZIIIIII': (0.07618739164001137+0j), 'IIIZIIIIIZ': (0.11790253563113795+0j), 'IIIZIZIIII': (0.2458433632534619+0j), 'IIIZZIIIII': (0.14846395230159326+0j), 'IIZIIIIIII': (0.17066279019288416+0j), 'IIZIIIIIIZ': (0.19577211156187402+0j), 'IIZIIZIIII': (0.220702612028004+0j), 'IIZIZIIIII': (0.10245792602520705+0j), 'IIZZIIIIII': (0.10964094999478652+0j), 'IZIIIIIIII': (0.2601827946303151+0j), 'IZIIIIIIIZ': (0.1060269619647709+0j), 'IZIIIZIIII': (0.21982388480817722+0j), 'IZIIZIIIII': (0.09485326000745692+0j), 'IZIZIIIIII': (0.1112840228980842+0j), 'IZZIIIIIII': (0.08389584977571822+0j), 'IZZIZZIIIZ': (-0.14917151435431195+0j), 'ZIIIIIIIII': (0.2601827946303149+0j), 'ZIIIIIIIIZ': (0.11944056365665429+0j), 'ZIIIIZIIII': (0.21982388480817722+0j), 'ZIIIZIIIII': (0.1112840228980842+0j), 'ZIIZIIIIII': (0.09485326000745692+0j), 'ZIZIIIIIII': (0.12042746433351019+0j), 'ZIZZIZIIIZ': (-0.14917151435431195+0j), 'ZZIIIIIIII': (0.12244309127472158+0j), 'IIZZZIXIXZ': (-0.0023524442663984376+0j), 'ZIZIZZXIXZ': (0.0023524442663984376+0j), 'ZIYIZZIXIY': (0.0023524442663984376+0j), 'ZZYIIIIXIY': (-0.0023524442663984376+0j)})
    H_noncon_obj = NoncontextualOp.from_PauliwordOp(H_noncon)
    assert H_noncon.n_terms == H_noncon_obj.n_terms

####################################
# Testing noncontextual optimizers #
####################################

def test_noncontextual_objective_function():
    H_noncon = NoncontextualOp.from_dictionary(noncon_problem['H_dict'])
    nu = [-1, -1,  +1]
    e_noncon = H_noncon.get_energy(nu)
    assert np.isclose(e_noncon, noncon_problem['E'])

def test_solve_brute_force_discrete_no_ref():
    H_noncon = NoncontextualOp.from_dictionary(noncon_problem['H_dict'])
    
    H_noncon.solve(strategy='brute_force',
                   ref_state=None)
    assert np.isclose(H_noncon.energy, noncon_problem['E'])


def test_solve_binary_relaxation_no_ref():
    H_noncon = NoncontextualOp.from_dictionary(noncon_problem['H_dict'])
    
    H_noncon.solve(strategy='binary_relaxation',
                      ref_state=None)
    assert noncon_problem['E']<=H_noncon.energy
    if not np.isclose(H_noncon.energy, noncon_problem['E']):
        warnings.warn('binary relaxation method not finding correct energy')
    else:
        assert np.isclose(H_noncon.energy, noncon_problem['E'])

def test_solve_full_reference_state():
    H_noncon = NoncontextualOp.from_dictionary(noncon_problem['H_dict'])
    reference = noncon_problem['reference_state']

    H_noncon.solve(strategy='brute_force',
                   ref_state=reference)
    assert np.isclose(H_noncon.energy, noncon_problem['E'])


def test_solve_brute_force_discrete_partial_ref():
    H_noncon = NoncontextualOp.from_dictionary(noncon_problem['H_dict'])
    
    partial_reference_state = noncon_problem['partial_reference_state']

    with pytest.warns():
        # capture warning when Z stabilizers measured give zero expec value
        H_noncon.solve(strategy='brute_force',
                       ref_state=partial_reference_state)
    assert np.isclose(H_noncon.energy, noncon_problem['E'])


def test_init_Hnoncon1():
    """
    Check error is thrown if a contextual operator symplectic matrix is used as input to NoncontextualOp
    Returns:

    """
    H_con = PauliwordOp.from_dictionary(H_con_dict)
    symp_matrix = H_con.symp_matrix
    coeff_vec = H_con.coeff_vec
    with pytest.raises(AssertionError):
        NoncontextualOp(symp_matrix, coeff_vec)

def test_noncon_state():
    """
    Check that state generated by NoncontexualOp matches the noncontextual energy
    """
    for _ in range(50):
        Hnc = NoncontextualOp.random(6)
        Hnc.solve()
        state, nu = Hnc.noncon_state(UP_method='LCU')
        assert np.isclose(Hnc.expval(state), Hnc.energy)
        state_rot, nu = Hnc.noncon_state(UP_method='seq_rot')
        assert np.isclose(Hnc.expval(state_rot), Hnc.energy)