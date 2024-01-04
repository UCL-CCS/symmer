from symmer.operators import AntiCommutingOp, PauliwordOp
import pytest
import numpy as np


anti_commuting_real = {
                        'ZIII': (0.8747970716927321+0j),
                        'XZII': (-1.0644565743109524+0j),
                        'YIII': (-0.8228629386183656+0j),
                        'XXZI': (0.055300207495717776+0j),
                        'XYII': (0.7954579805096648+0j),
                        'XXXZ': (-0.18153261708813911+0j),
                        'XXYI': (-0.3922409211719307+0j),
                        'XXXX': (-0.21221866688241092+0j),
                        'XXXY': (-1.307383058078484+0j)
                        }

anti_commuting_complex = {
                        'ZIII': (-1.463090893167244-1.0373683388860946j),
                        'XZII': (-0.6236959970817084+2.3463922466151983j),
                        'YIII': (-1.129271964294082+0.006613401026225518j),
                        'XXZI': (0.5943195511144899-0.5130626941203098j),
                        'XYII': (1.0739015702295176-0.7346971935019978j),
                        'XXXZ': (-0.37567421392003353-0.40031215375799156j),
                        'XXYI': (-0.6724574864687586+0.4742746791096707j),
                        'XXXX': (0.07393496974124038+0.755825537793846j),
                        'XXXY': (0.7526481862263222-1.112929874028072j)
                        }


def test_init_not_anticommuting():
    """
    check assert error thrown if input is not anticommuting
    """
    with pytest.raises(AssertionError):
        AntiCommutingOp.from_dictionary({'ZZZ':1,
                                         'ZIZ':1,
                                         'ZZI':-1,
                                         'III':1})


def test_init_commuting():
    """
    check assert error thrown if input is not anticommuting
    """
    AcOp = AntiCommutingOp.from_dictionary({'ZZZ': 1,
                                     'XXX': 1,
                                     'YYY': -1})
    P = PauliwordOp.from_list(['ZZZ', 'XXX', 'YYY'],
                               [1, 1, -1])
    assert np.allclose(AcOp.to_sparse_matrix.toarray(),
                       P.to_sparse_matrix.toarray())


def test_single_op():
    P = AntiCommutingOp.from_dictionary({'XX':2})
    assert P.n_terms==1


def test_unitary_partitioning_no_s_index_seq_rot():
    AcOp_real = AntiCommutingOp.from_dictionary(anti_commuting_real)
    Ps, rotations, gamma_l, AC_op = AcOp_real.unitary_partitioning(s_index=None, up_method='seq_rot')

    assert Ps.n_terms==1, 'can only rotate onto single Pauli operator'
    assert np.isclose(gamma_l, np.linalg.norm(list(anti_commuting_real.values()))), 'normalization wrong'
    assert len(rotations) == AcOp_real.n_terms-1, 'seq of rotation number incorrect'

    P_red = AcOp_real.perform_rotations(rotations)
    assert Ps.multiply_by_constant(gamma_l) == P_red

    R_seq_rot_Op = PauliwordOp.from_dictionary({'I'*AcOp_real.n_qubits : 1})
    for X_sk, theta_sk in rotations[::-1]:
        assert isinstance(X_sk, PauliwordOp), 'rotation operator not a PauliwordOp'
        assert isinstance(theta_sk.real, float)
        assert theta_sk.imag == 0, 'rotation cannot have complex component'
        assert X_sk.n_terms == 1, f'rotation generated  by single pauli operator only! Not {X_sk.n_terms}'

        # Let R(t) = e^{i t/2 Q} = cos(t/2)*I + i*sin(t/2)*Q
        R = (PauliwordOp.from_dictionary({'I'*AcOp_real.n_qubits : np.cos(theta_sk/2)}) \
                                         + X_sk.multiply_by_constant(1j*np.sin(theta_sk/2))
             )
        R_seq_rot_Op *= R

    R_AC_op_Rdag = R_seq_rot_Op * AcOp_real * R_seq_rot_Op.dagger
    assert R_AC_op_Rdag == Ps.multiply_by_constant(gamma_l)
    assert R_seq_rot_Op * R_seq_rot_Op.dagger == PauliwordOp.from_dictionary({'I' * AcOp_real.n_qubits: 1}), 'R not unitary'


def test_unitary_partitioning_no_s_index_LCU():
    AcOp_real = AntiCommutingOp.from_dictionary(anti_commuting_real)
    Ps, rotations, gamma_l, AC_op = AcOp_real.unitary_partitioning(s_index=None, up_method='LCU')

    assert Ps.n_terms==1, 'can only rotate onto single Pauli operator'
    assert np.isclose(gamma_l, np.linalg.norm(list(anti_commuting_real.values()))), 'normalization wrong'

    R_AC_op_Rdag = AcOp_real.perform_rotations(rotations)
    assert R_AC_op_Rdag == Ps.multiply_by_constant(gamma_l)

def test_unitary_partitioning_s_index_seq_rot():
    AcOp_real = AntiCommutingOp.from_dictionary(anti_commuting_real)

    for s_ind in range(AcOp_real.n_terms):
        Ps, rotations, gamma_l, AC_op = AcOp_real.unitary_partitioning(s_index=s_ind,
                                                                       up_method='seq_rot')

        assert Ps.n_terms==1, 'can only rotate onto single Pauli operator'
        assert np.isclose(gamma_l, np.linalg.norm(list(anti_commuting_real.values()))), 'normalization wrong'
        assert len(rotations) == AcOp_real.n_terms-1, 'seq of rotation number incorrect'

        P_red = AcOp_real.perform_rotations(rotations)
        assert Ps.multiply_by_constant(gamma_l) == P_red

        R_seq_rot_Op = PauliwordOp.from_dictionary({'I'*AcOp_real.n_qubits : 1})
        for X_sk, theta_sk in rotations[::-1]:
            assert isinstance(X_sk, PauliwordOp), 'rotation operator not a PauliwordOp'
            assert isinstance(theta_sk.real, float)
            assert theta_sk.imag == 0, 'rotation cannot have complex component'
            assert X_sk.n_terms == 1, f'rotation generated  by single pauli operator only! Not {X_sk.n_terms}'

            # Let R(t) = e^{i t/2 Q} = cos(t/2)*I + i*sin(t/2)*Q
            R = (PauliwordOp.from_dictionary({'I'*AcOp_real.n_qubits : np.cos(theta_sk/2)}) \
                                             + X_sk.multiply_by_constant(1j*np.sin(theta_sk/2))
                 )
            R_seq_rot_Op *= R

        R_AC_op_Rdag = R_seq_rot_Op * AcOp_real * R_seq_rot_Op.dagger
        assert R_AC_op_Rdag == Ps.multiply_by_constant(gamma_l)
        assert R_seq_rot_Op * R_seq_rot_Op.dagger == PauliwordOp.from_dictionary({'I' * AcOp_real.n_qubits: 1}), 'R not unitary'


def test_unitary_partitioning_s_index_LCU():
    AcOp_real = AntiCommutingOp.from_dictionary(anti_commuting_real)

    for s_ind in range(AcOp_real.n_terms):
        Ps, rotations, gamma_l, AC_op = AcOp_real.unitary_partitioning(s_index=s_ind,
                                                                       up_method='LCU')

        assert Ps.n_terms==1, 'can only rotate onto single Pauli operator'
        assert np.isclose(gamma_l, np.linalg.norm(list(anti_commuting_real.values()))), 'normalization wrong'

        R_AC_op_Rdag = AcOp_real.perform_rotations(rotations)
        assert R_AC_op_Rdag == Ps.multiply_by_constant(gamma_l)

def test_unitary_partitioning_seq_rot_complex():
    AcOp_comp = AntiCommutingOp.from_dictionary(anti_commuting_complex)
    with pytest.raises(AssertionError):
        AcOp_comp.unitary_partitioning(s_index=None,
                                       up_method='seq_rot')


def test_unitary_partitioning_LCU_complex():
    AcOp_comp = AntiCommutingOp.from_dictionary(anti_commuting_complex)
    with pytest.raises(AssertionError):
        AcOp_comp.unitary_partitioning(s_index=None,
                                       up_method='LCU')


def test_generate_LCU_operator():
    AcOp_real = AntiCommutingOp.from_dictionary(anti_commuting_real)
    normalization = np.linalg.norm(list(anti_commuting_real.values()))

    AcOp_normed = AcOp_real.multiply_by_constant(1/normalization)

    Ps_LCU = AcOp_real.generate_LCU_operator(AcOp_normed)
    R_LCU = AcOp_real.R_LCU

    R_AC_op_Rdag = R_LCU * AcOp_normed * R_LCU.dagger
    assert R_AC_op_Rdag.n_terms == 1
    assert R_AC_op_Rdag == Ps_LCU


def test_recursive_seq_rotations():
    AcOp_real = AntiCommutingOp.from_dictionary(anti_commuting_real)
    normalization = np.linalg.norm(list(anti_commuting_real.values()))

    AcOp_normed = AcOp_real.multiply_by_constant(1/normalization)

    Ps_LCU = AcOp_real._recursive_seq_rotations(AcOp_normed)
    R_seq_rot = AcOp_real.X_sk_rotations

    R_AC_op_Rdag = AcOp_normed.perform_rotations(R_seq_rot)
    assert R_AC_op_Rdag.n_terms == 1
    assert R_AC_op_Rdag == Ps_LCU


def test_ac_set_with_zero_ceoffs():
    AcOp_real = AntiCommutingOp.from_list(['YY', 'XI', 'ZI', 'YX'], [0, 0, 1, 0.2])

    Ps_seq_rot, rotations_seq_rot, gamma_l, AC_normed = AcOp_real.unitary_partitioning(s_index=1,
                                                                        up_method='seq_rot')
    seq_rot_output = AC_normed.perform_rotations(rotations_seq_rot)
    assert seq_rot_output.n_terms==1
    assert np.isclose(seq_rot_output.coeff_vec[0], 1)
    assert Ps_seq_rot == seq_rot_output

    Ps_LCU, rotations_LCU, gamma_l, AC_normed = AcOp_real.unitary_partitioning(s_index=1,
                                                                        up_method='LCU')

    LCU_output = AC_normed.perform_rotations(rotations_LCU).cleanup()
    assert LCU_output.n_terms==1
    assert np.isclose(LCU_output.coeff_vec[0], 1)
    assert Ps_LCU == LCU_output

def test_ac_set_with_negative_and_zero_ceoffs():
    AcOp_real = AntiCommutingOp.from_list(['YY', 'XI', 'ZI'], [-1, 0, 0])

    Ps_seq_rot, rotations_seq_rot, gamma_l, AC_normed = AcOp_real.unitary_partitioning(s_index=0,
                                                                        up_method='seq_rot')
    seq_rot_output = AC_normed.perform_rotations(rotations_seq_rot)
    assert seq_rot_output.n_terms==1
    assert np.isclose(seq_rot_output.coeff_vec[0], -1)
    assert Ps_seq_rot == seq_rot_output

    Ps_LCU, rotations_LCU, gamma_l, AC_normed = AcOp_real.unitary_partitioning(s_index=0,
                                                                        up_method='LCU')

    LCU_output = AC_normed.perform_rotations(rotations_LCU)
    assert LCU_output.n_terms==1
    assert np.isclose(LCU_output.coeff_vec[0], -1)
    assert Ps_LCU == LCU_output