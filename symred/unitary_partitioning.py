from symred.symplectic_form import PauliwordOp, symplectic_to_string
import numpy as np
from typing import Dict, List, Tuple, Union

class AntiCommutingOp(PauliwordOp):


    def __init__(self,
                 AC_operator: Union[List[str], Dict[str, float], np.array],
                 coeff_list: Union[List[complex], np.array] = None):
        super().__init__(AC_operator, coeff_list)

        # check all operators anticommute
        anti_comm_check = self.adjacency_matrix.astype(int) - np.eye(self.adjacency_matrix.shape[0])
        assert(np.einsum('ij->', anti_comm_check) == 0), 'operator needs to be made of anti-commuting Pauli operators'

        # normalization factor
        self.gamma_l = np.linalg.norm(self.coeff_vec)
        # normalize coefficients
        self.coeff_vec = self.coeff_vec/self.gamma_l

    def lexicographical_sort(self):
        """
        sort object into lexicographical order

        Returns:

        """
        # convert sym form to list of ints
        int_list = self.symp_matrix @ (1 << np.arange(self.symp_matrix.shape[1])[::-1])
        lex_ordered_indices = np.argsort(int_list)
        return lex_ordered_indices


    def get_lowest_dense_index(self):

        # np.logical_or(X_block, Z_block)
        pos_terms_occur = np.logical_or(self.symp_matrix[:, :self.n_qubits],self.symp_matrix[:, self.n_qubits:])

        int_list = pos_terms_occur @ (1 << np.arange(pos_terms_occur.shape[1])[::-1])
        s_index = np.argmin(int_list)
        return s_index


    def gen_sequence_of_rotations(self, s_index=None):
        """

        Args:
            s_index:

        Returns:

        """
        X_sk_theta_sk_list=[]
        if self.n_terms == 1:
            return None

        if s_index is None:
            s_index = self.get_lowest_dense_index()

        # take β_s P_s then remove from symp mat and coeff vec
        P_s = PauliwordOp(self.symp_matrix[s_index], [1])
        β_s = self.coeff_vec[s_index]
        symp_matrix_no_Ps =  np.delete(self.symp_matrix, s_index, axis=0)
        coeff_vec_no_βs =  np.delete(self.coeff_vec, s_index, axis=0)

        theta_sk = np.arctan(coeff_vec_no_βs[0] / β_s)
        if β_s.real < 0:
            theta_sk = theta_sk + np.pi

        assert(np.isclose((coeff_vec_no_βs[0] * np.cos(theta_sk) - β_s * np.sin(theta_sk)), 0)), 'term not zeroing out'

        # X_sk = 1j * Ps @ Pk
        X_sk = P_s * PauliwordOp(symp_matrix_no_Ps[0], [1j])

        if X_sk.coeff_vec[0].real<0:
            X_sk_theta_sk_list.append((X_sk.multiply_by_constant(-1), -1*theta_sk))
        else:
            X_sk_theta_sk_list.append((X_sk, theta_sk))

        # β_s_new = np.sqrt(coeff_vec_no_βs[0] ** 2 + β_s ** 2)
        β_s_new = np.linalg.norm([coeff_vec_no_βs[0],  β_s])
        for ind, Pk in enumerate(symp_matrix_no_Ps[1:]):
            β_k = coeff_vec_no_βs[1:][ind]

            # X_sk = 1j * Ps @ Pk
            theta_sk = np.arctan(β_k / β_s_new)
            assert (np.isclose((β_k * np.cos(theta_sk) - β_s_new * np.sin(theta_sk)), 0)), 'term not zeroing out'

            # X_sk = 1j * Ps @ Pk
            X_sk = P_s * PauliwordOp(Pk, [1j])

            if X_sk.coeff_vec[0].real < 0:
                X_sk_theta_sk_list.append((X_sk.multiply_by_constant(-1), -1 * theta_sk))
            else:
                X_sk_theta_sk_list.append((X_sk, theta_sk))

            β_s_new = np.linalg.norm([β_k, β_s])

        return X_sk_theta_sk_list

    def rotate_by_single_Rsk(self,
                                op_to_rotate: "PauliwordOp",
                                X_sk: "PauliwordOp",
                                theta_sk: float = None
                                ) -> "PauliwordOp":
        """
        def
        """

        commute_vec = op_to_rotate.commutes_termwise(X_sk).flatten()

        commute_symp = op_to_rotate.symp_matrix[commute_vec]
        commute_coeff = op_to_rotate.coeff_vec[commute_vec]
        # ~commute_vec == not commutes, this indexes the anticommuting terms
        anticommute_symp = op_to_rotate.symp_matrix[~commute_vec]
        anticommute_coeff = op_to_rotate.coeff_vec[~commute_vec]

        commute_self = PauliwordOp(commute_symp, commute_coeff)
        anticom_self = PauliwordOp(anticommute_symp, anticommute_coeff)

        anticom_part = (anticom_self.multiply_by_constant(np.cos(theta_sk)) +
                        (anticom_self * X_sk).multiply_by_constant(-1j * np.sin(theta_sk)))

        return commute_self + anticom_part

    def Apply_SeqRot(self, list_rotations):
        AC_op_rotated = PauliwordOp(self.symp_matrix, self.coeff_vec)
        for X_sk, theta_sk in list_rotations:
            # R_sk = PauliwordOp(['I'*X_sk.n_qubits],[1]).multiply_by_constant(np.cos(theta_sk/2)) + X_sk.multiply_by_constant(np.cos(theta_sk/2))
            # R_sk_dag = R_sk.conjugate
            # AC_op_rotated = R_sk * AC_op_rotated * R_sk_dag
            # AC_op_rotated.cleanup_zeros()
            # if X_sk.co
            AC_op_rotated = AC_op_rotated._rotate_by_single_Pword(X_sk, theta_sk).cleanup_zeros()
            print(AC_op_rotated)
        return AC_op_rotated


def unitary_partitioning_rotations(AC_op: PauliwordOp) -> List[Tuple[str, float]]:
    """ Perform unitary partitioning as per https://doi.org/10.1103/PhysRevA.101.062322 (Section A)
    Note unitary paritioning only works when the terms are mutually anticommuting
    """
    # check the terms are mutually anticommuting to avoid an infinite loop:
    assert (
        np.all(
            np.array(AC_op.adjacency_matrix, dtype=int)
            == np.eye(AC_op.n_terms, AC_op.n_terms)
        )
    ), 'Operator terms are not mutually anticommuting'

    rotations = []

    def _recursive_unitary_partitioning(AC_op: PauliwordOp) -> None:
        """ Always retains the first term of the operator, deletes the second
        term at each level of recursion and reads out the necessary rotations
        """
        if AC_op.n_terms == 1:
            return None
        else:
            op_for_rotation = AC_op.copy()
            A0, A1 = op_for_rotation[0], op_for_rotation[1]
            angle = np.arctan(A1.coeff_vec / A0.coeff_vec)
            # set coefficients to 1 since we only want to track sign flip from here
            A0.coeff_vec, A1.coeff_vec = [1], [1]
            pauli_rot = (A0 * A1).multiply_by_constant(-1j)
            angle *= pauli_rot.coeff_vec
            # perform the rotation, thus deleting a single term from the input operator
            AC_op_rotated = op_for_rotation._rotate_by_single_Pword(pauli_rot, angle).cleanup_zeros()

            # append the rotation to list
            rotations.append((symplectic_to_string(pauli_rot.symp_matrix[0]), angle.real[0]))

            return _recursive_unitary_partitioning(AC_op_rotated)

    _recursive_unitary_partitioning(AC_op)

    return rotations

