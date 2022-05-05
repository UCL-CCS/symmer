from symred.symplectic_form import PauliwordOp
import numpy as np
from typing import Dict, List, Union


# class AntiCommutingOp_OLD(PauliwordOp):
#
#     def __init__(self,
#                  AC_operator: Union[List[str], Dict[str, float], np.array],
#                  coeff_list: Union[List[complex], np.array] = None):
#         super().__init__(AC_operator, coeff_list)
#
#         # check all operators anticommute
#         anti_comm_check = self.adjacency_matrix.astype(int) - np.eye(self.adjacency_matrix.shape[0])
#         assert (np.einsum('ij->', anti_comm_check) == 0), 'operator needs to be made of anti-commuting Pauli operators'
#
#         # normalization factor
#         self.gamma_l = np.linalg.norm(self.coeff_vec)
#         # normalize coefficients
#         self.coeff_vec = self.coeff_vec / self.gamma_l
#
#         self.X_sk_rotations = None
#         self.LCU = None
#
#     def lexicographical_sort(self):
#         """
#         sort object into lexicographical order
#
#         Returns:
#
#         """
#         # convert sym form to list of ints
#         int_list = self.symp_matrix @ (1 << np.arange(self.symp_matrix.shape[1], dtype=object)[::-1])
#         lex_ordered_indices = np.argsort(int_list)
#         return lex_ordered_indices
#
#     def get_least_dense_term_index(self):
#
#         # np.logical_or(X_block, Z_block)
#         pos_terms_occur = np.logical_or(self.symp_matrix[:, :self.n_qubits], self.symp_matrix[:, self.n_qubits:])
#
#         int_list = pos_terms_occur @ (1 << np.arange(pos_terms_occur.shape[1])[::-1])
#         s_index = np.argmin(int_list)
#         return s_index
#
#     def gen_seq_rotations(self, re_order=True) -> "PauliwordOp":
#
#         PauliOp = self.copy()
#         if re_order:
#             re_order_inds = PauliOp.lexicographical_sort()
#             PauliOp.symp_matrix = PauliOp.symp_matrix[re_order_inds]
#             PauliOp.coeff_vec = PauliOp.coeff_vec[re_order_inds]
#
#         # make list of rotations empty!
#         self.X_sk_rotations = []
#
#         # recursively perform UP
#         reduced_op = self._recursive_unitary_partitioning(PauliOp)
#
#         assert (reduced_op.n_terms == 1), 'not reducing to a single term'
#         return reduced_op
#
#     def _recursive_unitary_partitioning(self, AC_op: PauliwordOp, zero_threshold=1e-15)  -> "PauliwordOp":
#
#         if AC_op.n_terms == 1:
#             return AC_op
#         else:
#             op_for_rotation = AC_op.copy()
#             # take β_s P_s
#             P_s = PauliwordOp(op_for_rotation.symp_matrix[0], [1])
#             β_s = op_for_rotation.coeff_vec[0]
#
#             β_k = op_for_rotation.coeff_vec[1]
#
#             theta_sk = np.arctan(β_k / β_s)
#             if β_s.real < 0:
#                 theta_sk = theta_sk + np.pi
#             #             assert(np.isclose((β_k*np.cos(theta_sk) - β_s*np.sin(theta_sk)), 0)), 'term not zeroing out'
#
#             # -X_sk = -1j * Ps @ Pk
#             jP_k = PauliwordOp(op_for_rotation.symp_matrix[1], [-1j])
#             X_sk = P_s * jP_k
#             if X_sk.coeff_vec[0].real < 0:
#                 X_sk.coeff_vec[0] *= -1
#                 theta_sk *= -1
#
#             self.X_sk_rotations.append((X_sk, theta_sk))
#             AC_op_rotated = op_for_rotation._rotate_by_single_Pword(X_sk, theta_sk)
#
#             # remove zeros (WITHOUT RE-ORDERING... don't use cleanup of PauliwordOp)
#             mask_nonzero = np.where(abs(AC_op_rotated.coeff_vec) > zero_threshold)
#             AC_op_rotated = PauliwordOp(AC_op_rotated.symp_matrix[mask_nonzero],
#                                         AC_op_rotated.coeff_vec[mask_nonzero])
#             del mask_nonzero
#
#             return self._recursive_unitary_partitioning(AC_op_rotated)
#
#     def gen_LCU(self, s_index=None, check_reduction=True) -> "PauliwordOp":
#
#         PauliOp = self.copy()
#
#         # make list of rotations empty!
#         self.LCU = []
#
#         if s_index is None:
#             s_index = self.get_least_dense_term_index()
#         P_s = PauliwordOp(self.symp_matrix[s_index], [1])
#         β_s = self.coeff_vec[s_index]
#
#         #  ∑ β_k 𝑃_k  ... note this doesn't contain 𝛽_s 𝑃_s
#         symp_matrix_no_Ps = np.delete(PauliOp.symp_matrix, s_index, axis=0)
#         coeff_vec_no_βs = np.delete(PauliOp.coeff_vec, s_index, axis=0)
#
#         # Ω_𝑙 ∑ 𝛿_k 𝑃_k  ... renormalized!
#         omega_l = np.linalg.norm(coeff_vec_no_βs)
#         coeff_vec_no_βs = coeff_vec_no_βs / omega_l
#
#         phi_n_1 = np.arccos(β_s)
#         # require sin(𝜙_{𝑛−1}) to be positive...
#         if (phi_n_1 > np.pi):
#             phi_n_1 = 2 * np.pi - phi_n_1
#
#         alpha = phi_n_1
#
#         I_term = 'I' * P_s.n_qubits
#         R_LCU = PauliwordOp({I_term: np.cos(alpha / 2)})
#
#         sin_term = -np.sin(alpha / 2)
#
#         for dk, Pk in zip(coeff_vec_no_βs, symp_matrix_no_Ps):
#             PkPs = PauliwordOp(Pk, [dk]) * P_s
#             R_LCU += PkPs.multiply_by_constant(sin_term)
#
#         if check_reduction is True:
#             #         reduced_op = (R_LCU * PauliOp * R_LCU.conjugate).cleanup_zeros()
#             #         assert(reduced_op.n_terms==1), 'not reducing to a single term'
#             R_AC_set = (R_LCU * PauliOp).cleanup_zeros()
#             R_AC_set_R_dagg = (R_AC_set * R_LCU.conjugate).cleanup_zeros()
#             assert (R_AC_set_R_dagg.n_terms == 1), 'not reducing to a single term'
#             del R_AC_set
#             del R_AC_set_R_dagg
#
#         return R_LCU

class AntiCommutingOp(PauliwordOp):

    def __init__(self,
                 AC_operator: Union[List[str], Dict[str, float], np.array],
                 coeff_list: Union[List[complex], np.array] = None):
        super().__init__(AC_operator, coeff_list)

        # check all operators anticommute
        anti_comm_check = self.adjacency_matrix.astype(int) - np.eye(self.adjacency_matrix.shape[0])
        assert (np.einsum('ij->', anti_comm_check) == 0), 'operator needs to be made of anti-commuting Pauli operators'

        # normalization factor
        self.gamma_l = np.linalg.norm(self.coeff_vec)
        # normalize coefficients
        self.coeff_vec = self.coeff_vec / self.gamma_l

        self.X_sk_rotations = None
        self.LCU = None

    def lexicographical_sort(self):
        """
        sort object into lexicographical order

        Returns:
            None
        """
        # convert sym form to list of ints
        int_list = self.symp_matrix @ (1 << np.arange(self.symp_matrix.shape[1], dtype=object)[::-1])
        lex_ordered_indices = np.argsort(int_list)

        self.symp_matrix = self.symp_matrix[lex_ordered_indices]
        self.coeff_vec = self.coeff_vec[lex_ordered_indices]

        return None

    def get_least_dense_term_index(self):

        # np.logical_or(X_block, Z_block)
        pos_terms_occur = np.logical_or(self.symp_matrix[:, :self.n_qubits], self.symp_matrix[:, self.n_qubits:])

        int_list = pos_terms_occur @ (1 << np.arange(pos_terms_occur.shape[1], dtype=object)[::-1])
        s_index = np.argmin(int_list)
        return s_index

    def gen_seq_rotations(self, s_index=None, check_reduction=False) -> "PauliwordOp":

        PauliOp = self.copy()
        if s_index is None:
            s_index = self.get_least_dense_term_index()
            # make s_index be the zeroth index
            re_order_inds = np.array([s_index, *np.setdiff1d(np.arange(self.n_terms), s_index)])
            PauliOp.symp_matrix = PauliOp.symp_matrix[re_order_inds]
            PauliOp.coeff_vec = PauliOp.coeff_vec[re_order_inds]

        # make list of rotations empty!
        self.X_sk_rotations = []

        # recursively perform UP till only P_s remains
        P_s = self._recursive_unitary_partitioning(PauliOp)
        assert (P_s.n_terms == 1), 'not reducing to a single term'

        if check_reduction:
            AC_op_rotated = PauliOp.copy()
            for X_sk, theta_sk in self.X_sk_rotations:
                AC_op_rotated = AC_op_rotated._rotate_by_single_Pword(X_sk, theta_sk)

                # remove zeros (WITHOUT RE-ORDERING... don't use cleanup of PauliwordOp)
                zero_threshold=1e-12
                mask_nonzero = np.where(abs(AC_op_rotated.coeff_vec) > zero_threshold)
                AC_op_rotated = PauliwordOp(AC_op_rotated.symp_matrix[mask_nonzero],
                                            AC_op_rotated.coeff_vec[mask_nonzero])
                del mask_nonzero
            assert (AC_op_rotated.n_terms == 1), 'rotations not reducing to a single term'

        return self.X_sk_rotations, P_s

    def _recursive_unitary_partitioning(self, AC_op: PauliwordOp) -> "PauliwordOp": #zero_threshold=1e-12)-> "PauliwordOp":
        """
        hidden recursive function assumes term reducing too has been positioned at the begninning of AC_op (aka at
        index position 0 = s_index). Function then recursively removes the terms in order starting at index 1.
        Args:
            AC_op (PauliwordOp): A PauliwordOp made of normalised anticommuting operators

        Returns:

        """
        if AC_op.n_terms == 1:
            return AC_op
        else:
            op_for_rotation = AC_op.copy()
            # take β_s P_s
            ### s_index=0
            k_index=1
            P_s = PauliwordOp(op_for_rotation.symp_matrix[0], [1])
            β_s = op_for_rotation.coeff_vec[0]

            β_k = op_for_rotation.coeff_vec[1]

            theta_sk = np.arctan(β_k / β_s)
            if β_s.real < 0:
                theta_sk = theta_sk + np.pi

            # check
            assert(np.isclose((β_k*np.cos(theta_sk) - β_s*np.sin(theta_sk)), 0)), 'term not zeroing out'

            # -X_sk = -1j * Ps @ Pk
            jP_k = PauliwordOp(op_for_rotation.symp_matrix[1], [-1j])
            X_sk = P_s * jP_k
            if X_sk.coeff_vec[0].real < 0:
                X_sk.coeff_vec[0] *= -1
                theta_sk *= -1

            self.X_sk_rotations.append((X_sk, theta_sk))

            op_for_rotation.coeff_vec[0] = np.sqrt(β_s**2 + β_k**2)
            op_for_rotation.coeff_vec[1] = 0

            # build op without k term and (included modified s term)
            AC_op_rotated = PauliwordOp(np.delete(op_for_rotation.symp_matrix, k_index, axis=0),
                                        np.delete(op_for_rotation.coeff_vec, k_index, axis=0))

            ## know how operator acts therefore don't need to actually do rotations
            # AC_op_rotated = op_for_rotation._rotate_by_single_Pword(X_sk, theta_sk)

            # remove zeros (WITHOUT RE-ORDERING... don't use cleanup of PauliwordOp)
            # mask_nonzero = np.where(abs(op_for_rotation.coeff_vec) > zero_threshold)
            # AC_op_rotated = PauliwordOp(op_for_rotation.symp_matrix[mask_nonzero],
            #                             op_for_rotation.coeff_vec[mask_nonzero])
            # del mask_nonzero
            # del op_for_rotation

            return self._recursive_unitary_partitioning(AC_op_rotated)

    def gen_LCU(self, s_index=None, check_reduction=False) -> "PauliwordOp":

        PauliOp = self.copy()

        # make list of rotations empty!
        self.LCU = []

        if s_index is None:
            s_index = self.get_least_dense_term_index()
        P_s = PauliwordOp(self.symp_matrix[s_index], [1])
        β_s = self.coeff_vec[s_index]

        #  ∑ β_k 𝑃_k  ... note this doesn't contain 𝛽_s 𝑃_s
        symp_matrix_no_Ps = np.delete(PauliOp.symp_matrix, s_index, axis=0)
        coeff_vec_no_βs = np.delete(PauliOp.coeff_vec, s_index, axis=0)

        # Ω_𝑙 ∑ 𝛿_k 𝑃_k  ... renormalized!
        omega_l = np.linalg.norm(coeff_vec_no_βs)
        coeff_vec_no_βs = coeff_vec_no_βs / omega_l

        phi_n_1 = np.arccos(β_s)
        # require sin(𝜙_{𝑛−1}) to be positive...
        if (phi_n_1 > np.pi):
            phi_n_1 = 2 * np.pi - phi_n_1

        alpha = phi_n_1

        I_term = 'I' * P_s.n_qubits
        R_LCU = PauliwordOp({I_term: np.cos(alpha / 2)})

        sin_term = -np.sin(alpha / 2)

        for dk, Pk in zip(coeff_vec_no_βs, symp_matrix_no_Ps):
            PkPs = PauliwordOp(Pk, [dk]) * P_s
            R_LCU += PkPs.multiply_by_constant(sin_term)

        if check_reduction is True:
            # reduced_op = (R_LCU * PauliOp * R_LCU.conjugate).cleanup_zeros()
            # assert(reduced_op.n_terms==1), 'not reducing to a single term'
            R_AC_set = (R_LCU * PauliOp).cleanup_zeros()
            R_AC_set_R_dagg = (R_AC_set * R_LCU.conjugate).cleanup_zeros()
            assert (R_AC_set_R_dagg.n_terms == 1), 'not reducing to a single term'
            del R_AC_set
            del R_AC_set_R_dagg

        return R_LCU, P_s