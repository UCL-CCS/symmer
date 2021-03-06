from symmer.symplectic import PauliwordOp
import numpy as np
from typing import Dict, List, Union, Tuple

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

    def lexicographical_sort(self) -> None:
        """
        sort object into lexicographical order (changes symp_matrix and coeff_vec order of PauliWord)

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
        """
        Takes the current symp_matrix of object and finds the index of the least dense Pauli
        operator (aka least Pauli matrices). This can be used to define the term to reduce too

        Note one needs to re-run this function if ordering changed (e.g. if lexicographical_sort is run)

        Returns:
            s_index (int): index of least dense term in objects symp_matrix and coeff_vec
        """
        # np.logical_or(X_block, Z_block)
        pos_terms_occur = np.logical_or(self.symp_matrix[:, :self.n_qubits], self.symp_matrix[:, self.n_qubits:])

        int_list = pos_terms_occur @ (1 << np.arange(pos_terms_occur.shape[1], dtype=object)[::-1])
        s_index = np.argmin(int_list)
        return s_index

    def gen_seq_rotations(self, s_index=None, check_reduction=False) -> Tuple[List[tuple], "PauliwordOp"]:
        """
        Given the normalized anticommuting operator object, function takes current symp ordering and
        finds the Sequence of Rotations (https://arxiv.org/pdf/1908.08067.pdf) to reduce the operator
        to a single Pauli operator in the operator.

        Note if s_index is set to none then the least dense term is reduced too. Also this function DEPENDS on the
        current order of the objects symplectic matrix. Therefore order operators properly first before using function
        e.g. run lexicographical_sort or a manual ordering

        Args:
            s_index (int): optional integar defining index of term to reduce too. Note if not set then defaults to
                           least dense Pauli operator in AC operator.
            check_reduction (bool): flag to check reduction by applying sequence of rotations on original operator.

        Returns:
            X_sk_rotations (list): list of tuples contains rotations operator and angle. Note order of implementation matters
                                    and should be started from zeroth index onwards.
             P_s (PauliwordOp): single PauliwordOp that has been reduced too.
        """
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

                # # remove zeros (WITHOUT RE-ORDERING... don't use cleanup of PauliwordOp)
                # zero_threshold=1e-12
                # mask_nonzero = np.where(abs(AC_op_rotated.coeff_vec) > zero_threshold)
                # AC_op_rotated = PauliwordOp(AC_op_rotated.symp_matrix[mask_nonzero],
                #                             AC_op_rotated.coeff_vec[mask_nonzero])
                # del mask_nonzero
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
            # take ??_s P_s
            ###??s_index=0
            k_index=1
            P_s = PauliwordOp(op_for_rotation.symp_matrix[0], [1])
            ??_s = op_for_rotation.coeff_vec[0]

            ??_k = op_for_rotation.coeff_vec[1]

            theta_sk = np.arctan(??_k / ??_s)
            if ??_s.real < 0:
                theta_sk = theta_sk + np.pi

            # check
            assert(np.isclose((??_k*np.cos(theta_sk) - ??_s*np.sin(theta_sk)), 0)), 'term not zeroing out'

            # -X_sk = -1j * Ps @ Pk
            jP_k = PauliwordOp(op_for_rotation.symp_matrix[1], [-1j])
            X_sk = P_s * jP_k
            if X_sk.coeff_vec[0].real < 0:
                X_sk.coeff_vec[0] *= -1
                theta_sk *= -1

            self.X_sk_rotations.append((X_sk, theta_sk))

            op_for_rotation.coeff_vec[0] = np.sqrt(??_s**2 + ??_k**2)
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

    def gen_LCU(self, s_index=None, check_reduction=False)-> Tuple["PauliwordOp", "PauliwordOp"]:
        """
        Given the normalized anticommuting operator object, function takes current symp ordering and
        finds the linear combination of unitaries (LCU) (https://arxiv.org/pdf/1908.08067.pdf) to reduce the operator
        to a single Pauli operator in the operator.

        Note if s_index is set to none then the least dense term is reduced too. Unlike the gen_seq_rotations method
        this approach doesn't acutally matter what order the objects symplectic matrix.


        Args:
            s_index (int): optional integar defining index of term to reduce too. Note if not set then defaults to
                           least dense Pauli operator in AC operator.
            check_reduction (bool): flag to check reduction by applying LCU on original operator.

        Returns:
            R_LCU (PauliwordOp): PauliwordOp that is a linear combination of unitaries
            P_s (PauliwordOp): single PauliwordOp that has been reduced too.
        """

        PauliOp = self.copy()

        if PauliOp.n_terms==1:
            R_LCU = None
            return R_LCU, PauliOp


        # make list of rotations empty!
        self.LCU = []

        if s_index is None:
            s_index = self.get_least_dense_term_index()
        P_s = PauliwordOp(self.symp_matrix[s_index], [1])
        ??_s = self.coeff_vec[s_index]

        #  ??? ??_k ????_k  ... note this doesn't contain ????_s ????_s
        symp_matrix_no_Ps = np.delete(PauliOp.symp_matrix, s_index, axis=0)
        coeff_vec_no_??s = np.delete(PauliOp.coeff_vec, s_index, axis=0)

        # ??_???? ??? ????_k ????_k  ... renormalized!
        omega_l = np.linalg.norm(coeff_vec_no_??s)
        coeff_vec_no_??s = coeff_vec_no_??s / omega_l

        phi_n_1 = np.arccos(??_s)
        # require sin(????_{???????1}) to be positive...
        if (phi_n_1 > np.pi):
            phi_n_1 = 2 * np.pi - phi_n_1

        alpha = phi_n_1

        I_term = 'I' * P_s.n_qubits
        R_LCU = PauliwordOp({I_term: np.cos(alpha / 2)})

        sin_term = -np.sin(alpha / 2)

        for dk, Pk in zip(coeff_vec_no_??s, symp_matrix_no_Ps):
            PkPs = PauliwordOp(Pk, [dk]) * P_s
            R_LCU += PkPs.multiply_by_constant(sin_term)

        if check_reduction is True:
            # R_AC_set = R_LCU * PauliOp
            # R_AC_set_R_dagg = R_AC_set * R_LCU.conjugate
            # assert (R_AC_set_R_dagg.n_terms == 1), 'not reducing to a single term'
            # del R_AC_set
            # del R_AC_set_R_dagg

            ### faster to check R_LCU @ P_s @ R_LCU_dag = AC_op (normalized)
            AC_op_gen = R_LCU.conjugate * P_s * R_LCU
            assert (AC_op_gen == self), 'rotations NOT performing unitary part LCU correclty'
            del AC_op_gen

        return R_LCU, P_s


def apply_LCU_to_operator(op_to_rotate: PauliwordOp,
                          LCU_operator: PauliwordOp):

    dI = LCU_operator.coeff_vec[0]
    dI2 = dI**2
    non_I_terms = LCU_operator[1:]
    rot_H = op_to_rotate.multiply_by_constant(dI2)  # first terms
    for j, P_jk in enumerate(non_I_terms):
        for ci_Pi in op_to_rotate:
            vi_Pi = P_jk * ci_Pi * P_jk.conjugate
            rot_H += vi_Pi

            P_kl = P_jk.conjugate
            Pi_Pkl = (ci_Pi * P_kl).multiply_by_constant(dI)
            Pjk_Pi = (P_jk * ci_Pi).multiply_by_constant(dI)

            rot_H += (Pi_Pkl + Pjk_Pi)

            for l, P_lk in enumerate(non_I_terms[(j + 1):]):
                P_jk_Pi_P_kl = P_jk * ci_Pi * P_lk.conjugate
                P_lk_Pi_P_kj = P_lk * ci_Pi * P_jk.conjugate
                rot_H += (P_jk_Pi_P_kl + P_lk_Pi_P_kj)
    return rot_H


def apply_LCU_to_operator_only_necessary_ops(op_to_rotate: PauliwordOp,
                          LCU_operator: PauliwordOp):

    if LCU_operator is not None:
        dI = LCU_operator.coeff_vec[0]
        two_dI = 2*dI
        dI2 = dI**2

        non_I_terms = LCU_operator[1:]
        commutation_check = op_to_rotate.commutes_termwise(non_I_terms)

        rot_H = op_to_rotate.multiply_by_constant(dI2)  # first terms
        for j, P_jk in enumerate(non_I_terms):
            for i, ci_Pi in enumerate(op_to_rotate):

                # vi_Pi = P_jk * ci_Pi * P_jk.conjugate
                # rot_H += vi_Pi
                signed_bjk_blk_2 = (-1) ** (not commutation_check[i, j]) * P_jk.coeff_vec[0] * P_jk.coeff_vec[0].conjugate()
                vi_Pi = ci_Pi.multiply_by_constant(signed_bjk_blk_2)
                rot_H += vi_Pi

                # P_kl = P_jk.conjugate
                # Pi_Pkl = (ci_Pi * P_kl).multiply_by_constant(dI)
                # Pjk_Pi = (P_jk * ci_Pi).multiply_by_constant(dI)
                # rot_H += (Pi_Pkl + Pjk_Pi)
                if not commutation_check[i, j]:  # ci_Pi.commutes(P_rot)
                    ci_dj_PjkPi = P_jk * ci_Pi
                    rot_H += ci_dj_PjkPi.multiply_by_constant(two_dI)

                for l, P_lk in enumerate(non_I_terms[(j + 1):]):
                    # P_jk_Pi_P_kl = P_jk * ci_Pi * P_lk.conjugate
                    # P_lk_Pi_P_kj = P_lk * ci_Pi * P_jk.conjugate
                    # rot_H += (P_jk_Pi_P_kl + P_lk_Pi_P_kj)
                    l_ind = l + (j + 1)
                    if (commutation_check[i, j] and (not commutation_check[i, l_ind])):
                        Pi_Pjk_Pkl = ci_Pi * P_jk * P_lk.conjugate
                        rot_H += Pi_Pjk_Pkl.multiply_by_constant(2)

                    elif ((not commutation_check[i, j]) and (commutation_check[i, l_ind])):
                        Pi_Pkl_Pjk = ci_Pi * P_lk.conjugate * P_jk
                        rot_H += Pi_Pkl_Pjk.multiply_by_constant(2)
    return rot_H