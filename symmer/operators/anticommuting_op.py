from symmer.operators import PauliwordOp, IndependentOp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.simplefilter('always', UserWarning)
class AntiCommutingOp(PauliwordOp):

    def __init__(self,
                 AC_op_symp_matrix: np.array,
                 coeff_list: np.array):
        super().__init__(AC_op_symp_matrix, coeff_list)

        # check all operators anticommute
        anti_comm_check = self.adjacency_matrix.astype(int) - np.eye(self.adjacency_matrix.shape[0])
        assert (np.sum(anti_comm_check) == 0), 'operator needs to be made of anti-commuting Pauli operators'

        self.X_sk_rotations = []
        self.R_LCU = None

    @classmethod
    def from_list(cls,
            pauli_terms :List[str],
            coeff_vec:   List[complex] = None
        ) -> "AntiCommutingOp":
        PwordOp = super().from_list(pauli_terms, coeff_vec)
        return cls.from_PauliwordOp(PwordOp)

    @classmethod
    def from_dictionary(cls,
            operator_dict: Dict[str, complex]
        ) -> "AntiCommutingOp":
        """ Initialize a PauliwordOp from its dictionary representation {pauli:coeff, ...}
        """
        PwordOp = super().from_dictionary(operator_dict)
        return cls.from_PauliwordOp(PwordOp)

    @classmethod
    def from_PauliwordOp(cls,
            PwordOp: PauliwordOp
        ) -> 'AntiCommutingOp':
        return cls(PwordOp.symp_matrix, PwordOp.coeff_vec)


    def get_least_dense_term_index(self):
        """
        Takes the current symp_matrix of object and finds the index of the least dense Pauli
        operator (aka least Pauli matrices). This can be used to define the term to reduce too

        Note one needs to re-run this function if ordering changed (e.g. if lexicographical_sort is run)

        Returns:
            s_index (int): index of least dense term in objects symp_matrix and coeff_vec
        """
        ### np.logical_or(X_block, Z_block)
        # pos_terms_occur = np.logical_or(self.symp_matrix[:, :self.n_qubits], self.symp_matrix[:, self.n_qubits:])
        # int_list = pos_terms_occur @ (1 << np.arange(pos_terms_occur.shape[1], dtype=object)[::-1])
        # s_index = np.argmin(int_list)

        pos_terms_occur = np.logical_or(self.symp_matrix[:, :self.n_qubits], self.symp_matrix[:, self.n_qubits:])
        symp_matrix_view = np.ascontiguousarray(pos_terms_occur).view(
            np.dtype((np.void, pos_terms_occur.dtype.itemsize * pos_terms_occur.shape[1]))
        )
        sort_order = np.argsort(symp_matrix_view.ravel())
        s_index = sort_order[0]

        return s_index


    def _recursive_seq_rotations(self, AC_op: PauliwordOp) -> "PauliwordOp":
        if AC_op.n_terms == 1:
            return AC_op
        else:
            # s_index fixed to zero (re-order done in unitary_partitioning method!)
            s_index = 0
            k_index = 1

            op_for_rotation = AC_op.copy()

            # take β_s P_s
            P_s = PauliwordOp(op_for_rotation.symp_matrix[s_index], [1])
            β_s = op_for_rotation.coeff_vec[s_index]
            β_k = op_for_rotation.coeff_vec[k_index]

            theta_sk = np.arctan(β_k / β_s)
            if β_s.real < 0:
                theta_sk = theta_sk + np.pi

            # check
            assert (np.isclose((β_k * np.cos(theta_sk) - β_s * np.sin(theta_sk)), 0)), 'term not zeroing out'

            # -X_sk = -1j * Ps @ Pk
            jP_k = PauliwordOp(op_for_rotation.symp_matrix[k_index], [-1j])
            X_sk = P_s * jP_k
            if X_sk.coeff_vec[0].real < 0:
                X_sk.coeff_vec[0] *= -1
                theta_sk *= -1

            self.X_sk_rotations.append((X_sk, theta_sk))

            # update coeffs
            op_for_rotation.coeff_vec[s_index] = np.sqrt(β_s ** 2 + β_k ** 2)
            op_for_rotation.coeff_vec[k_index] = 0

            # build op without k term and (included modified s term)
            AC_op_rotated = PauliwordOp(np.delete(op_for_rotation.symp_matrix, k_index, axis=0),
                                        np.delete(op_for_rotation.coeff_vec, k_index, axis=0))

            ## know how operator acts therefore don't need to actually do rotations

            return self._recursive_seq_rotations(AC_op_rotated)


    def unitary_partitioning(self, s_index: int=None, up_method: Optional[str]='seq_rot') \
            -> Tuple[PauliwordOp, Union[PauliwordOp, List[Tuple[PauliwordOp, float]]], float, "AntiCommutingOp"]:
        """
        Apply unitary partitioning on anticommuting operator (self)

        Args:
            s_index (int): index of row in symplectic matrix that defines Pauli operator to reduce too (Ps).
                           if set to None then code will find least dense Pauli operator and use that.
            up_method (str): unitary partitoning method ['LCU', 'seq_rot']

        Returns:
            Ps (PauliwordOp): Pauli operator of term reduced too
            rotations (PauliwordOp): rotations to perform unitary partitioning
            gamma_l (float): normalization constant of clique (anticommuting operator)
            AC_op (AntiCommutingOp): normalized clique - i.e. self == gamma_l * AC_op
        """
        assert up_method in ['LCU', 'seq_rot'], f'unknown unitary partitioning method: {up_method}'
        AC_op = self.copy()

        if AC_op.n_terms == 1:
            rotations = None
            gamma_l = np.linalg.norm(AC_op.coeff_vec)
            AC_op.coeff_vec = AC_op.coeff_vec / gamma_l
            Ps = PauliwordOp(AC_op.symp_matrix, [1])
            return Ps, rotations, gamma_l, AC_op

        else:

            assert np.isclose(np.sum(AC_op.coeff_vec.imag), 0), 'cannot apply unitary partitioning to operator with complex coeffs'

            gamma_l = np.linalg.norm(AC_op.coeff_vec)
            AC_op.coeff_vec = AC_op.coeff_vec / gamma_l

            if s_index is None:
                s_index = self.get_least_dense_term_index()

            if s_index!=0:
                # re-order so s term is ALWAYS at top of symplectic matrix and thus is index as 0!
                ### assert s_index <= AC_op.n_terms-1, 's_index out of range'
                AC_op.coeff_vec[[0, s_index]] = AC_op.coeff_vec[[s_index, 0]]
                AC_op.symp_matrix[[0, s_index]] = AC_op.symp_matrix[[s_index, 0]]
                AC_op = AntiCommutingOp(AC_op.symp_matrix, AC_op.coeff_vec) # need to reinit otherwise Z and X blocks wrong

            # assert not np.isclose(AC_op.coeff_vec[0], 0), f's_index cannot have zero coefficent: {AC_op.coeff_vec[0]}'
            if np.isclose(AC_op[0].coeff_vec, 0):
                # need to correct for s_index having zero coeff... then need to swap to nonzero index
                non_zero_index = np.argmax(abs(AC_op.coeff_vec))
                AC_op.coeff_vec[[0, non_zero_index]] = AC_op.coeff_vec[[non_zero_index, 0]]
                AC_op.symp_matrix[[0, non_zero_index]] = AC_op.symp_matrix[[non_zero_index, 0]]

            if up_method=='seq_rot':
                if len(self.X_sk_rotations)!=0:
                    self.X_sk_rotations = []
                Ps = self._recursive_seq_rotations(AC_op)
                rotations = self.X_sk_rotations
            elif up_method=='LCU':
                if self.R_LCU is not None:
                    self.R_LCU = None

                Ps = self.generate_LCU_operator(AC_op)
                rotations = self.R_LCU
            else:
                raise ValueError(f'unknown unitary partitioning method: {up_method}!')

            return Ps, rotations, gamma_l, AC_op

    def generate_LCU_operator(self, AC_op) -> PauliwordOp:
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
        # need to remove zero coeff terms
        AC_op_cpy = AC_op.copy()
        before_cleanup = AC_op_cpy.n_terms
        AC_op = AC_op_cpy[np.where(abs(AC_op.coeff_vec)>1e-15)[0]]
        post_cleanup = AC_op.n_terms
        # AC_op = AC_op.cleanup(zero_threshold=1e-15)  ## cleanup re-orders which is BAD for s_index


        if (before_cleanup>1 and post_cleanup==1):
            if AC_op.coeff_vec[0]<0:
                # need to fix neg sign (use Pauli multiplication)

                # as s index defaults to 0, take the next term (in CS-VQE this will commute with symmetries)!
                if np.isclose(AC_op_cpy[0].coeff_vec, 0):
                    # need to correct for s_index having zero coeff... then need to swap to nonzero index
                    non_zero_index = np.argmax(abs(AC_op_cpy.coeff_vec))

                    AC_op_cpy.coeff_vec[[0, non_zero_index]] = AC_op_cpy.coeff_vec[[non_zero_index, 0]]
                    AC_op_cpy.symp_matrix[[0, non_zero_index]] = AC_op_cpy.symp_matrix[[non_zero_index, 0]]


                sign_correction = PauliwordOp(AC_op_cpy.symp_matrix[1],[1])

                self.R_LCU = sign_correction
                Ps_LCU = PauliwordOp(AC_op.symp_matrix, [1])
            else:
                self.R_LCU = PauliwordOp.from_list(['I'*AC_op.n_qubits])
                Ps_LCU = PauliwordOp(AC_op.symp_matrix, AC_op.coeff_vec)
        else:
            s_index=0

            # note gamma_l norm applied on init!
            Ps_LCU = PauliwordOp(AC_op.symp_matrix[s_index], [1])
            βs = AC_op.coeff_vec[s_index]

            #  ∑ β_k 𝑃_k  ... note this doesn't contain 𝛽_s 𝑃_s
            no_βsPs = AC_op - (Ps_LCU.multiply_by_constant(βs))

            # Ω_𝑙 ∑ 𝛿_k 𝑃_k  ... renormalized!
            omega_l = np.linalg.norm(no_βsPs.coeff_vec)
            no_βsPs.coeff_vec = no_βsPs.coeff_vec / omega_l

            phi_n_1 = np.arccos(βs)
            # require sin(𝜙_{𝑛−1}) to be positive...
            if (phi_n_1 > np.pi):
                phi_n_1 = 2 * np.pi - phi_n_1

            alpha = phi_n_1
            I_term = 'I' * Ps_LCU.n_qubits
            self.R_LCU = PauliwordOp.from_dictionary({I_term: np.cos(alpha / 2)})

            sin_term = -np.sin(alpha / 2)

            for dkPk in no_βsPs:
                dk_PkPs = dkPk * Ps_LCU
                self.R_LCU += dk_PkPs.multiply_by_constant(sin_term)

        return Ps_LCU

# from symmer.operators.utils import mul_symplectic
# def conjugate_Pop_with_R(Pop:PauliwordOp,
#                         R: PauliwordOp) -> PauliwordOp:
#     """
#     For a defined linear combination of pauli operators : R = ∑_{𝑖} ci Pi ... (note each P self-adjoint!)
#
#     perform the adjoint rotation R op R† =  R [∑_{a} ca Pa] R†
#
#     Args:
#         R (PauliwordOp): operator to rotate Pop by
#     Returns:
#         rot_H (PauliwordOp): rotated operator
#
#     ### Notes
#     R = ∑_{𝑖} ci Pi
#     R^{†} = ∑_{j}  cj^{*} Pj
#     note i and j here run over the same indices!
#     apply R H R^{†} where H is Pop (current Pauli defined in class object)
#
#     ### derivation:
#
#     = (∑_{𝑖} ci Pi ) * (∑_{a} ca Pa ) * ∑_{j} cj^{*} Pj
#
#     = ∑_{a}∑_{i}∑_{j} (ci ca cj^{*}) Pi  Pa Pj
#
#     # can write as case for when i==j and i!=j
#
#     = ∑_{a}∑_{i=j} (ci ca ci^{*}) Pi  Pa Pi + ∑_{a}∑_{i}∑_{j!=i} (ci ca cj^{*}) Pi  Pa Pj
#
#     # let C by the termwise commutator matrix between H and R
#     = ∑_{a}∑_{i=j} (-1)^{C_{ia}} (ci ca ci^{*}) Pa  + ∑_{a}∑_{i}∑_{j!=i} (ci ca cj^{*}) Pi  Pa Pj
#
#     # next write final term over upper triange (as i and j run over same indices)
#     ## so add common terms for i and j and make j>i
#
#     = ∑_{a}∑_{i=j} (-1)^{C_{ia}} (ci ca ci^{*}) Pa
#       + ∑_{a}∑_{i}∑_{j>i} (ci ca cj^{*}) Pi  Pa Pj + (cj ca ci^{*}) Pj  Pa Pi
#
#     # then need to know commutation relation betwen terms in R
#     ## given by adjaceny matrix of R... here A
#
#
#     = ∑_{a}∑_{i=j} (-1)^{C_{ia}} (ci ca ci^{*}) Pa
#      + ∑_{a}∑_{i}∑_{j>i} (ci ca cj^{*}) Pi  Pa Pj + (-1)^{C_{ia}+A_{ij}+C_{ja}}(cj ca ci^{*}) Pi  Pa Pj
#
#
#     = ∑_{a}∑_{i=j} (-1)^{C_{ia}} (ci ca ci^{*}) Pa
#      + ∑_{a}∑_{i}∑_{j>i} (ci ca cj^{*} + (-1)^{C_{ia}+A_{ij}+C_{ja}}(cj ca ci^{*})) Pi  Pa Pj
#
#
#     """
#     if Pop.n_terms == 1:
#         rot_H = (R * Pop * R.dagger).cleanup()
#     else:
#         # anticommutes == 1 and commutes == 0
#         commutation_check = (~Pop.commutes_termwise(R)).astype(int)
#         adj_matrix = (~R.adjacency_matrix).astype(int)
#
#         c_list = R.coeff_vec
#         ca_list = Pop.coeff_vec
#
#         # rot_H = PauliwordOp.empty(Pop.n_qubits)
#         coeff_list = []
#         sym_vec_list = []
#         for ind_a, Pa_vec in enumerate(Pop.symp_matrix):
#             for ind_i, Pi_vec in enumerate(R.symp_matrix):
#                 sign = (-1) ** (commutation_check[ind_a, ind_i])
#
#                 coeff = c_list[ind_i] * c_list[ind_i].conj() * ca_list[ind_a]
#                 # rot_H += PauliwordOp(Pa_vec, [coeff * sign])
#                 sym_vec_list.append(Pa_vec)
#                 coeff_list.append(coeff * sign)
#
#                 # PiPa = PauliwordOp(Pi_vec, [1]) * PauliwordOp(Pa_vec, [1])
#                 phaseless_prod_PiPa, PiPa_coeff_vec = mul_symplectic(Pi_vec, 1,
#                                                                      Pa_vec, 1)
#
#                 for ind_j, Pj_vec in enumerate(R.symp_matrix[ind_i + 1:]):
#                     ind_j += ind_i + 1
#
#                     sign2 = (-1) ** (commutation_check[ind_a, ind_i] +
#                                      commutation_check[ind_a, ind_j] +
#                                      adj_matrix[ind_i, ind_j])
#
#                     coeff1 = c_list[ind_i] * ca_list[ind_a] * c_list[ind_j].conj()
#                     coeff2 = c_list[ind_j] * ca_list[ind_a] * c_list[ind_i].conj()
#
#                     overall_coeff = (coeff1 + sign2 * coeff2)
#                     if overall_coeff:
#                         # calculate PiPa term outside of j loop
#                         # overall_coeff_PiPaPj = PiPa * PauliwordOp(Pj_vec, [overall_coeff])
#                         # rot_H += overall_coeff_PiPaPj
#
#                         phaseless_prod, coeff_vec = mul_symplectic(phaseless_prod_PiPa, PiPa_coeff_vec,
#                                                                    Pj_vec, overall_coeff)
#                         sym_vec_list.append(phaseless_prod)
#                         coeff_list.append(coeff_vec)
#
#         rot_H = PauliwordOp(np.array(sym_vec_list),
#                             coeff_list).cleanup()
#     return rot_H
#
