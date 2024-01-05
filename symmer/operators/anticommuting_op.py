from symmer.operators import PauliwordOp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.simplefilter('always', UserWarning)
class AntiCommutingOp(PauliwordOp):

    def __init__(self,
                 AC_op_symp_matrix: np.array,
                 coeff_list: np.array):
        """
        Args:
            AC_op_symp_matrix (np.array): The symmetric matrix representation of the anti-commuting operator.
            coeff_list (np.array): The coefficient list associated with the anti-commuting operator.

        Raises:
            AssertionError: If the operators do not anticommute.
        """
        super().__init__(AC_op_symp_matrix, coeff_list)

        # check all operators anticommute
        adj_mat = self.adjacency_matrix
        adj_mat[np.diag_indices_from(adj_mat)] = False
        assert ~np.any(adj_mat), 'operator needs to be made of anti-commuting Pauli operators'

        self.X_sk_rotations = []
        self.R_LCU = None

    @classmethod
    def from_list(cls,
            pauli_terms :List[str],
            coeff_vec:   List[complex] = None
        ) -> "AntiCommutingOp":
        """
        Args:
            pauli_terms (List[str]): A list of Pauli terms represented as strings.
            coeff_vec (List[complex], optional): A list of complex coefficients associated with the Pauli terms.
                If not provided, the default coefficients are assumed to be 1.0 for each term.

        Returns:
            AntiCommutingOp: An AntiCommutingOp instance initialized from the list of Pauli terms.
        """
        PwordOp = super().from_list(pauli_terms, coeff_vec)
        return cls.from_PauliwordOp(PwordOp)

    @classmethod
    def from_dictionary(cls,
            operator_dict: Dict[str, complex]
        ) -> "AntiCommutingOp":
        """ 
        Initialize a PauliwordOp from its dictionary representation {pauli:coeff, ...}

        Args:
            operator_dict (Dict[str, complex]): A dictionary representing the operator, where the keys are Pauli strings
                and the values are complex coefficients.

        Returns:
            AntiCommutingOp: An AntiCommutingOp instance initialized from the dictionary representation.
        """
        PwordOp = super().from_dictionary(operator_dict)
        return cls.from_PauliwordOp(PwordOp)

    @classmethod
    def from_PauliwordOp(cls,
            PwordOp: PauliwordOp
        ) -> 'AntiCommutingOp':
        """
        Args:
            PwordOp (PauliwordOp): The PauliwordOp instance to initialize the AntiCommutingOp from.

        Returns:
            AntiCommutingOp: An AntiCommutingOp instance initialized from the given PauliwordOp.
        """
        return cls(PwordOp.symp_matrix, PwordOp.coeff_vec)


    def get_least_dense_term_index(self):
        """
        Takes the current symp_matrix of object and finds the index of the least dense Pauli
        operator (aka least Pauli matrices). This can be used to define the term to reduce too

        Note one needs to re-run this function if ordering changed (e.g. if lexicographical_sort is run)

        Returns:
            s_index (int): index of least dense term in objects symp_matrix and coeff_vec.
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
        """
        Args:
            AC_op (PauliwordOp): The AntiCommutingOp to apply sequence rotations to.

        Returns:
            PauliwordOp: The AntiCommutingOp after applying sequence rotations.
        """
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

        if s_index is None:
            s_index = self.get_least_dense_term_index()

        if np.isclose(self.coeff_vec[s_index], 0):
            # need to correct for s_index having zero coeff...
            s_index = np.argmax(abs(self.coeff_vec))
            warnings.warn(f's indexed term has zero coeff, s_index set to {s_index} so that nonzero operator is rotated onto')
       
        s_index = int(s_index)
        BsPs    = self[s_index]

        # NOTE: term to reduce to is at the top of sym matrix i.e. s_index of ZERO now!
        no_BsPs = (self - BsPs).cleanup()
        if (len(no_BsPs.coeff_vec)==1 and no_BsPs.coeff_vec[0]==0):
            AC_op = BsPs
        else:
            AC_op = BsPs.append(no_BsPs)

        if AC_op.n_terms == 1:
            rotations = []
            gamma_l = np.linalg.norm(AC_op.coeff_vec)
            AC_op.coeff_vec = AC_op.coeff_vec / gamma_l
            Ps = AC_op
            return Ps, rotations, gamma_l, self.multiply_by_constant(1/gamma_l)

        else:

            assert np.isclose(np.sum(AC_op.coeff_vec.imag), 0), 'cannot apply unitary partitioning to operator with complex coeffs'

            gamma_l = np.linalg.norm(AC_op.coeff_vec)
            AC_op.coeff_vec = AC_op.coeff_vec / gamma_l

            if up_method=='seq_rot':
                if len(self.X_sk_rotations)!=0:
                    self.X_sk_rotations = []
                Ps = self._recursive_seq_rotations(AC_op)
                rotations = self.X_sk_rotations
            elif up_method=='LCU':
                if self.R_LCU is not None:
                    self.R_LCU = None

                Ps = self.generate_LCU_operator(AC_op)
                rotations = LCU_as_seq_rot(self.R_LCU)
            else:
                raise ValueError(f'unknown unitary partitioning method: {up_method}!')

            return Ps, rotations, gamma_l, self.multiply_by_constant(1/gamma_l)
        
    def multiply_by_constant(self, constant: float) -> "AntiCommutingOp":
        """ Return AntiCommutingOp under constant multiplication
        """
        AC_op_copy = self.copy()
        AC_op_copy.coeff_vec *= constant
        return AC_op_copy
    
    @classmethod
    def random(cls, n_qubits: int, n_terms: Union[None, int]=None, apply_clifford=True) -> "AntiCommutingOp":
        """
        generate a random real coefficient anticommuting op

        """
        from symmer.utils import random_anitcomm_2n_1_PauliwordOp
        if n_terms is None:
            n_terms = 2*n_qubits+1

        assert n_terms<= 2*n_qubits+1, f'cannot have {n_terms} Pops on {n_qubits} qubits'
        return cls.from_PauliwordOp( random_anitcomm_2n_1_PauliwordOp(n_qubits, apply_clifford=apply_clifford)[:n_terms])

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
        ## s_index is ensured to be in zero position in unitary_partitioning method! 
        ## if using function without this method need to ensure term to rotate onto is the zeroth index of AC_op!
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

def LCU_as_seq_rot(R_LCU: PauliwordOp) -> List[Tuple[PauliwordOp, float]]:
    """
    Convert a unitary composed of a
    See equations 18 and 19 of https://arxiv.org/pdf/1907.09040.pdf

    number of rotations is 2*(R_LCU.n_terms-1), which can at most be 4*n_qubits

    Args:
        R_LCU (PauliwordOp): unitary composed as a normalized linear combination of imaginary anticommuting Pauli operators (excluding identity)
    Returns:
        expon_p_terms (list): list of rotations generated by Pauli operators to implement AC_op unitary

    ** Example use **

    from symmer.utils import random_anitcomm_2n_1_PauliwordOp
    from symmer.operators import AntiCommutingOp
    from symmer.evolution.exponentiation import exponentiate_single_Pop
    from symmer.utils import product_list

    nq = 3 # change (do not make too large as has exp checking cost)

    AC_2n1 = random_anitcomm_2n_1_PauliwordOp(nq)
    AC_op = AntiCommutingOp.from_PauliwordOp(AC_2n1)
    Ps_LCU, rotations_LCU, gamma_l, AC_normed = AC_op.unitary_partitioning(s_index=0, up_method= 'LCU')
    print(AC_normed.perform_rotations(rotations_LCU) == Ps_LCU)

    ## expensive check to see if operation is identical! should NOT do this when using
    a2 = product_list([exponentiate_single_Pop(P.multiply_by_constant(1j*angle/2)) for P, angle in rotations_LCU])
    print(AC_op.R_LCU == a2)
    """
    if isinstance(R_LCU, list) and len(R_LCU)==0:
        # case where there are no rotations
        return list()
    
    assert R_LCU.n_terms > 1, 'AC_op must have more than 1 term'
    assert np.isclose(np.linalg.norm(R_LCU.coeff_vec), 1), 'AC_op must be l2 normalized'

    expon_p_terms = []

    # # IF imaginary components the this makes real (but need phase correction later!)
    coeff_vec = R_LCU.coeff_vec.real + R_LCU.coeff_vec.imag

    # for k, c_k in enumerate(coeff_vec):
    #     P_k = R_LCU[k]
    #     theta_k = np.arcsin(c_k / np.linalg.norm(coeff_vec[:(k + 1)]))
    #     P_k.coeff_vec[0] = 1
    #     expon_p_terms.append(tuple((P_k, theta_k)))
    # ## phase correction - change angle by -pi in first rotation!
    # expon_p_terms[0] = (expon_p_terms[0][0], expon_p_terms[0][1]-np.pi)

    for k in range(1, R_LCU.n_terms):
        P_k = R_LCU[k]
        c_k = coeff_vec[k]
        theta_k = np.arcsin(c_k / np.linalg.norm(coeff_vec[:(k + 1)]))
        P_k.coeff_vec[0] = 1
        expon_p_terms.append(tuple((P_k, theta_k)))

    expon_p_terms = [*expon_p_terms, *expon_p_terms[::-1]]
    
    return expon_p_terms

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
