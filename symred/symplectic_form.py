from functools import reduce
from cached_property import cached_property
from openfermion import QubitOperator, MajoranaOperator, get_sparse_operator, FermionOperator, count_qubits, get_majorana_operator #, get_fermion_operator
from symred.utils import gf2_gaus_elim
import numpy as np
import scipy as sp
from copy import deepcopy
from typing import List, Union, Dict
from scipy.sparse import csr_matrix
import warnings
warnings.simplefilter('always', UserWarning)

def symplectic_to_string(symp_vec) -> str:
    """
    Returns string form of symplectic vector defined as (X | Z)

    Args:
        symp_vec (array): symplectic Pauliword array

    Returns:
        Pword_string (str): String version of symplectic array

    """
    n_qubits = len(symp_vec) // 2

    X_block = symp_vec[:n_qubits]
    Z_block = symp_vec[n_qubits:]

    Y_loc = np.bitwise_and(X_block, Z_block).astype(bool)
    X_loc = np.bitwise_xor(Y_loc, X_block).astype(bool)
    Z_loc = np.bitwise_xor(Y_loc, Z_block).astype(bool)

    char_aray = np.array(list('I' * n_qubits), dtype=str)

    char_aray[Y_loc] = 'Y'
    char_aray[X_loc] = 'X'
    char_aray[Z_loc] = 'Z'

    Pword_string = ''.join(char_aray)

    return Pword_string


def string_to_symplectic(pauli_str, n_qubits):
    """
    """
    assert(len(pauli_str) == n_qubits), 'Number of qubits is incompatible with pauli string'
    assert (set(pauli_str).issubset({'I', 'X', 'Y', 'Z'})), 'pauliword must only contain X,Y,Z,I terms'

    char_aray = np.array(list(pauli_str), dtype=str)
    X_loc = (char_aray == 'X')
    Z_loc = (char_aray == 'Z')
    Y_loc = (char_aray == 'Y')

    symp_vec = np.zeros(2*n_qubits, dtype=int)
    symp_vec[:n_qubits] += X_loc
    symp_vec[n_qubits:] += Z_loc
    symp_vec[:n_qubits] += Y_loc
    symp_vec[n_qubits:] += Y_loc

    return symp_vec


def count1_in_int_bitstring(i):
    """
    Count number of "1" bits in integer i to be thought of in binary representation

    https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer#109025
    https://web.archive.org/web/20151229003112/http://blogs.msdn.com/b/jeuge/archive/2005/06/08/hakmem-bit-count.aspx
    """
    i = i - ((i >> 1) & 0x55555555)  # add pairs of bits
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)  # quads
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


def symplectic_to_sparse_matrix(symp_vec, coeff) -> csr_matrix:
    """
    Returns (2**n x 2**n) matrix of paulioperator kronector product together
     defined from symplectic vector defined as (X | Z)

    Args:
        symp_vec (array): symplectic Pauliword array

    Returns:
        sparse_matrix (csr_matrix): sparse matrix of Pauliword

    """
    n_qubits = len(symp_vec) // 2

    X_block = symp_vec[:n_qubits]
    Z_block = symp_vec[n_qubits:]

    Y_number = sum(np.bitwise_and(X_block, Z_block).astype(int))
    global_phase = (-1j) ** Y_number

    # reverse order to match bitstring int valu of each bit in binary: [..., 32, 16, 8, 4, 2, 1]
    binary_int_array = 1 << np.arange(n_qubits-1, -1, -1)

    x_int = X_block @ binary_int_array
    z_int = Z_block @ binary_int_array

    dimension = 2**n_qubits

    row_ind = np.arange(dimension)
    col_ind = np.bitwise_xor(row_ind, x_int)

    row_inds_and_Zint = np.bitwise_and(row_ind, z_int)
    vals = global_phase * (-1) ** (count1_in_int_bitstring(row_inds_and_Zint)%2)

    sparse_matrix = csr_matrix(
        (vals, (row_ind, col_ind)),
        shape=(dimension, dimension),
        dtype=complex
            )

    return coeff*sparse_matrix


class PauliwordOp:
    """ 
    A class thats represents an operator defined over the Pauli group in the symplectic representation.
    """
    def __init__(self, 
            operator:   Union[List[str], Dict[str, float], np.array], 
            coeff_list: Union[List[complex], np.array] = None
        ) -> None:
        """ 
        PauliwordOp may be initialized from either a dictionary in the form {pauli:coeff, ...}, 
        a list of Pauli strings or in the symplectic representation. In the latter two cases a 
        supplementary list of coefficients is also required, whereas this is inherent within the 
        dictionary representation. Operating on the level of the symplectic matrix is fastest 
        since it circumvents various conversions required - this is how the methods defined 
        below function.
        """

        if isinstance(operator, np.ndarray):
            if len(operator.shape)==1:
                operator = operator.reshape([1, len(operator)])
            self.symp_matrix = operator
            self.n_qubits = self.symp_matrix.shape[1]//2
        else:
            if isinstance(operator, dict):
                operator, coeff_list = zip(*operator.items())
                operator = list(operator)
            if isinstance(operator, list):
                self._init_from_paulistring_list(operator)
            else:
                raise ValueError(f'unkown operator type: must be dict or np.array: {type(operator)}')
        
        assert(coeff_list is not None), 'A list of coefficients has not been supplied'
        self.coeff_vec = np.asarray(coeff_list, dtype=complex)
        self.n_terms = self.symp_matrix.shape[0]
        assert(self.n_terms==len(self.coeff_vec)), 'coeff list and Pauliwords not same length'
        assert(set(np.unique(self.symp_matrix)).issubset({0,1})), 'symplectic array not defined with 0 and 1 only'
        self.X_block = self.symp_matrix[:, :self.n_qubits]
        self.Z_block = self.symp_matrix[:, self.n_qubits:]
        
    def _init_from_paulistring_list(self, 
            operator_list: List[str]
        ) -> None:
        """
        """
        n_rows = len(operator_list)
        self.n_qubits = len(operator_list[0])

        self.symp_matrix = np.zeros((n_rows, 2 * self.n_qubits), dtype=int)
        for row_ind, pauli_str in enumerate(operator_list):
            self.symp_matrix[row_ind] = string_to_symplectic(pauli_str, self.n_qubits)

    def __str__(self) -> str:
        """ 
        Defines the print behaviour of PauliwordOp - 
        returns the operator in an easily readable format

        Returns:
            out_string (str): human-readable PauliwordOp string
        """
        out_string = ''
        for pauli_vec, ceoff in zip(self.symp_matrix, self.coeff_vec):
            p_string = symplectic_to_string(pauli_vec)
            out_string += (f'{ceoff} {p_string} +\n')
        return out_string[:-3]

    def copy(self) -> "PauliwordOp":
        """ 
        Create a carbon copy of the class instance
        """
        return deepcopy(self)

    def basis_reconstruction(self, 
            operator_basis: "PauliwordOp"
        ) -> np.array:
        """ simultaneously reconstruct every operator term in the supplied basis.
        Performs Gaussian elimination on [op_basis.T | self_symp_csc.T] and restricts 
        so that the row-reduced identity block is removed. Each row of the
        resulting matrix will index the basis elements required to reconstruct
        the corresponding term in the operator.
        """
        dim = operator_basis.n_terms
        basis_symp_csc = operator_basis.symp_matrix
        basis_op_stack = np.vstack([basis_symp_csc, self.symp_matrix])
        op_reconstruction = gf2_gaus_elim(basis_op_stack.T)[:dim,dim:].T

        return op_reconstruction

    @cached_property
    def Y_count(self) -> np.array:
        """ 
        Count the qubit positions of each term set to Pauli Y

        cached_property means this only runs once and then is stored
        as self.Y_count

        Returns:
            numpy array of Y counts over terms of PauliwordOp
        """
        # Y_coords = self.X_block + self.Z_block == 2
        Y_coords = np.bitwise_and(self.X_block, self.Z_block)
        return np.array(Y_coords.sum(axis=1))

    def _multiply_single_Pword_phaseless(self,
            Pword:"PauliwordOp"
        ) -> np.array:
        """ performs *phaseless* Pauli multiplication via binary summation 
        of the symplectic matrix. Phase requires additional operations that
        are computed in _multiply_single_Pword.
        """
        pauli_mult_phaseless = np.bitwise_xor(self.symp_matrix, Pword.symp_matrix)
        return PauliwordOp(pauli_mult_phaseless, np.ones(self.n_terms))
    
    def _multiply_single_Pword(self, 
            Pword:"PauliwordOp"
        ) -> "PauliwordOp":
        """ performs Pauli multiplication with phases. The phase compensation 
        is implemented as per https://doi.org/10.1103/PhysRevA.68.042318
        """
        phaseless_prod_Pword = self._multiply_single_Pword_phaseless(Pword)

        # counts ZX mismatches for sign flip
        assert(Pword.n_terms==1), 'not single Pauliword'
        num_sign_flips = np.sum(np.bitwise_and(self.X_block, Pword.Z_block),
                               axis=1)
        sign_change = (-1) ** num_sign_flips

        # mapping from sigma to tau representation
        full_Y_count = self.Y_count + Pword.Y_count
        sigma_tau_compensation = (-1j) ** full_Y_count

        # back from tau to sigma (note uses output Pword)
        tau_sigma_compensation = (1j) ** phaseless_prod_Pword.Y_count

        # the full phase modification
        phase_mod = sign_change * sigma_tau_compensation * tau_sigma_compensation
        new_coeff_vec = phase_mod * self.coeff_vec * Pword.coeff_vec

        return PauliwordOp(phaseless_prod_Pword.symp_matrix, new_coeff_vec)

    def cleanup(self) -> "PauliwordOp":
        """ Remove duplicated rows of symplectic matrix terms, whilst summing
        the corresponding coefficients of the deleted rows in coeff
        """
        # convert sym form to list of ints
        int_list = self.symp_matrix @ (1 << np.arange(self.symp_matrix.shape[1])[::-1])
        re_order_indices = np.argsort(int_list)
        sorted_int_list = int_list[re_order_indices]

        sorted_symp_matrix = self.symp_matrix[re_order_indices]
        sorted_coeff_vec = self.coeff_vec[re_order_indices]

        # determine the first indices of each element in the sorted list (and ignore duplicates)
        elements, indices = np.unique(sorted_int_list, return_counts=True)
        row_summing = np.append([0], np.cumsum(indices))[:-1]  # [0, index1, index2,...]

        # reduced_symplectic_matrix = np.add.reduceat(sorted_symp_matrix, row_summing, axis=0)
        reduced_symplectic_matrix = sorted_symp_matrix[row_summing]
        reduced_coeff_vec = np.add.reduceat(sorted_coeff_vec, row_summing, axis=0)

        return PauliwordOp(reduced_symplectic_matrix, reduced_coeff_vec)

    def cleanup_zeros(self):
        """ 
        Delete terms with zero coefficient - this is not included in the cleanup method
        as one may wish to allow zero coefficients (e.g. as an Ansatz parameter angle)
        """
        clean_operator = self.cleanup()
        mask_nonzero = np.where(abs(clean_operator.coeff_vec)>1e-15)
        return PauliwordOp(clean_operator.symp_matrix[mask_nonzero], 
                            clean_operator.coeff_vec[mask_nonzero])

    def __add__(self, 
            Pword: "PauliwordOp"
        ) -> "PauliwordOp":
        """ Add to this PauliwordOp another PauliwordOp by stacking the
        respective symplectic matrices and cleaning any resulting duplicates
        """
        assert (self.n_qubits == Pword.n_qubits), 'Pauliwords defined for different number of qubits'
        P_symp_mat_new = np.vstack((self.symp_matrix, Pword.symp_matrix))
        P_new_coeffs = np.hstack((self.coeff_vec, Pword.coeff_vec)) 

        # cleanup run to remove duplicate rows (Pauliwords)
        return PauliwordOp(P_symp_mat_new, P_new_coeffs).cleanup()

    def __sub__(self,
            Pword: "PauliwordOp"
        ) -> "PauliwordOp":
        """ Subtract from this PauliwordOp another PauliwordOp 
        by negating the coefficients and summing
        """     
        op_copy = Pword.copy()
        op_copy.coeff_vec*=-1
        
        return self+op_copy

    def __mul__(self, 
            Pword: "PauliwordOp"
        ) -> "PauliwordOp":
        """ Right-multiplication of this PauliwordOp by another PauliwordOp.
        The phaseless multiplication is achieved via binary summation of the
        symplectic matrix in _multiply_single_Pword_phaseless whilst the phase
        compensation is introduced in _multiply_single_Pword.
        """
        assert (self.n_qubits == Pword.n_qubits), 'Pauliwords defined for different number of qubits'
        P_updated_list =[]
        for Pvec_single,coeff_single in zip(Pword.symp_matrix,Pword.coeff_vec):
            Pword_temp = PauliwordOp(Pvec_single, [coeff_single])
            P_new = self._multiply_single_Pword(Pword_temp)
            P_updated_list.append(P_new)

        P_final = reduce(lambda x,y: x+y, P_updated_list)
        return P_final

    def __getitem__(self, key: Union[slice, int]) -> "PauliwordOp":
        """ Makes the PauliwordOp subscriptable - returns a PauliwordOp constructed
        from the indexed row and coefficient from the symplectic matrix 
        """
        if isinstance(key, int):
            if key<0:
                # allow negative subscript
                key+=self.n_terms
            assert(key<self.n_terms), 'Index out of range'
            symp_index = self.symp_matrix[key]
            coef_index = self.coeff_vec[key]
            return PauliwordOp(symp_index, [coef_index])
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start=0
            if stop is None:
                stop=self.n_terms
            mask = np.arange(start, stop, key.step)
            symp_index = self.symp_matrix[mask]
            coef_index = self.coeff_vec[mask]
            return PauliwordOp(symp_index, coef_index)

    def __iter__(self):
        """ Makes a PauliwordOp instance iterable
        """
        return iter([self[i] for i in range(self.n_terms)])

    def multiply_by_constant(self, 
            const: complex
        ) -> "PauliwordOp":
        """
        Multiply the PauliwordOp by a complex coefficient
        """
        return PauliwordOp(self.symp_matrix, self.coeff_vec*const)

    def commutes_termwise(self, 
            Pword: "PauliwordOp"
        ) -> np.array:
        """ Outputs an array in which rows correspond with terms of the internal PauliwordOp (self)
        and colummns of Pword - True where terms commute and False if anticommutes

        **example
        op1 = PauliwordOp(['XYXZ', 'YYII'], [1,1])
        op2 = PauliwordOp(['YYZZ', 'XIXZ', 'XZZI'], [1,1,1])
        op1.commutes_termwise(op2)
        >> array([ 
                [ True,  True,  True],
                [ True, False,  True]]
                )
        """
        assert (self.n_qubits == Pword.n_qubits), 'Pauliwords defined for different number of qubits'
        Omega_Pword_symp = np.hstack((Pword.Z_block,  Pword.X_block)).T
        return (self.symp_matrix @ Omega_Pword_symp) % 2 == 0

    def commutes(self, 
            Pword: "PauliwordOp"
        ) -> bool:
        """ Checks if every term of self commutes with every term of Pword
        """
        return np.all(self.commutes_termwise(Pword))
    
    @cached_property
    def adjacency_matrix(self):
        """ Checks which terms of self commute within itself
        """
        return self.commutes_termwise(self)

    def _rotate_by_single_Pword(self, 
            Pword: "PauliwordOp", 
            angle: float = None
        ) -> "PauliwordOp":
        """ 
        Let R(t) = e^{i t/2 Q} = cos(t/2)*I + i*sin(t/2)*Q, then one of the following can occur:
        R(t) P R^\dag(t) = P when [P,Q] = 0
        R(t) P R^\dag(t) = cos(t) P + sin(t) (-iPQ) when {P,Q} = 0

        This operation is Clifford when t=pi/2, since cos(pi/2) P - sin(pi/2) iPQ = -iPQ.
        For t!=pi/2 an increase in the number of terms can be observed (non-Clifford unitary).
        
        <!> Please note the definition of the angle in R(t)...
            different implementations could be out by a factor of 2!
        """
        assert(Pword.n_terms==1), 'Only rotation by single Pauliword allowed here'
        if Pword.coeff_vec[0] != 1:
            # non-1 coefficients will affect the sign and angle in the exponent of R(t)
            # imaginary coefficients result in non-unitary R(t)
            Pword_copy = Pword.copy()
            Pword_copy.coeff_vec[0] = 1
            warnings.warn(f'Pword coefficient {Pword.coeff_vec[0]: .8f} has been set to 1')
        else:
            Pword_copy = Pword

        commute_vec = self.commutes_termwise(Pword_copy).flatten()
        commute_symp = self.symp_matrix[commute_vec]
        commute_coeff = self.coeff_vec[commute_vec]
        # ~commute_vec == not commutes, this indexes the anticommuting terms
        anticommute_symp = self.symp_matrix[~commute_vec]
        anticommute_coeff = self.coeff_vec[~commute_vec]

        commute_self = PauliwordOp(commute_symp, commute_coeff)
        anticom_self = PauliwordOp(anticommute_symp, anticommute_coeff)

        if angle is None:
            # assumes pi/2 rotation so Clifford
            anticom_part = (anticom_self*Pword_copy).multiply_by_constant(-1j)
        else:
            # if angle is specified, performs non-Clifford rotation
            anticom_part = (anticom_self.multiply_by_constant(np.cos(angle)) + 
                            (anticom_self*Pword_copy).multiply_by_constant(-1j*np.sin(angle)))
        
        return commute_self + anticom_part

    def recursive_rotate_by_Pword(self, 
            pauli_rot_list: Union[List[str], List[np.array]], 
            angles: List[float] = None
        ) -> "PauliwordOp":
        """ 
        Performs single Pauli rotations recursively left-to-right given a list of paulis supplied 
        either as strings or in the symplectic representation. This method does not allow coefficients 
        to be specified as rotation in this setting is ill-defined.

        If no angles are given then rotations are assumed to be pi/2 (Clifford)
        """
        if isinstance(pauli_rot_list[0], str):
            pauli_rot_list = [string_to_symplectic(r, self.n_qubits) for r in pauli_rot_list]
        if angles is None:
            angles = [None for t in range(len(pauli_rot_list))]
        assert(len(angles) == len(pauli_rot_list)), 'Mismatch between number of angles and number of Pauli terms'
        P_rotating = self.copy()
        for pauli_single,angle in zip(pauli_rot_list, angles):#.symp_matrix,Pword.coeff_vec,angles):
            Pword_temp = PauliwordOp(pauli_single, [1]) # enforcing coefficient to be 1, see above
            P_rotating = P_rotating._rotate_by_single_Pword(Pword_temp, angle).cleanup()
        return P_rotating

    @cached_property
    def PauliwordOp_to_OF(self) -> List[QubitOperator]:
        """ TODO Interface with converter.py (replace with to_dictionary method)
        """
        OF_list = []
        for Pvec_single, coeff_single in zip(self.symp_matrix, self.coeff_vec):
            P_string = symplectic_to_string(Pvec_single)
            OF_string = ' '.join([Pi+str(i) for i,Pi in enumerate(P_string) if Pi!='I'])
            OF_list.append(QubitOperator(OF_string, coeff_single))
        return OF_list

    @cached_property
    def to_dictionary(self) -> Dict[str, complex]:
        """
        Method for converting the operator from the symplectic representation 
        to a dictionary of the form {P_string:coeff, ...}
        """
        # clean the operator since duplicated terms will be overwritten in the conversion to a dictionary
        op_to_convert = self.cleanup()
        out_dict = {symplectic_to_string(symp_vec):coeff for symp_vec, coeff 
                    in zip(op_to_convert.symp_matrix, op_to_convert.coeff_vec)}
        return out_dict

    @cached_property
    def to_sparse_matrix(self) -> csr_matrix:
        """
        Function to get (2**n, 2**n) matrix of operator acting in Hilbert space

        """
        out_matrix = csr_matrix( ([],([],[])),
                                  shape=(2**self.n_qubits,2**self.n_qubits)
                                  )
        for Pvec_single, coeff_single in zip(self.symp_matrix, self.coeff_vec):
            out_matrix += symplectic_to_sparse_matrix(Pvec_single, coeff_single)

        return out_matrix

    def qwc_single_Pword(self,
            Pword: "PauliwordOp"
        ) -> bool:
        """ Checks self qubit wise commute (QWC) with another single Pauliword
        """
        assert (self.n_terms == 1), 'self operator must be a single Pauliword'
        assert (Pword.n_terms == 1), 'Pword must be a single Pauliword'

        # NOT identity locations (used for mask)
        self_I = np.bitwise_or(self.X_block, self.Z_block).astype(bool)
        Pword_I = np.bitwise_or(Pword.X_block, Pword.Z_block).astype(bool)

        # Get the positions where neither self nor Pword have I acting on them
        unique_non_I_locations = np.bitwise_and(self_I, Pword_I)

        # check non I operators are the same!
        same_Xs = np.bitwise_not(
            np.bitwise_xor(self.X_block[unique_non_I_locations], Pword.X_block[unique_non_I_locations]).astype(
                bool))
        same_Zs = np.bitwise_not(
            np.bitwise_xor(self.Z_block[unique_non_I_locations], Pword.Z_block[unique_non_I_locations]).astype(
                bool))

        if np.all(same_Xs) and np.all(same_Zs):
            return True
        else:
            return False


class StabilizerOp(PauliwordOp):
    """ Special case of PauliwordOp, in which the operator terms must
    by algebraically independent, with all coefficients set to one.
    """
    def __init__(self,
                 operator:   Union[List[str], Dict[str, float], np.array],
                 coeff_list: Union[List[complex], np.array] = None):
        """
        """
        super().__init__(operator, coeff_list)
        self._check_stab()
        self._check_independent()

    def _check_stab(self):
        """ Checks the stabilizer coefficients are +1
        """
        assert(np.all(self.coeff_vec==1)), 'Stabilizer coefficients not +1'

    def _check_independent(self):
        """ Check the supplied stabilizers are algebraically independent
        """
        check_independent = gf2_gaus_elim(self.symp_matrix)
        for row in check_independent:
            if np.all(row==0):
                # there is a dependent row
                raise ValueError('The supplied stabilizers are not independent')


def convert_openF_fermionic_op_to_maj_op(fermionic_op):
    """
    Convserion as:
        a_{p} = 0.5*(γ_{2p} + iγ_{2p+1})
        a†_{p} = 0.5*(γ_{2p} - iγ_{2p+1})
     note goes from N to 2N sites!

    # uses inbuilt functions in OpenFermion and maps to symplectic form

    """
    if not isinstance(fermionic_op, FermionOperator):
        raise ValueError('not an openfermion Fermionic operator')

    N_sites = count_qubits(fermionic_op)
    maj_operator = get_majorana_operator(fermionic_op)

    N_terms = len(maj_operator.terms)
    majorana = np.zeros((N_terms, 2 * N_sites))
    coeffs = np.zeros(N_terms, dtype=complex)
    for ind, term_coeff in enumerate(maj_operator.terms.items()):
        majorana[ind, term_coeff[0]] = 1
        coeffs[ind] = term_coeff[1]

    op_out = MajoranaOp(majorana, coeffs).cleanup()

    #     if op_out.to_OF_op() != get_majorana_operator(fermionic_op):
    #         # check using openF == comparison
    #         raise ValueError('op not converted correctly')

    return op_out


def bubble_sort_maj(array):

    arr = np.asarray(array)
    n_sites = arr.shape[0]
    sign_dict = {0: +1, 1:-1}
    # Traverse through all array elements
    swap_counter = 0
    for i in range(n_sites):
        swapped = False
        for j in range(0, n_sites - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                swap_counter+=1

        if swapped == False:
            break

    return arr.tolist(), sign_dict[swap_counter%2]

class MajoranaOp:
    """
    A class thats represents an operator defined as Majorana fermionic operators (stored in a symplectic representation).

    Note Majorana operators follow the following definition:

    { γ_{j} , γ_{k} } = 2δ_{jk}I (aka same as the Pauli group!)

    """

    def __init__(self,
                 list_lists_OR_sympletic_form,
                 coeff_list
                 ) -> None:
        """
        TODO: need to fix ordering in init! ( aka if one defines [[12,10]] then we get y_10 y_12 (but order change here must change sign!)
        """
        self.coeff_vec = np.asarray(coeff_list, dtype=complex)
        self.initalize_op(list_lists_OR_sympletic_form)
        self.term_index_list = np.arange(0, self.n_sites, 1)
        self.n_terms = self.symp_matrix.shape[0]

    def initalize_op(self, input_term):

        if isinstance(input_term, np.ndarray):
            if (len(input_term)==0)and (len(self.coeff_vec)==1):
                self.n_sites = 1
                self.symp_matrix = np.array([[0]], dtype=int)
            else:
                self.n_sites = input_term.shape[1]
                self.symp_matrix = input_term
        else:
            flat_list = set(item for sublist in input_term for item in sublist)
            if flat_list:
                self.n_sites = max(flat_list) + 1
            else:
                self.n_sites = 1
            n_terms = len(input_term)
            self.symp_matrix = np.zeros((n_terms, self.n_sites), dtype=int)
            for ind, term in enumerate(input_term):
                ordered_term, sign = bubble_sort_maj(term)
                self.symp_matrix[ind, ordered_term] = 1
                self.coeff_vec[ind] *= sign

        assert (self.symp_matrix.shape[0] == len(self.coeff_vec)), 'incorrect number of coefficients'

    def __str__(self) -> str:
        """
        Defines the print behaviour of MajoranaOp -
        returns the operator in an easily readable format

        Returns:
            out_string (str): human-readable MajoranaOp string
        """
        out_string = ''
        for majorana_vec, ceoff in zip(self.symp_matrix, self.coeff_vec):
            maj_inds = self.term_index_list[majorana_vec.astype(bool)]
            maj_string = ' '.join([f'γ{ind}' for ind in maj_inds])
            if maj_string == '':
                maj_string = 'I'

            out_string += (f'{ceoff} {maj_string} +\n')
        return out_string[:-3]

    def commutes(self, M_OP):

        termwise_commutes = self.commutes_termwise(M_OP)
        unique_terms = np.unique(termwise_commutes.flatten())  # equiv of set operation

        return np.all(unique_terms == 0)

    def commutes_termwise(self, M_OP):
        # 1 means terms anticommute!
        # 0 means terms commute!
        # { γA, γB } = [1 + (−1)|A||B|+|A∩B|]γAγB
        # https://arxiv.org/pdf/2101.09349.pdf (eq 9)
        if self.n_sites != M_OP.n_sites:
            sites = min(self.n_sites, M_OP.n_sites)
        else:
            sites = self.n_sites

        suppA = np.einsum('ij->i', self.symp_matrix)
        suppB = np.einsum('ij->i', M_OP.symp_matrix)
        AtimeB = np.outer(suppA, suppB)

        # only look over common inds
        AandB = np.dot(self.symp_matrix[:, :sites], M_OP.symp_matrix[:, :sites].T)
        comm_flag = (AtimeB + AandB) % 2

        return comm_flag

    def adjacency_matrix(self):
        """ Checks which terms of self commute within itself
        """
        adj = self.commutes_termwise(self)
        return adj

    def copy(self) -> "MajoranaOp":
        """
        Create a carbon copy of the class instance
        """
        return deepcopy(self)

    def __add__(self,
                M_OP: "MajoranaOp"
                ) -> "MajoranaOp":
        """ Add to this PauliwordOp another PauliwordOp by stacking the
        respective symplectic matrices and cleaning any resulting duplicates
        """
        if self.n_sites != M_OP.n_sites:
            if self.n_sites < M_OP.n_sites:
                temp_mat = np.zeros((self.n_terms, M_OP.n_sites))
                temp_mat[:, :self.n_sites] += self.symp_matrix
                P_symp_mat_new = np.vstack((temp_mat, M_OP.symp_matrix))
            else:
                temp_mat = np.zeros((M_OP.n_terms, self.n_sites))
                temp_mat[:, :M_OP.n_sites] += M_OP.symp_matrix
                P_symp_mat_new = np.vstack((self.symp_matrix, temp_mat))
        else:
            P_symp_mat_new = np.vstack((self.symp_matrix, M_OP.symp_matrix))

        P_new_coeffs = np.hstack((self.coeff_vec, M_OP.coeff_vec))

        # cleanup run to remove duplicate rows (Pauliwords)
        return MajoranaOp(P_symp_mat_new, P_new_coeffs).cleanup()

    def cleanup(self) -> "MajoranaOp":
        """ Remove duplicated rows of symplectic matrix terms, whilst summing
        the corresponding coefficients of the deleted rows in coeff
        """
        # convert sym form to list of ints
        int_list = self.symp_matrix @ (1 << np.arange(self.symp_matrix.shape[1])[::-1])
        re_order_indices = np.argsort(int_list)
        sorted_int_list = int_list[re_order_indices]

        sorted_symp_matrix = self.symp_matrix[re_order_indices]
        sorted_coeff_vec = self.coeff_vec[re_order_indices]

        # determine the first indices of each element in the sorted list (and ignore duplicates)
        elements, indices = np.unique(sorted_int_list, return_counts=True)
        row_summing = np.append([0], np.cumsum(indices))[:-1]  # [0, index1, index2,...]

        # reduced_symplectic_matrix = np.add.reduceat(sorted_symp_matrix, row_summing, axis=0)
        reduced_symplectic_matrix = sorted_symp_matrix[row_summing]
        reduced_coeff_vec = np.add.reduceat(sorted_coeff_vec, row_summing, axis=0)

        # return nonzero coeff terms!
        mask_nonzero = np.where(abs(reduced_coeff_vec) > 1e-15)
        return MajoranaOp(reduced_symplectic_matrix[mask_nonzero],
                          reduced_coeff_vec[mask_nonzero])

    def to_OF_op(self):
        open_f_op = MajoranaOperator()
        for majorana_vec, ceoff in zip(self.symp_matrix, self.coeff_vec):
            maj_inds = self.term_index_list[majorana_vec.astype(bool)]

            open_f_op += MajoranaOperator(term=tuple(maj_inds),
                                          coefficient=ceoff)
        return open_f_op

    def __mul__(self,
                M_OP: "MajoranaOp"
                ) -> "MajoranaOp":
        """
        Right-multiplication of this MajoranaOp by another MajoranaOp
        """
        if self.n_sites != M_OP.n_sites:
            if self.n_sites < M_OP.n_sites:
                temp_mat_self = np.zeros((self.n_terms, M_OP.n_sites))
                temp_mat_self[:, :self.n_sites] += self.symp_matrix
                temp_mat_M_OP = M_OP.symp_matrix
            else:
                temp_mat_M_OP = np.zeros((M_OP.n_terms, self.n_sites))
                temp_mat_M_OP[:, :M_OP.n_sites] += M_OP.symp_matrix
                temp_mat_self = self.symp_matrix
        else:
            temp_mat_M_OP = M_OP.symp_matrix
            temp_mat_self = self.symp_matrix

        new_vec = np.zeros((self.n_terms * M_OP.n_terms, max(temp_mat_M_OP.shape[1],
                                                             temp_mat_self.shape[1])
                            ))

        new_coeff_vec = np.outer(self.coeff_vec, M_OP.coeff_vec).flatten()
        sign_dict = {0: 1, 1: -1}
        ind = 0
        for ind1, vec in enumerate(temp_mat_self):
            for ind2, vec2 in enumerate(temp_mat_M_OP):
                new_vec[ind] = np.logical_xor(vec, vec2).astype(int)

                # track changes to make operator in normal order
                #                 reordering_sign = sum(sum(vec[i+1:]) for i, term in enumerate(vec2[:-1]) if term!=0)%2
                reordering_sign = sum(term * (sum(vec[i + 1:])) for i, term in enumerate(vec2[:-1])) % 2
                new_coeff_vec[ind] *= sign_dict[reordering_sign]
                ind += 1

        return MajoranaOp(new_vec, new_coeff_vec).cleanup()


class QubitHamiltonian(PauliwordOp):
    """
    Qubit Hamiltonian made up as a linear combination of Pauliwords

    """
    def __init__(self,
            operator: Union[List[str], Dict[str, float], np.array],
            coeff_list=None):
        super().__init__(operator, coeff_list)

        self.eig_vals = None
        self.eig_vecs = None
    def Get_ground_state(self, n_eig_vals=1):
        sparse_hamiltonian_mat = self.to_sparse_matrix
        H_mat_shape = sparse_hamiltonian_mat.shape[0]
        assert (n_eig_vals<=H_mat_shape), 'cannot have more eigenvalues than dimension of matrix'
        if H_mat_shape<=64:
            # dense eigh
            eig_values, eig_vectors = np.linalg.eigh(sparse_hamiltonian_mat.todense())
        else:
            # sparse eigh
            eig_values, eig_vectors = sp.sparse.linalg.eigsh(sparse_hamiltonian_mat,
                                                     k=n_eig_vals,
                                                     # v0=initial_guess,
                                                     which='SA',
                                                     maxiter=1e7)

        order = np.argsort(eig_values)
        self.eig_vals = eig_values[order]
        self.eig_vecs = eig_vectors[:, order].T