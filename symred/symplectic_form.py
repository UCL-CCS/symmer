# general imports
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from functools import reduce
from itertools import product
from sympy import Q
from psutil import net_if_stats
from cached_property import cached_property
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import scipy as sp
import warnings
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
warnings.simplefilter('always', UserWarning)

# specialized imports
from symred.utils import gf2_gaus_elim, norm, ZX_calculus_reduction
from openfermion import (
    QubitOperator, 
    MajoranaOperator, 
    get_sparse_operator, 
    FermionOperator, 
    count_qubits, 
    get_majorana_operator, 
    #get_fermion_operator
)
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow import CircuitStateFn
from qiskit import BasicAer, execute


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

    This follows because tensor products of Pauli operators are one-sparse: they each have only
    one nonzero entry in each row and column

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
    binary_int_array = 1 << np.arange(n_qubits-1, -1, -1, dtype=object)

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
            coeff_vec: Union[List[complex], np.array] = None
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
                operator, coeff_vec = zip(*operator.items())
                operator = list(operator)
            if isinstance(operator, list):
                self._init_from_paulistring_list(operator)
            else:
                raise ValueError(f'unkown operator type: must be dict or np.array: {type(operator)}')
        
        assert(coeff_vec is not None), 'A list of coefficients has not been supplied'
        self.coeff_vec = np.asarray(coeff_vec, dtype=complex)
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
        # only display single significant figure if StabilizerOp object
        if isinstance(self, StabilizerOp):
            sigfig = '.0f'
        else:
            sigfig = '.10f'

        out_string = ''
        for pauli_vec, coeff in zip(self.symp_matrix, self.coeff_vec):
            p_string = symplectic_to_string(pauli_vec)
            out_string += (f'{format(coeff, sigfig)} {p_string} +\n')
        return out_string[:-3]

    def copy(self) -> "PauliwordOp":
        """ 
        Create a carbon copy of the class instance
        """
        return deepcopy(self)

    def sort(self, by='decreasing', key='magnitude') -> "PauliwordOp":
        """
        Sort the terms by some key, either magnitude, weight X, Y or Z
        """
        if key=='magnitude':
            sort_order = np.argsort(-abs(self.coeff_vec))
        elif key=='weight':
            sort_order = np.argsort(-np.einsum('ij->i', self.symp_matrix))
        elif key=='Z':
            sort_order = np.argsort(np.einsum('ij->i', (self.n_qubits+1)*self.X_block + self.Z_block))
        elif key=='X':
            sort_order = np.argsort(np.einsum('ij->i', self.X_block + (self.n_qubits+1)*self.Z_block))
        elif key=='Y':
            sort_order = np.argsort(np.einsum('ij->i', abs(self.X_block - self.Z_block)))
        else:
            raise ValueError('Only permitted sort key values are magnitude, weight, X, Y or Z')
        if by=='increasing':
            sort_order = sort_order[::-1]
        elif by!='decreasing':
            raise ValueError('Only permitted sort by values are increasing or decreasing')
        return PauliwordOp(self.symp_matrix[sort_order], self.coeff_vec[sort_order])

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

    def cleanup_zeros(self, zero_threshold=1e-15):
        """ 
        Delete terms with zero coefficient - this is not included in the cleanup method
        as one may wish to allow zero coefficients (e.g. as an Ansatz parameter angle)
        """
        clean_operator = self.cleanup()
        mask_nonzero = np.where(abs(clean_operator.coeff_vec)>zero_threshold)
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
            mul_obj: Union["PauliwordOp", "QuantumState"]
        ) -> "PauliwordOp":
        """ Right-multiplication of this PauliwordOp by another PauliwordOp.
        The phaseless multiplication is achieved via binary summation of the
        symplectic matrix in _multiply_single_Pword_phaseless whilst the phase
        compensation is introduced in _multiply_single_Pword.
        """
        if isinstance(mul_obj, QuantumState):
            assert(mul_obj.vec_type == 'ket'), 'cannot multiply a bra from the left'
            Pword = mul_obj.state_op
        else:
            Pword = mul_obj
        assert (self.n_qubits == Pword.n_qubits), 'Pauliwords defined for different number of qubits'
        P_updated_list =[]
        for Pvec_single,coeff_single in zip(Pword.symp_matrix,Pword.coeff_vec):
            Pword_temp = PauliwordOp(Pvec_single, [coeff_single])
            P_new = self._multiply_single_Pword(Pword_temp)
            P_updated_list.append(P_new)

        P_final = reduce(lambda x,y: x+y, P_updated_list)

        if isinstance(mul_obj, QuantumState):
            coeff_vector = P_final.coeff_vec*(1j**P_final.Y_count)
            # need to run a separate cleanup since identities are all mapped to Z 
            # i.e. ZZZZ==IIII in QuantumState
            return QuantumState(P_final.X_block, coeff_vector).cleanup()
        else:
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

    def commutator(self, Pword: "PauliwordOp") -> "PauliwordOp":
        """ Computes the commutator [A, B] = AB - BA
        """
        return (self * Pword - Pword * self).cleanup_zeros()

    def anticommutator(self, Pword: "PauliwordOp") -> "PauliwordOp":
        """ Computes the anticommutator {A, B} = AB + BA
        """
        return (self * Pword + Pword * self).cleanup_zeros()

    def commutes(self, 
            Pword: "PauliwordOp"
        ) -> bool:
        """ Checks if every term of self commutes with every term of Pword
        """
        return self.commutator(Pword).n_terms == 0
    
    @cached_property
    def adjacency_matrix(self):
        """ Checks which terms of self commute within itself
        """
        return self.commutes_termwise(self)

    @cached_property
    def is_noncontextual(self):
        """ Returns True if the operator is noncontextual, False if contextual
        Scales as O(N^2), compared with the O(N^3) algorithm of https://doi.org/10.1103/PhysRevLett.123.200501
        """
        # mask the terms that do not commute universally amongst the operator
        mask_non_universal = np.where(np.any(~self.adjacency_matrix, axis=1))[0]
        # look only at the unique rows in the masked adjacency matrix -
        # identical rows correspond with operators of the same clique
        unique_commutation_character = np.unique(
            np.array(
                self.adjacency_matrix[mask_non_universal,:][:,mask_non_universal], 
                dtype=int
                ), 
            axis=0
        )
        # if the unique commutation characteristics are disjoint, i.e. no overlapping ones 
        # between rows, the operator is noncontextual - hence we sum over rows and check
        # the resulting vector consists of all ones.
        return np.all(np.einsum('ij->j', unique_commutation_character)==1)

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
            rotations: List[Tuple[str, float]]
        ) -> "PauliwordOp":
        """ 
        Performs single Pauli rotations recursively left-to-right given a list of paulis supplied 
        either as strings or in the symplectic representation. This method does not allow coefficients 
        to be specified as rotation in this setting is ill-defined.

        If no angles are given then rotations are assumed to be pi/2 (Clifford)
        """
        op_copy = self.copy()
        for pauli_rotation,angle in rotations:
            symp_rotation = string_to_symplectic(pauli_rotation, self.n_qubits)
            Pword_temp = PauliwordOp(symp_rotation, [1]) # enforcing coefficient to be 1, see above
            op_copy = op_copy._rotate_by_single_Pword(Pword_temp, angle).cleanup()
        return op_copy

    @cached_property
    def conjugate(self) -> "PauliwordOp":
        """
        Returns:
            Pword_conj (PauliwordOp): The Hermitian conjugated operator
        """
        Pword_conj = PauliwordOp(
            operator  = self.symp_matrix, 
            coeff_vec = self.coeff_vec.conjugate()
        )
        return Pword_conj

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


class ObservableGraph(PauliwordOp):
    # TODO
    def __init__(self,
            operator:   Union[List[str], Dict[str, float], np.array],
            coeff_list: Union[List[complex], np.array] = None):
        super().__init__(operator, coeff_list)

    def build_graph(self, edge_relation='C', weighted=None):

        if edge_relation =='AC':
            # commuting edges
            adjacency_mat = np.bitwise_not(self.adjacency_matrix)

            # removes self adjacency for graph
            np.fill_diagonal(adjacency_mat, 0)

        elif edge_relation =='C':
            # anticommuting edges
            adjacency_mat = self.adjacency_matrix
            np.fill_diagonal(adjacency_mat, 0)

        elif edge_relation =='QWC':
            adjacency_mat = np.zeros((self.n_terms, self.n_terms))
            for i in range(self.n_terms):
                for j in range(i+1, self.n_terms):
                    Pword_i = self.symp_matrix[i]
                    Pword_j = self.symp_matrix[j]

                    self_I = np.bitwise_or(Pword_i[:self.n_qubits], Pword_i[self.n_qubits:]).astype(bool)
                    Pword_I = np.bitwise_or(Pword_j[:self.n_qubits], Pword_j[self.n_qubits:]).astype(bool)

                    # Get the positions where neither self nor Pword have I acting on them
                    unique_non_I_locations = np.bitwise_and(self_I, Pword_I)

                    # check non I operators are the same!
                    same_Xs = np.bitwise_not(
                        np.bitwise_xor(Pword_i[:self.n_qubits][unique_non_I_locations],
                                       Pword_j[:self.n_qubits][unique_non_I_locations]).astype(
                            bool))
                    same_Zs = np.bitwise_not(
                        np.bitwise_xor(Pword_i[self.n_qubits:][unique_non_I_locations],
                                       Pword_j[self.n_qubits:][unique_non_I_locations]).astype(
                            bool))

                    if np.all(same_Xs) and np.all(same_Zs):
                        adjacency_mat[i,j] = adjacency_mat[j,i] = 1
                    else:
                        continue
        else:
            raise ValueError(f'unknown edge relation: {edge_relation}')

        graph = nx.from_numpy_matrix(adjacency_mat)

        return graph

    def clique_cover(self, clique_relation, colouring_strategy, colour_interchange=False,
                     plot_graph=False, with_node_label=False, node_sizes=True):

        if clique_relation == 'AC':
            graph = self.build_graph(edge_relation='C')
        elif clique_relation == 'C':
            graph = self.build_graph(edge_relation='AC')
        elif clique_relation == 'QWC':
            graph = self.build_graph(edge_relation='QWC')
            graph = nx.complement(graph)
        else:
            raise ValueError(f'unknown clique relation: {clique_relation}')

        # keys give symplectic row index and value gives colour of clique
        greedy_colouring_output_dic = nx.greedy_color(graph,
                                                      strategy=colouring_strategy,
                                                      interchange=colour_interchange)

        unique_colours = set(greedy_colouring_output_dic.values())

        clique_dict = {}
        for Clique_ind in unique_colours:
            clique_Pword_symp = []
            clique_coeff_symp = []
            for sym_row_ind, clique_id in greedy_colouring_output_dic.items():
                if clique_id == Clique_ind:
                    clique_Pword_symp.append(self.symp_matrix[sym_row_ind,:])
                    clique_coeff_symp.append(self.coeff_vec[sym_row_ind])

            clique = PauliwordOp(np.array(clique_Pword_symp, dtype=int),
                                 clique_coeff_symp)

            clique_dict[Clique_ind] = clique

        if plot_graph:
            possilbe_colours = cm.rainbow(np.linspace(0, 1, len(unique_colours)))
            colour_list = [possilbe_colours[greedy_colouring_output_dic[node_id]] for node_id in graph.nodes()]
            # print(colour_list)
            # print([symplectic_to_string(self.symp_matrix[row_ind]) for row_ind in graph.nodes])
            self.draw_graph(graph, with_node_label=with_node_label, node_sizes=node_sizes, node_colours=colour_list)

        return clique_dict

    def draw_graph(self, graph_input, with_node_label=False, node_sizes=True, node_colours=None):

        if node_sizes:
            node_sizes = 200 * np.abs(np.round(self.coeff_vec)) + 1
            options = {
                'node_size': node_sizes,
                'node_color': 'r' if node_colours is None else node_colours
                     }
        else:
            options = {
                'node_color': 'r' if node_colours is None else node_colours
                     }

        plt.figure()
        pos = nx.circular_layout(graph_input)
        # # pos = nx.spring_layout(graph_input)
        # pos = nx.nx_pydot.graphviz_layout(graph_input)

        nx.draw_networkx_nodes(graph_input,
                               pos,
                               nodelist=list(graph_input.nodes),
                               **options)
        nx.draw_networkx_edges(graph_input, pos,
                               width=1.0,
                               # alpha=0.5,
                               nodelist=list(graph_input.nodes),
                               )

        if with_node_label:
            labels = {row_ind: symplectic_to_string(self.symp_matrix[row_ind]) for row_ind in graph_input.nodes}
            nx.draw_networkx_labels(graph_input,
                                    pos,
                                    labels,
                                    font_size=18)

        # plt.savefig('G_raw', dpi=300, transparent=True, )  # edgecolor='black', facecolor='white')
        plt.show()
        return None


class AnsatzOp(PauliwordOp):
    """ Based on PauliwordOp and introduces functionality for converting operators to quantum circuits
    """
    def __init__(self,
            operator:   Union[List[str], Dict[str, float], np.array],
            coeff_vec: Union[List[complex], np.array] = None
        ) -> None:
        """
        """
        super().__init__(operator, coeff_vec)
        assert(np.all(self.coeff_vec.imag==0)), 'Coefficients must have zero imaginary component'
        self.coeff_vec = self.coeff_vec.real

    def exponentiate(self):
        """
        Returns:
            exp_T (PauliwordOp): exponentiated form of the ansatz operator
        """
        exp_bin = []
        for term, angle in zip(self.symp_matrix, self.coeff_vec):
            exp_bin.append(
                PauliwordOp(
                    np.vstack([np.zeros_like(term), term]), 
                    [np.cos(angle), 1j*np.sin(angle)]
                )
            )
        exp_T = reduce(lambda x,y:x*y, exp_bin)

        return exp_T

    @cached_property
    def to_instructions(self) -> Dict[int, Dict[str, List[int]]]:
        """ Stores a dictionary of gate instructions at each step, where each value
        is a dictionary indicating the indices on which to apply each H,S,CNOT and RZ gate
        """
        circuit_instructions = {}
        for step, (X,Z) in enumerate(zip(self.X_block, self.Z_block)):
            # locations for H and S gates to transform into Pauli Z basis
            H_indices = np.where(X)[0][::-1]
            S_indices = np.where(X & Z)[0][::-1]
            # CNOT cascade indices
            CNOT_indices = np.where(X | Z)[0][::-1]
            circuit_instructions[step] = {'H_indices':H_indices, 
                                        'S_indices':S_indices, 
                                        'CNOT_indices':CNOT_indices,
                                        'RZ_index':CNOT_indices[-1]}
        return circuit_instructions

    def to_QuantumCircuit(self, 
        ref_state: np.array = None,
        basis_change_indices: Dict[str, List[int]] = {'X_indices':[],'Y_indices':[]},
        trotter_number: int = 1, 
        bind_params: bool = True,
        ZX_reduction = False
        ) -> str:
        """
        Convert the operator to a QASM circuit string for input 
        into quantum computing packages such as Qiskit and Cirq

        basis_change_indices in form [X_indices, Y_indices]
        """
        def qiskit_ordering(indices):
            """ we index from left to right - in Qiskit this ordering is reversed
            """
            return self.n_qubits - 1 - indices

        qc = QuantumCircuit(self.n_qubits)
        for i in qiskit_ordering(np.where(ref_state==1)[0]):
            qc.x(i)

        def CNOT_cascade(cascade_indices, reverse=False):
            index_pairs = list(zip(cascade_indices[:-1], cascade_indices[1:]))
            if reverse:
                index_pairs = index_pairs[::-1]
            for source, target in index_pairs:
                qc.cx(source, target)

        def circuit_from_step(angle, H_indices, S_indices, CNOT_indices, RZ_index):
            # to Pauli X basis
            for i in S_indices:
                qc.sdg(i)
            # to Pauli Z basis
            for i in H_indices:
                qc.h(i)
            # compute parity
            CNOT_cascade(CNOT_indices)
            qc.rz(-2*angle, RZ_index)
            CNOT_cascade(CNOT_indices, reverse=True)
            for i in H_indices:
                qc.h(i)
            for i in S_indices:
                qc.s(i)

        if bind_params:
            angles = self.coeff_vec.real/trotter_number
        else:
            angles = np.array(ParameterVector('P', self.n_terms))/trotter_number

        assert(len(angles)==len(self.to_instructions)), 'Number of parameters does not match the circuit instructions'
        for trot_step in range(trotter_number):
            for step, gate_indices in self.to_instructions.items():
                qiskit_gate_indices = [qiskit_ordering(indices) for indices in gate_indices.values()]
                qc.barrier()
                circuit_from_step(angles[step], *qiskit_gate_indices)

        qc.barrier()
        for i in basis_change_indices['Y_indices']:
            qc.s(qiskit_ordering(i))
        for i in basis_change_indices['X_indices']:
            qc.h(qiskit_ordering(i))

        if ZX_reduction:
            qc = ZX_calculus_reduction(qc)

        return qc


class ObservableOp(PauliwordOp):
    """ Based on PauliwordOp and introduces functionality for evaluating expectation values
    """
    # here we define the expectation evaluation parameters and can be updated by the user
    evaluation_method = 'statevector' # one of statevector, trotter_rotations or sampled
    trotter_number = 1 # the number of repetition in the QuantumCircuit
    n_shots = 2**10 # number of samples taken from each QWC group circuit
    n_realizations = 1 # number of expectation evaluations in average for sampled method
    backend = BasicAer.get_backend('qasm_simulator')

    def __init__(self,
        operator:   Union[List[str], Dict[str, float], np.array],
        coeff_vec: Union[List[complex], np.array] = None
        ) -> None:
        """
        """
        super().__init__(operator, coeff_vec)
        assert(np.all(self.coeff_vec.imag==0)), 'Coefficients must be real, ensuring the operator is Hermitian'
        self.coeff_vec = self.coeff_vec.real

    def Z_basis_expectation(self, ref_state: np.array) -> float:
        """ Provided a single Pauli-Z basis state, computes the expectation value of the operator
        """
        assert(set(ref_state).issubset({0,1})), f'Basis state must consist of binary elements, not {set(ref_state)}'
        assert(len(ref_state)==self.n_qubits), f'Number of qubits {len(ref_state)} in the basis state incompatible with {self.n_qubits}'
        mask_diagonal = np.where(np.all(self.X_block==0, axis=1))
        measurement_signs = (-1)**np.einsum('ij->i', self.Z_block[mask_diagonal] & ref_state)
        return np.sum(measurement_signs * self.coeff_vec[mask_diagonal]).real

    def _ansatz_expectation_trotter_rotations(self, 
            ansatz_op: AnsatzOp, 
            ref_state: np.array
        ):
        """ Exact expectation value - expensive! Trotterizes the ansatz operator and applies the terms as
        Pauli rotations to the observable operator, resulting in an exponential increase in the number of terms
        """
        pauli_rotations = [symplectic_to_string(row) for row in ansatz_op.symp_matrix]*self.trotter_number
        angles = -2*np.tile(ansatz_op.coeff_vec, self.trotter_number)/self.trotter_number

        trotterized_observable = self.recursive_rotate_by_Pword(zip(pauli_rotations[::-1], angles[::-1]))
        trotterized_observable = ObservableOp(trotterized_observable.symp_matrix, trotterized_observable.coeff_vec)

        return trotterized_observable.Z_basis_expectation(ref_state)

    def _ansatz_expectation_statevector(self, 
            ansatz_op: AnsatzOp, 
            ref_state: np.array,
            sparse = False
        ) -> float:
        """ Exact expectation value - expensive! Converts the ansatz operator to a sparse vector | psi >
        and return the quantity < psi | Observable | psi >
        """
        if sparse:
            # sparse multiplication does not scale nicely with number of qubits
            ansatz_qc = ansatz_op.to_QuantumCircuit(ref_state=ref_state, trotter_number=self.trotter_number)
            psi = CircuitStateFn(ansatz_qc).to_spmatrix()
            return (psi @ self.to_sparse_matrix @ psi.T)[0,0].real
        else:
            psi = ansatz_op.exponentiate() * QuantumState([ref_state])
            return (psi.conjugate * self * psi).real

    @cached_property
    def QWC_decomposition(self):
        """
        """
        HamGraph = ObservableGraph(self.symp_matrix, self.coeff_vec)
        QWC_operators = HamGraph.clique_cover(clique_relation='QWC', colouring_strategy='largest_first')
        # check the QWC groups sum to the original observable operator
        reconstructed = reduce(lambda x,y:x+y, QWC_operators.values())
        assert(
            np.all((self-reconstructed).coeff_vec == 0)
        ), 'Summing QWC group operators does not yield the original Hamiltonian'
        
        return QWC_operators

    def _ansatz_expectation_sampled(self, 
            ansatz_op: AnsatzOp, 
            ref_state: np.array
        ) -> float:
        """ Evaluates epectation values by quantum circuit sampling. Decomposes the ObservableOp into
        qubitwise commuting (QWC) components that may be measured simultaneously by transforming onto 
        the Pauli Z basis. This allows one to reconstruct the ansatz operator in a tomography-esque
        fashion and determine the expectation value w.r.t. the observable operator.

        Returns:
            expectation (float): the expectation value of the ObseravleOp w.r.t. the given ansatz and reference state
        """
        #start = time.time()
        QWC_group_data = {}
        for group_index, group_operator in self.QWC_decomposition.items():
            # find the qubit positions containing X's or Y's in the QWC group
            X_indices = np.where(np.einsum('ij->j', group_operator.X_block)!=0)[0]
            Y_indices = np.where(np.einsum('ij->j', group_operator.X_block & group_operator.Z_block)!=0)[0]
            basis_change_indices={'X_indices':X_indices, 'Y_indices':Y_indices}
            # we make a change of basis from Pauli X/Y --> Z, allowing us to take measurements in the Z basis
            group_qc = ansatz_op.to_QuantumCircuit(
                ref_state = ref_state,
                trotter_number=self.trotter_number,
                basis_change_indices=basis_change_indices, 
                ZX_reduction=False
            )
            group_qc.measure_all()
            QWC_group_data[group_index] = {
                "operator":PauliwordOp(group_operator.symp_matrix, group_operator.coeff_vec),
                "circuit":group_qc,
                "measured_state":None
            }
        #stop = time.time()
        #print('QWC group data', stop - start)
        #start = time.time()
        # send all the QWC group circuits off to be executed via the backend
        QWC_group_circuits = [group['circuit'] for group in QWC_group_data.values()]
        job = execute(QWC_group_circuits, self.backend, shots=self.n_shots)
        #stop = time.time()
        #print('Obtain measurements', stop - start)
        #start = time.time()
        expectation = 0

        for i in range(len(QWC_group_circuits)):
            # reconstruct ansatz state from measurement outcomes (tomography-esque)
            states_hex, frequency = zip(*job.result().data(i)['counts'].items())
            state_matrix = [
                [int(i) for i in np.binary_repr(int(state, 16), self.n_qubits)]
                for state in states_hex
            ]
            state = QuantumState(state_matrix, np.array(frequency)/self.n_shots)
            QWC_group_data[i]['measured_state'] = state
            observable = QWC_group_data[i]['operator']
            # modify the Z block in line with the change of basis
            Z_symp_matrix = observable.X_block | observable.Z_block
            for b_state, b_state_coeff in zip(state.state_matrix, state.coeff_vector):
                expectation += np.sum((-1)**np.count_nonzero(Z_symp_matrix & b_state, axis=1)*observable.coeff_vec*b_state_coeff)
        #stop = time.time()
        #print('Evaluate expectation value', stop - start)

        return expectation.real

    def ansatz_expectation(self, 
            ansatz_op: AnsatzOp, 
            ref_state: np.array, 
        ) -> float:
        """ 
        """
        if self.evaluation_method == 'statevector':
            return self._ansatz_expectation_statevector(ansatz_op, ref_state)
        elif self.evaluation_method == 'trotter_rotations':
            return self._ansatz_expectation_trotter_rotations(ansatz_op, ref_state)
        elif self.evaluation_method == 'sampled':
            return np.mean([self._ansatz_expectation_sampled(ansatz_op, ref_state) for i in range(self.n_realizations)])
        else:
            raise ValueError('Invalid evaluation method, must be one of statevector, trotter_rotations or sampled')

    def parameter_shift_at_index(self,
        param_index: int,
        ansatz_op: AnsatzOp, 
        ref_state: np.array
        ):
        """ return the first-order derivative at x w.r.t. to the observable operator
        """
        assert(param_index<ansatz_op.n_terms), 'Indexing outside ansatz parameters'
        
        anz_upper_param = ansatz_op.copy()
        anz_lower_param = ansatz_op.copy()
        anz_upper_param.coeff_vec[param_index] += np.pi/4
        anz_lower_param.coeff_vec[param_index] -= np.pi/4 
        
        shift_upper = self.ansatz_expectation(anz_upper_param, ref_state=ref_state)
        shift_lower = self.ansatz_expectation(anz_lower_param, ref_state=ref_state)

        return shift_upper-shift_lower

    def gradient(self,
        ansatz_op: AnsatzOp,
        ref_state: np.array
        ) -> np.array:
        """
        """
        return np.array(
            [self.parameter_shift_at_index(i, ansatz_op, ref_state) 
                for i in range(ansatz_op.n_terms)]
        )

    def VQE(self,
        ansatz_op:  AnsatzOp,
        ref_state:  np.array,
        optimizer:  str   = 'SLSQP',
        maxiter:    int   = 10, 
        opt_tol:    float = None
        ):
        """ 
        Rcommended optimizers:
            - SLSQP  (gradient-descent, does not evaluate Jacobian at each iterate like BFGS or CG so is faster)
            - COBYLA (gradient-free)
        """
        interim_values = {'values':[], 'params':[], 'gradients':[], 'count':0}

        init_params = ansatz_op.coeff_vec

        def fun(x):
            interim_values['count']+=1
            ansatz_op.coeff_vec = x
            energy = self.ansatz_expectation(ansatz_op, ref_state)
            interim_values['params'].append((interim_values['count'], x))
            interim_values['values'].append((interim_values['count'], energy))
            return energy

        def jac(x):
            ansatz_op.coeff_vec = x
            grad = self.gradient(ansatz_op, ref_state)
            interim_values['gradients'].append((interim_values['count'], grad))
            return grad

        vqe_result = minimize(
            fun=fun, 
            jac=jac,
            x0=init_params,
            method=optimizer,
            tol=opt_tol,
            options={'maxiter':maxiter}
        )

        return vqe_result, interim_values


class StabilizerOp(PauliwordOp):
    """ Special case of PauliwordOp, in which the operator terms must
    by algebraically independent, with all coefficients set to integers +/-1.

    - stabilizer_rotations
        This method determines a sequence of Clifford rotations mapping the
        provided stabilizers onto single-qubit Paulis (sqp), either X or Z
    """
    def __init__(self,
            operator:   Union[List[str], Dict[str, float], np.array],
            coeff_vec: Union[List[complex], np.array] = None,
            target_sqp: str = 'X'):
        """
        """
        super().__init__(operator, coeff_vec)
        self._check_stab()
        self._check_independent()
        if target_sqp in ['X', 'Z']:
            self.target_sqp = target_sqp
        elif target_sqp == 'Y':
            raise NotImplementedError('Currently only accepts X or Z and target single-qubit Pauli')
        else:
            raise ValueError('Target single-qubit Pauli not recognised - must be X or Z')

    def _check_stab(self):
        """ Checks the stabilizer coefficients are +/-1
        """
        assert(set(self.coeff_vec).issubset({+1,-1})), f'Stabilizer coefficients not +/-1: {self.coeff_vec}'

    def _check_independent(self):
        """ Check the supplied stabilizers are algebraically independent
        """
        check_independent = gf2_gaus_elim(self.symp_matrix)
        for row in check_independent:
            if np.all(row==0):
                # there is a dependent row
                raise ValueError('The supplied stabilizers are not independent')

    @cached_property
    def stabilizer_rotations(self) -> Tuple[List[str], List[Union[None,float]]]:
        """ 
        Implementation of procedure described in https://doi.org/10.22331/q-2021-05-14-456 (Lemma A.2)
        
        Returns 
            - a list of Pauli rotations in the form List[str]
            - a list of rotation angles in the form List[float]
        """
        rotations=[]
        
        def update_sets(base_vector, pivot_index):
            """ 
            - ammend the X_, Z_block positions at pivot_index to 1
                (corresponds with fixing the pivot_index qubit to Pauli Y)
            - append the rotation to the rotations list
            - update used_indices with the fixed qubit position.
            - also returns the Pauli rotation so it may be applied in _recursive_rotate_onto_sqp
            """
            pivot_index_X = pivot_index % self.n_qubits # index in the X block
            base_vector[np.array([pivot_index_X, pivot_index_X+self.n_qubits])]=1

            rotations.append((symplectic_to_string(base_vector), None))
            used_indices.append(pivot_index_X)
            used_indices.append(pivot_index_X + self.n_qubits)
            
            return PauliwordOp(base_vector, [1])

        def _recursive_rotate_onto_sqp(basis: StabilizerOp):
            """ recursively generates Clifford operations mapping the input basis 
            onto single-qubit Pauli operators. Works in order of increasing term
            weight (i.e. the number of non-identity positions)
            """
            if basis is None:
                return None
            else:
                row_sum = np.einsum('ij->i',basis.symp_matrix)
                col_sum = np.einsum('ij->j',basis.symp_matrix)
                sort_rows_by_weight = np.argsort(row_sum)
                pivot_row = basis.symp_matrix[sort_rows_by_weight][0]
                non_I = np.setdiff1d(np.where(pivot_row)[0], np.array(used_indices))
                support = pivot_row*col_sum
                pivot_point = non_I[np.argmin(support[non_I])]
                pivot_rotation = update_sets(pivot_row.copy(), pivot_point)
                rotated_basis = basis._rotate_by_single_Pword(pivot_rotation)
                non_sqp = np.where(np.einsum('ij->i', rotated_basis.symp_matrix)!=1)[0].tolist()
                try:
                    new_basis = reduce(lambda x,y:x+y, [rotated_basis[i] for i in non_sqp])
                except:
                    new_basis = None
                return _recursive_rotate_onto_sqp(new_basis)

        # identify any basis elements that already single-qubit Paulis 
        row_sum = np.einsum('ij->i',self.symp_matrix)
        sqp_indices = np.where(self.symp_matrix[np.where(row_sum==1)])[1]
        sqp_X_block = sqp_indices % self.n_qubits
        used_indices = list(np.concatenate([sqp_X_block, sqp_X_block+self.n_qubits]))
        # find rotations for the non-single-qubit Pauli terms
        non_sqp_basis = StabilizerOp(self.symp_matrix[np.where(row_sum!=1)],
                                    self.coeff_vec[np.where(row_sum!=1)])
        if non_sqp_basis.n_terms != 0:
            # i.e. the operator does not already consist of single-qubit Paulis
            _recursive_rotate_onto_sqp(non_sqp_basis)
            rotated_op = self.recursive_rotate_by_Pword(rotations)
        else:
            rotated_op = self

        # This part produces rotations onto the target sqp
        for row in rotated_op.symp_matrix:
            sqp_index = np.where(row)[0]
            if ((self.target_sqp == 'Z' and sqp_index< self.n_qubits) or 
                (self.target_sqp == 'X' and sqp_index>=self.n_qubits)):
                update_sets(np.zeros(2*self.n_qubits, dtype=int), sqp_index)

        return rotations

    def update_sector(self, 
            ref_state: Union[List[int], np.array]
        ) -> None:
        """ Given the specified reference state, e.g. Hartree-Fock |1...10...0>, 
        determine the corresponding sector by measuring the stabilizers

        TODO: currently only measures in Z basis
            only supports single basis vector reference - should accept a linear combination
        """
        ref_state = np.array(ref_state)
        self.coeff_vec = (-1)**np.count_nonzero(self.Z_block & ref_state, axis=1)

    def rotate_onto_single_qubit_paulis(self) -> "StabilizerOp":
        """ Returns the rotated single-qubit Pauli stabilizers
        """
        if self.stabilizer_rotations != []:
            rotated_stabilizers = self.recursive_rotate_by_Pword(self.stabilizer_rotations)
        else:
            rotated_stabilizers = self
        return StabilizerOp(
            rotated_stabilizers.symp_matrix,
            rotated_stabilizers.coeff_vec
        )


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
    """

    Given an array/list of majorana modes use bubble sort alg to reorder by size and keep track of sign

    e.g. given
    [12,10] then we get y_10 y_12 (but order change here must generate a negative sign!)
     y_12 y_10 ==  -1*(y_10 y_12)

    Args:
        array (list): list of ints

    Returns:
        arr (list): sorted list of ints
        sign (int): +1 or -1 sign generated by normal ordering

    """

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

class QuantumState:
    """ Class to represent quantum states.
    
    This is achieved by identifying the state with a 
    state_op (PauliwordOp), namely |0> --> Z, |1> --> X. 
    
    For example, the 2-qubit Bell state is mapped as follows: 
        1/sqrt(2) (|00> + |11>) --> 1/sqrt(2) (ZZ + XX)
    Observe the state is recovered by applying the state_op to the 
    zero vector |00>, which will be the X_block of state_op.
    
    This ensures correct phases when multiplying the quantum state by a PauliwordOp.
    """
    def __init__(self, 
            state_matrix: Union[List[List[int]], np.array], 
            coeff_vector: Union[List[complex], np.array] = None,
            vec_type: str = 'ket'
        ) -> None:
        """ The state is not normalized by default, since this would result
        in incorrect behaviour when perfoming non-unitary multiplications,
        e.g. for evaluating expectation values of Hamiltonians. However, if
        one wishes to normalize the state, it is stored as a cached propoerty
        as QuantumState.normalize.
        """
        if isinstance(state_matrix, list):
            state_matrix = np.array(state_matrix)
        if isinstance(coeff_vector, list):
            coeff_vector = np.array(coeff_vector)
        assert(set(state_matrix.flatten()).issubset({0,1})) # must be binary, does not support N-ary qubits
        self.n_terms, self.n_qubits = state_matrix.shape
        self.state_matrix = state_matrix
        if coeff_vector is None:
            # if no coefficients specified produces a uniform superposition
            self.coeff_vector = np.ones(self.n_terms)/np.sqrt(self.n_terms)
        else:
            self.coeff_vector = coeff_vector
        self.vec_type = vec_type
        # the quantum state is manipulated via the state_op PauliwordOp
        symp_matrix = np.hstack([state_matrix, 1-state_matrix])
        self.state_op = PauliwordOp(symp_matrix, self.coeff_vector)

    def copy(self) -> "QuantumState":
        """ 
        Create a carbon copy of the class instance
        """
        return deepcopy(self)

    def __str__(self) -> str:
        """ 
        Defines the print behaviour of QuantumState - differs depending on vec_type

        Returns:
            out_string (str): human-readable QuantumState string
        """
        out_string = ''
        for basis_vec, coeff in zip(self.state_matrix, self.coeff_vector):
            basis_string = ''.join([str(i) for i in basis_vec])
            if self.vec_type == 'ket':
                out_string += (f'{coeff: .10f} |{basis_string}> +\n')
            elif self.vec_type == 'bra':
                out_string += (f'{coeff: .10f} <{basis_string}| +\n')
            else:
                raise ValueError('Invalid vec_type, must be bra or ket')
        return out_string[:-3]
    
    def __add__(self, 
            Qstate: "QuantumState"
        ) -> "QuantumState":
        """ Add to this QuantumState another QuantumState by summing 
        the respective state_op (PauliwordOp representing the state)
        """
        new_state = self.state_op + Qstate.state_op
        return QuantumState(new_state.X_block, new_state.coeff_vec)
    
    def __sub__(self, 
            Qstate: "QuantumState"
        ) -> "QuantumState":
        """ Subtract from this QuantumState another QuantumState by subtracting 
        the respective state_op (PauliwordOp representing the state)
        """
        new_state_op = self.state_op - Qstate.state_op
        return QuantumState(new_state_op.X_block, new_state_op.coeff_vec)
    
    def __mul__(self,
        mul_obj: Union["QuantumState", PauliwordOp]
        ) -> Union["QuantumState", complex]:
        """
        Right multiplication of a bra QuantumState by either a ket QuantumState or PauliwordOp
        
        Returns:
            - inner_product (complex): when mul_obj is a ket state
            - new_bra_state (QuantumState): when mul_obj is a PauliwordOp
        """
        assert(self.n_qubits == mul_obj.n_qubits), 'Multiplication object defined for different number of qubits'
        assert(self.vec_type=='bra'), 'Cannot multiply a ket from the right'
        
        if isinstance(mul_obj, QuantumState):
            assert(mul_obj.vec_type=='ket'), 'Cannot multiply a bra with another bra'
            inner_product=0
            for (bra_string, bra_coeff),(ket_string, ket_coeff) in product(
                    zip(self.state_matrix, self.coeff_vector), 
                    zip(mul_obj.state_matrix, mul_obj.coeff_vector)
                ):
                if np.all(bra_string == ket_string):
                    inner_product += (bra_coeff*ket_coeff)
            return inner_product

        elif isinstance(mul_obj, PauliwordOp):
            new_state_op = self.state_op * mul_obj
            new_state_op.coeff_vec*=((-1j)**new_state_op.Y_count)
            new_bra_state = QuantumState(
                new_state_op.X_block, 
                new_state_op.coeff_vec, 
                vec_type='bra'
            )
            return new_bra_state.cleanup()

        else:
            raise ValueError('Trying to multiply QuantumState by unrecognised object - must be another Quantum state or PauliwordOp')   

    def cleanup(self, zero_threshold=1e-15) -> "QuantumState":
        """ Combines duplicate basis states, summing their coefficients
        """
        clean_state_op = self.state_op.cleanup_zeros(zero_threshold=zero_threshold)
        return QuantumState(
            clean_state_op.X_block, 
            clean_state_op.coeff_vec, 
            vec_type=self.vec_type
        )

    @cached_property
    def normalize(self):
        """
        Returns:
            self (QuantumState)
        """
        coeff_vector = self.coeff_vector/norm(self.coeff_vector)
        return QuantumState(self.state_matrix, coeff_vector)
        
    @cached_property
    def conjugate(self) -> "QuantumState":
        """
        Returns:
            conj_state (QuantumState): The Hermitian conjugated state i.e. bra -> ket, ket -> bra
        """
        if self.vec_type == 'ket':
            new_type = 'bra'
        else:
            new_type = 'ket'
        conj_state = QuantumState(
            state_matrix = self.state_matrix, 
            coeff_vector = self.coeff_vector.conjugate(),
            vec_type     = new_type
        )
        return conj_state

    @cached_property
    def to_sparse_matrix(self):
        """
        Returns:
            sparse_Qstate (csr_matrix): sparse matrix representation of the statevector
        """
        nonzero_indices = [int(''.join([str(i) for i in row]),2) for row in self.state_matrix]
        sparse_Qstate = csr_matrix(
            (self.coeff_vector, (nonzero_indices, np.zeros_like(nonzero_indices))), 
            shape = (2**self.n_qubits, 1), 
            dtype=np.complex128
        )
        return sparse_Qstate

def array_to_QuantumState(statevector, threshold=1e-15):
    """ Given a vector of 2^N elements over N qubits, convert to a QuantumState object.
    
    Returns:
        Qstate (QuantumState): a QuantumState object representing the input vector
        
    **example
        statevector = array([0.57735027,0,0,0,0,0.81649658,0,0])
        print(array_to_QuantumState(statevector)) 
        >>  0.5773502692 |000> + 
            0.8164965809 |101>
    """
    N = np.log2(statevector.shape[0])
    assert(N-int(N) == 0), 'the statevector dimension is not a power of 2'
    N = int(N)
    non_zero = np.where(abs(statevector)>=threshold)[0]
    state_matrix = np.array([[int(i) for i in list(np.binary_repr(index,N))] for index in non_zero])
    coeff_vector = statevector[non_zero]
    Qstate = QuantumState(state_matrix, coeff_vector)
    return Qstate