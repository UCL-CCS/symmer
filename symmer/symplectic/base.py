import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import product
from functools import reduce
from typing import Dict, List, Tuple, Union
from numbers import Number
from cached_property import cached_property
from scipy.sparse import csr_matrix
from symmer.utils import *
from openfermion import QubitOperator
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliOp, PauliSumOp
import warnings
warnings.simplefilter('always', UserWarning)

class PauliwordOp:
    """ 
    A class thats represents an operator defined over the Pauli group in the symplectic representation.
    """
    sigfig = 3 # specifies the number of significant figures for printing
    
    def __init__(self, 
            symp_matrix: Union[List[str], Dict[str, float], np.array], 
            coeff_vec:   Union[List[complex], np.array]
        ) -> None:
        """ 
        PauliwordOp may be initialized from either a dictionary in the form {pauli:coeff, ...}, 
        a list of Pauli strings or in the symplectic representation. In the latter two cases a 
        supplementary list of coefficients is also required, whereas this is inherent within the 
        dictionary representation. Operating on the level of the symplectic matrix is fastest 
        since it circumvents various conversions required - this is how the methods defined 
        below function.
        """
        symp_matrix = np.array(symp_matrix)
        assert(symp_matrix.dtype == int), 'Symplectic matrix must be defined over integers'
        assert(set(np.unique(symp_matrix)).issubset({0,1})), 'symplectic matrix not defined with 0 and 1 only'
        if len(symp_matrix.shape)==1:
            symp_matrix = symp_matrix.reshape([1, len(symp_matrix)])
        self.symp_matrix = symp_matrix
        self.n_qubits = self.symp_matrix.shape[1]//2
        self.coeff_vec = np.asarray(coeff_vec, dtype=complex)
        self.n_terms = self.symp_matrix.shape[0]
        assert(self.n_terms==len(self.coeff_vec)), 'coeff list and Pauliwords not same length'
        self.X_block = self.symp_matrix[:, :self.n_qubits]
        self.Z_block = self.symp_matrix[:, self.n_qubits:]
        
    @classmethod
    def empty(cls, n_qubits):
        return cls.from_dictionary({'I'*n_qubits:0})

    @classmethod
    def from_dictionary(cls,
            operator_dict: Dict[str, complex]
        ) -> "PauliwordOp":
        """ Initialize a PauliwordOp from its dictionary representation {pauli:coeff, ...}
        """
        pauli_terms, coeff_vec = zip(*operator_dict.items())
        pauli_terms = list(pauli_terms)
        return cls.from_list(pauli_terms, coeff_vec)

    @classmethod
    def from_list(cls, 
            pauli_terms :List[str], 
            coeff_vec:   List[complex]
        ) -> "PauliwordOp":
        """ Initialize a PauliwordOp from its Pauli terms and coefficients stored as lists
        """
        n_rows = len(pauli_terms)
        if pauli_terms:
            n_qubits = len(pauli_terms[0])
            symp_matrix = np.zeros((n_rows, 2 * n_qubits), dtype=int)
            for row_ind, pauli_str in enumerate(pauli_terms):
                symp_matrix[row_ind] = string_to_symplectic(pauli_str, n_qubits)
        else:
            n_qubits = 0
            symp_matrix = np.array([[]], dtype=int)
        return cls(symp_matrix, coeff_vec)

    @classmethod
    def from_matrix(cls, 
            matrix: np.array, 
            operator_basis: "PauliwordOp" = None
        ) -> "PauliwordOp":
        """
        Given any arbitrary matrix convert it into a linear combination of Pauli operators (PauliwordOp)
        Note this decomposition builds a basis of size 4^{N_qubits} to decompose matrix into
        this can be costly! (TODO: may be better way of chosing a basis to represent operator in! aka smaller than 4^N
                                see jupyter notebook for further details)

        If user doesn't define an operator basis then builds the full 4^N Hilbert space.
        The user can avoid this by specificying a reduced basis that targets a subspace; there is a check 
        to assess whether the basis is sufficiently expressible to represent the input matrix in this case.
        """
        if isinstance(matrix, np.matrix):
            # summing over numpy matrices does not function correctly
            matrix = np.array(matrix)

        n_qubits = int(np.ceil(np.log2(max(matrix.shape))))
        if n_qubits > 64 and operator_basis is None:
            # could change XZ_block builder to use numpy objects (allows above 64-bit integers) but very slow and matrix will be too large to build 
            raise ValueError('Matrix too large! Integer to binary conversion not defined above 64-bits.')

        if not (2**n_qubits, 2**n_qubits) == matrix.shape:
            # padding matrix with zeros so correct size
            temp_mat = np.zeros((2 ** n_qubits, 2 ** n_qubits))
            temp_mat[:matrix.shape[0],
                :matrix.shape[1]] = matrix
            matrix = temp_mat

        if operator_basis is None:
            # fast method to build all binary assignments
            int_list = np.arange(4 ** (n_qubits))
            XZ_block = (((int_list[:, None] & (1 << np.arange(2 * n_qubits))[::-1])) > 0).astype(int)
            op_basis = cls(XZ_block, np.ones(XZ_block.shape[0]))
        else:
            op_basis = operator_basis

        denominator = 2 ** n_qubits
        decomposition = cls.empty(n_qubits)
        for op in op_basis:
            const = sum((op.to_sparse_matrix @ matrix).diagonal()) / denominator
            decomposition += op.multiply_by_constant(const)

        operator_out = decomposition.cleanup()
        if operator_basis is not None:
            if not np.all(operator_out.to_sparse_matrix.toarray() == matrix):
                raise ValueError('Basis not sufficiently expressive.')
        
        return operator_out
        
            
    def __str__(self) -> str:
        """ 
        Defines the print behaviour of PauliwordOp - 
        returns the operator in an easily readable format

        Returns:
            out_string (str): human-readable PauliwordOp string
        """
        if self.symp_matrix.shape[1]:
            out_string = ''
            for pauli_vec, coeff in zip(self.symp_matrix, self.coeff_vec):
                p_string = symplectic_to_string(pauli_vec)
                out_string += (f'{coeff: .{self.sigfig}f} {p_string} +\n')
            return out_string[:-3]
        else: 
            return f'{self.coeff_vec[0]: .{self.sigfig}f}'

    def __repr__(self):
        return str(self)

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

        Nonzero entries ocurring below the resulting identity block cannot be reconstructed
        in the supplied basis - index_successfully_reconstructed indicates those which succeeded
        """
        dim = operator_basis.n_terms
        basis_symp = operator_basis.symp_matrix
        basis_op_stack = np.vstack([basis_symp, self.symp_matrix])
        reduced = gf2_gaus_elim(basis_op_stack.T)

        index_successfully_reconstructed = np.where(
            np.einsum('ij->j', reduced[dim:,dim:])==0
        )[0]
        #if index_unsuccessful_reconstruction:
        #    warnings.warn(f'Terms {index_unsuccessful_reconstruction} cannot be reconstructed.')
        op_reconstruction = reduced[:dim,dim:].T

        return op_reconstruction, index_successfully_reconstructed

    @cached_property
    def Y_count(self) -> np.array:
        """ 
        Count the qubit positions of each term set to Pauli Y

        cached_property means this only runs once and then is stored
        as self.Y_count

        Returns:
            numpy array of Y counts over terms of PauliwordOp
        """
        return np.einsum('ij->i', np.bitwise_and(self.X_block, self.Z_block))

    def cleanup(self, zero_threshold=1e-15):
        """ Apply symplectic_cleanup and delete terms with negligible coefficients
        """
        if self.n_qubits == 0:
            clean_operator =  PauliwordOp([], [np.sum(self.coeff_vec)])
        elif self.n_terms == 0:
            clean_operator =  PauliwordOp(np.zeros((1, self.symp_matrix.shape[1]), dtype=int), [0])
        else:
            clean_operator = PauliwordOp(*symplectic_cleanup(self.symp_matrix, self.coeff_vec))
        mask_nonzero = np.where(abs(clean_operator.coeff_vec)>zero_threshold)
        return PauliwordOp(
            clean_operator.symp_matrix[mask_nonzero], 
            clean_operator.coeff_vec[mask_nonzero]
        )

    def __eq__(self, Pword: "PauliwordOp") -> bool:
        """ In theory should use logical XNOR to check symplectic matrix match, however
        can use standard logical XOR and look for False indices instead (implementation
        skips an additional NOT operation) 
        """
        check_1 = self.cleanup()
        check_2 = Pword.cleanup()
        if check_1.n_qubits != check_2.n_qubits:
            raise ValueError('Operators defined over differing numbers of qubits.')
        elif check_1.n_terms != check_2.n_terms:
            return False
        else:
            return (
                not np.einsum('ij->', np.logical_xor(check_1.symp_matrix, check_2.symp_matrix)) and 
                np.allclose(check_1.coeff_vec, check_2.coeff_vec)
            )

    def __add__(self, 
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
        """ Add to this PauliwordOp another PauliwordOp by stacking the
        respective symplectic matrices and cleaning any resulting duplicates
        """
        assert (self.n_qubits == PwordOp.n_qubits), 'Pauliwords defined for different number of qubits'
        P_symp_mat_new = np.vstack((self.symp_matrix, PwordOp.symp_matrix))
        P_new_coeffs = np.hstack((self.coeff_vec, PwordOp.coeff_vec)) 

        # cleanup run to remove duplicate rows (Pauliwords)
        return PauliwordOp(P_symp_mat_new, P_new_coeffs).cleanup()

    def __radd__(self,
        add_obj: Union[int, "PauliwordOp"]) -> "PauliwordOp":
        """ Allows use of sum() over a list of PauliwordOps
        """
        if add_obj == 0:
            return self
        else:
            return self + add_obj

    def __sub__(self,
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
        """ Subtract from this PauliwordOp another PauliwordOp 
        by negating the coefficients and summing
        """     
        op_copy = PwordOp.copy()
        op_copy.coeff_vec*=-1
        
        return self+op_copy

    def multiply_by_constant(self, 
            const: complex
        ) -> "PauliwordOp":
        """
        Multiply the PauliwordOp by a complex coefficient
        """
        return PauliwordOp(self.symp_matrix, self.coeff_vec*const)

    def _mul_symplectic(self, 
            symp_vec: np.array, 
            coeff: complex, 
            Y_count_in: np.array
        ) -> Tuple[np.array, np.array]:
        """ performs Pauli multiplication with phases at the level of the symplectic 
        matrices to avoid superfluous PauliwordOp initializations. The phase compensation 
        is implemented as per https://doi.org/10.1103/PhysRevA.68.042318.
        """
        # phaseless multiplication is binary addition in symplectic representation
        phaseless_prod = np.bitwise_xor(self.symp_matrix, symp_vec)
        # phase is determined by Y counts plus additional sign flip
        Y_count_out = np.einsum('ij->i', np.bitwise_and(*np.hsplit(phaseless_prod,2)))
        sign_change = (-1) ** np.einsum('ij->i', np.bitwise_and(self.X_block, np.hsplit(symp_vec,2)[1]))
        # final phase modification
        phase_mod = sign_change * (1j) ** (3*Y_count_in + Y_count_out)
        coeff_vec = phase_mod * self.coeff_vec * coeff
        return phaseless_prod, coeff_vec

    def __mul__(self, 
            mul_obj: Union["PauliwordOp", "QuantumState", complex]
        ) -> "PauliwordOp":
        """ Right-multiplication of this PauliwordOp by another PauliwordOp or QuantumState ket.
        """
        if isinstance(mul_obj, Number):
            return self.multiply_by_constant(mul_obj)

        if isinstance(mul_obj, QuantumState):
            # allows one to apply PauliwordOps to QuantumStates
            # (corresponds with multipcation of the underlying state_op)
            assert(mul_obj.vec_type == 'ket'), 'cannot multiply a bra from the left'
            PwordOp = mul_obj.state_op
        else:
            PwordOp = mul_obj
        assert (self.n_qubits == PwordOp.n_qubits), 'PauliwordOps defined for different number of qubits'

        # multiplication is performed at the symplectic level, before being stacked and cleaned
        symp_stack, coeff_stack = zip(
            *[self._mul_symplectic(symp_vec=symp_vec, coeff=coeff, Y_count_in=Y_count+self.Y_count) 
            for symp_vec, coeff, Y_count in zip(PwordOp.symp_matrix, PwordOp.coeff_vec, PwordOp.Y_count)]
        )
        pauli_mult_out = PauliwordOp(*symplectic_cleanup(np.vstack(symp_stack), np.hstack(coeff_stack)))
        
        if isinstance(mul_obj, QuantumState):
            coeff_vec = pauli_mult_out.coeff_vec*(1j**pauli_mult_out.Y_count)
            # need to run a separate cleanup since identities are all mapped to Z, i.e. II==ZZ in QuantumState
            return QuantumState(pauli_mult_out.X_block, coeff_vec).cleanup()
        else:
            return pauli_mult_out

    def __imul__(self, PwordOp: "PauliwordOp") -> "PauliwordOp":
        """ in-place multiplication behaviour
        """
        return self.__mul__(PwordOp)

    def __pow__(self, exponent:int) -> "PauliwordOp":
        assert(isinstance(exponent, int)), 'the exponent is not an integer'
        if exponent == 0:
            return PauliwordOp.from_list(['I'*self.n_qubits],[1])
        else:
            factors = [self.copy()]*exponent
            return reduce(lambda x,y:x*y, factors)

    def __getitem__(self, key: Union[slice, int]) -> "PauliwordOp":
        """ Makes the PauliwordOp subscriptable - returns a PauliwordOp constructed
        from the indexed row and coefficient from the symplectic matrix 
        """
        if isinstance(key, int):
            if key<0:
                # allow negative subscript
                key+=self.n_terms
            assert(key<self.n_terms), 'Index out of range'
            mask = [key]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start=0
            if stop is None:
                stop=self.n_terms
            mask = np.arange(start, stop, key.step)
        
        symp_items = self.symp_matrix[mask]
        coeff_items = self.coeff_vec[mask]
        return PauliwordOp(symp_items, coeff_items)

    def __iter__(self):
        """ Makes a PauliwordOp instance iterable
        """
        return iter([self[i] for i in range(self.n_terms)])

    def commutes_termwise(self, 
            PwordOp: "PauliwordOp"
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
        assert (self.n_qubits == PwordOp.n_qubits), 'Pauliwords defined for different number of qubits'
        Omega_PwordOp_symp = np.hstack((PwordOp.Z_block,  PwordOp.X_block)).T
        return (self.symp_matrix @ Omega_PwordOp_symp) % 2 == 0

    def qubitwise_commutes_termwise(self,
        PwordOp: "PauliwordOp"
        ) -> np.array:
        """ Given the symplectic representation of a single Pauli operator,
        determines which operator terms of the internal PauliwordOp qubitwise commute

        Returns:  
            QWC_matrix (np.array): 
                an array whose elements are True if the corresponding term
                qubitwise commutes with the input PwordOp
        """
        QWC_matrix = []
        for X_term, Z_term in zip(PwordOp.X_block, PwordOp.Z_block):
            # identify the qubit positions on which there is at least one non-identity operation
            non_I = (self.X_block | self.Z_block) & (X_term | Z_term)
            # identify matches between the operator and term of PwordOp - these indicate qubitwise commutation
            X_match = np.all((self.X_block & non_I) == (X_term & non_I), axis=1)
            Z_match = np.all((self.Z_block & non_I) == (Z_term & non_I), axis=1)
            # mask the terms of self.observable that qubitwise commute with the PwordOp term
            QWC_matrix.append((X_match & Z_match).reshape(self.n_terms, 1))
        return np.hstack(QWC_matrix)

    def commutator(self, PwordOp: "PauliwordOp") -> "PauliwordOp":
        """ Computes the commutator [A, B] = AB - BA
        """
        return self * PwordOp - PwordOp * self

    def anticommutator(self, PwordOp: "PauliwordOp") -> "PauliwordOp":
        """ Computes the anticommutator {A, B} = AB + BA
        """
        return self * PwordOp + PwordOp * self

    def commutes(self, 
            PwordOp: "PauliwordOp"
        ) -> bool:
        """ Checks if every term of self commutes with every term of PwordOp
        """
        return self.commutator(PwordOp).n_terms == 0
    
    @cached_property
    def adjacency_matrix(self):
        """ Checks which terms of self commute within itself
        """
        return self.commutes_termwise(self)

    @cached_property
    def adjacency_matrix_qwc(self):
        """ Checks which terms of self qubitwise commute within itself
        """
        return self.qubitwise_commutes_termwise(self)

    @cached_property
    def is_noncontextual(self):
        """ Returns True if the operator is noncontextual, False if contextual
        Scales as O(N^2), compared with the O(N^3) algorithm of https://doi.org/10.1103/PhysRevLett.123.200501
        Constructing the adjacency matrix is by far the most expensive part - very fast once that has been built.

        Note, the legacy utils.contextualQ function CAN be faster than this method when the input operator
        contains MANY triples that violate transitivity of commutation. However, if this is not the case - for
        example when the diagonal contribution dominates the operator - this method is significantly faster.
        """
        # mask the terms that do not commute universally amongst the operator
        mask_non_universal = np.where(~np.all(self.adjacency_matrix, axis=1))[0]
        # look only at the unique rows in the masked adjacency matrix -
        # identical rows correspond with operators of the same clique
        unique_commutation_character = np.unique(
            self.adjacency_matrix[mask_non_universal,:][:,mask_non_universal],
            axis=0
        )
        # if the unique commutation characteristics are disjoint, i.e. no overlapping ones 
        # between rows, the operator is noncontextual - hence we sum over rows and check
        # the resulting vector consists of all ones.
        return np.all(np.count_nonzero(unique_commutation_character, axis=0)==1)

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
        if np.all(commute_vec):
            # if Pword commutes with self then the rotation has identity action
            return self
        else:
            # note ~commute_vec == not commutes, this indexes the anticommuting terms
            commute_self = PauliwordOp(self.symp_matrix[commute_vec], self.coeff_vec[commute_vec])
            anticom_self = PauliwordOp(self.symp_matrix[~commute_vec], self.coeff_vec[~commute_vec])

            if angle is None:
                # assumes pi/2 rotation so Clifford
                anticom_part = (anticom_self*Pword_copy).multiply_by_constant(-1j)
                # if rotation is Clifford cannot produce duplicate terms so cleanup not necessary
                return PauliwordOp(
                    np.vstack([anticom_part.symp_matrix, commute_self.symp_matrix]), 
                    np.hstack([anticom_part.coeff_vec, commute_self.coeff_vec])
                )
            else:
                # if angle is specified, performs non-Clifford rotation
                anticom_part = (anticom_self.multiply_by_constant(np.cos(angle)) + 
                                (anticom_self*Pword_copy).multiply_by_constant(-1j*np.sin(angle)))
                return commute_self + anticom_part
                

    def perform_rotations(self, 
            rotations: List[Tuple["PauliwordOp", float]]
        ) -> "PauliwordOp":
        """ 
        Performs single Pauli rotations recursively left-to-right given a list of paulis supplied 
        either as strings or in the symplectic representation. This method does not allow coefficients 
        to be specified as rotation in this setting is ill-defined.

        If no angles are given then rotations are assumed to be pi/2 (Clifford)
        """
        op_copy = self.copy()
        for pauli_rotation,angle in rotations:
            op_copy = op_copy._rotate_by_single_Pword(pauli_rotation, angle).cleanup()
        return op_copy

    @cached_property
    def conjugate(self) -> "PauliwordOp":
        """
        Returns:
            Pword_conj (PauliwordOp): The Hermitian conjugated operator
        """
        Pword_conj = PauliwordOp(
            symp_matrix = self.symp_matrix, 
            coeff_vec   = self.coeff_vec.conjugate()
        )
        return Pword_conj

    @cached_property
    def to_openfermion(self) -> QubitOperator:
        """ convert to OpenFermion Pauli operator representation
        """
        pauli_terms = []
        for symp_vec, coeff in zip(self.symp_matrix, self.coeff_vec):
            pauli_terms.append(
                QubitOperator(' '.join([Pi+str(i) for i,Pi in enumerate(symplectic_to_string(symp_vec)) if Pi!='I']), 
                coeff)
            )
        return sum(pauli_terms)

    @cached_property
    def to_qiskit(self) -> PauliSumOp:
        """ convert to Qiskit Pauli operator representation
        """
        pauli_terms = []
        for symp_vec, coeff in zip(self.symp_matrix, self.coeff_vec):
            pauli_terms.append(PauliOp(primitive=Pauli(symplectic_to_string(symp_vec)), coeff=coeff))
        return sum(pauli_terms)

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
        Returns (2**n x 2**n) matrix of PauliwordOp where each Pauli operator has been kronector producted together

        This follows because tensor products of Pauli operators are one-sparse: they each have only
        one nonzero entry in each row and column

        Returns:
            sparse_matrix (csr_matrix): sparse matrix of PauliOp
        """
        if self.n_qubits == 0:
            return csr_matrix(self.coeff_vec)

        if self.n_qubits > 64:
            # numpy cannot handle ints over int64s (2**64) therefore use python objects
            binary_int_array = 1 << np.arange(self.n_qubits - 1, -1, -1).astype(object)
        else:
            binary_int_array = 1 << np.arange(self.n_qubits - 1, -1, -1)

        x_int = (self.X_block @ binary_int_array).reshape(-1, 1)
        z_int = (self.Z_block @ binary_int_array).reshape(-1, 1)

        Y_number = np.sum(np.bitwise_and(self.X_block, self.Z_block).astype(int), axis=1)
        global_phase = (-1j) ** Y_number

        dimension = 2 ** self.n_qubits
        row_ind = np.repeat(np.arange(dimension).reshape(1, -1), self.X_block.shape[0], axis=0)
        col_ind = np.bitwise_xor(row_ind, x_int)

        row_inds_and_Zint = np.bitwise_and(row_ind, z_int)
        vals = global_phase.reshape(-1, 1) * (-1) ** (
                    count1_in_int_bitstring(row_inds_and_Zint) % 2)  # .astype(complex))

        values_and_coeff = np.einsum('ij,i->ij', vals, self.coeff_vec)

        sparse_matrix = csr_matrix(
            (values_and_coeff.flatten(), (row_ind.flatten(), col_ind.flatten())),
            shape=(dimension, dimension),
            dtype=complex
        )
        return sparse_matrix

    def clique_cover(self, 
        edge_relation = 'C', 
        colouring_strategy='random_sequential', 
        colouring_interchange=False
        ) -> Dict[int, "PauliwordOp"]:
        """ Build a graph based on edge relation C (commuting), AC (anticommuting) or
        QWC (qubitwise commuting) and perform a graph colouring to identify cliques

        ------------------------
        | colouring strategies |
        ------------------------
        'largest_first'
        'random_sequential'
        'smallest_last'
        'independent_set'
        'connected_sequential_bfs'
        'connected_sequential_dfs'
        'connected_sequential' #(alias for the previous strategy)
        'saturation_largest_first'
        'DSATUR' #(alias for the previous strategy)

        """
        # build the adjacency matrix for the chosen edge relation
        if edge_relation == 'AC':
            adjmat = ~self.adjacency_matrix.copy()
        elif edge_relation == 'C':
            adjmat = self.adjacency_matrix.copy()
        elif edge_relation == 'QWC':
            adjmat = self.adjacency_matrix_qwc.copy()
        else:
            raise TypeError('Unrecognised edge relation, must be one of C (commuting), AC (anticommuting) or QWC (qubitwise commuting).')
        np.fill_diagonal(adjmat,False) # avoids self-adjacency
        # convert to a networkx graph and perform colouring on complement
        graph = nx.from_numpy_matrix(adjmat)
        inverted_graph = nx.complement(graph)
        col_map = nx.greedy_color(inverted_graph, strategy=colouring_strategy, interchange=colouring_interchange)
        # invert the resulting colour map to identify cliques
        cliques = {}
        for p_index, colour in col_map.items():
            cliques[colour] = cliques.get(
                colour, 
                PauliwordOp.from_list(['I'*self.n_qubits],[0])
            ) + self[p_index]
        return cliques


def random_PauliwordOp(n_qubits, n_terms, diagonal=False, complex_coeffs=True):
    """ Generate a random PauliwordOp with normally distributed coefficients
    """
    symp_matrix = random_symplectic_matrix(n_qubits, n_terms, diagonal)
    coeff_vec = np.random.randn(n_terms).astype(complex)
    if complex_coeffs:
        coeff_vec += 1j * np.random.randn(n_terms)

    return PauliwordOp(symp_matrix, coeff_vec)


def random_anitcomm_2n_1_PauliwordOp(n_qubits, complex_coeff=True, apply_clifford=True):
    """ Generate a anticommuting PauliOperator of size 2n+1 on n qubits (max possible size)
        with normally distributed coefficients. Generates in structured way then uses Clifford rotation (default)
        to try and make more random (can stop this to allow FAST build, but inherenet structure
         will be present as operator is formed in specific way!)
    """
    base = 'X' * n_qubits
    I_term = 'I' * n_qubits

    P_list = [base]
    for i in range(n_qubits):
        # Z_term
        P_list.append(base[:i] + 'Z' + I_term[i + 1:])
        # Y_term
        P_list.append(base[:i] + 'Y' + I_term[i + 1:])

    coeff_vec = np.random.randn(len(P_list)).astype(complex)
    if complex_coeff:
        coeff_vec += 1j * np.random.randn((len(P_list)))

    P_anticomm = PauliwordOp((dict(zip(P_list, coeff_vec))))

    # random rotations to get rid of structure
    if apply_clifford:
        for _ in range(10):
            P_rand = random_PauliwordOp(n_qubits, 1, complex_coeffs=complex_coeff)
            P_rand.coeff_vec[0] = 1
            P_anticomm = P_anticomm._rotate_by_single_Pword(P_rand,
                                                            None)

    anti_comm_check = P_anticomm.adjacency_matrix.astype(int) - np.eye(P_anticomm.adjacency_matrix.shape[0])
    assert (np.einsum('ij->', anti_comm_check) == 0), 'operator needs to be made of anti-commuting Pauli operators'

    return P_anticomm


class QuantumState:
    """ Class to represent quantum states.
    
    This is achieved by identifying the state with a 
    state_op (PauliwordOp), namely |0> --> Z, |1> --> X. 
    
    For example, the 2-qubit Bell state is mapped as follows: 
        1/sqrt(2) (|00> + |11>) --> 1/sqrt(2) (ZZ + XX)
    Observe the state is recovered by applying the state_op to the 
    zero vector |00>, which will be the X_block of state_op.
    
    This ensures correct phases when multiplying the quantum state by a PauliwordOp.

    QuantumState is defined in base.py to avoid circular imports since multiplication
    behaviour is defined between QuantumState and PauliwordOp
    """
    sigfig = 3 # specifies the number of significant figures for printing
    
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
                out_string += (f'{coeff: .{self.sigfig}f} |{basis_string}> +\n')
            elif self.vec_type == 'bra':
                out_string += (f'{coeff: .{self.sigfig}f} <{basis_string}| +\n')
            else:
                raise ValueError('Invalid vec_type, must be bra or ket')
        return out_string[:-3]

    def __repr__(self):
        return str(self)
    
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
    
    def __getitem__(self, key: Union[slice, int]) -> "QuantumState":
        """ Makes the QuantumState subscriptable - returns a QuantumState 
        constructed from the indexed rows and coefficients of the state matrix 
        """
        if isinstance(key, int):
            if key<0:
                # allow negative subscript
                key+=self.n_terms
            assert(key<self.n_terms), 'Index out of range'
            mask = [key]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start=0
            if stop is None:
                stop=self.n_terms
            mask = np.arange(start, stop, key.step)
        
        state_items = self.state_matrix[mask]
        coeff_items = self.coeff_vector[mask]
        return QuantumState(state_items, coeff_items)

    def __iter__(self):
        """ Makes a QuantumState instance iterable
        """
        return iter([self[i] for i in range(self.n_terms)])

    def cleanup(self, zero_threshold=1e-15) -> "QuantumState":
        """ Combines duplicate basis states, summing their coefficients
        """
        clean_state_op = self.state_op.cleanup(zero_threshold=zero_threshold)
        return QuantumState(
            clean_state_op.X_block, 
            clean_state_op.coeff_vec, 
            vec_type=self.vec_type
        )

    def sort(self, by='decreasing', key='magnitude') -> "QuantumState":
        """
        Sort the terms by some key, either magnitude, weight X, Y or Z
        """
        if key=='magnitude':
            sort_order = np.argsort(-abs(self.coeff_vector))
        elif key=='support':
            sort_order = np.argsort(-np.einsum('ij->i', self.state_matrix))
        else:
            raise ValueError('Only permitted sort key values are magnitude or support')
        if by=='increasing':
            sort_order = sort_order[::-1]
        elif by!='decreasing':
            raise ValueError('Only permitted sort by values are increasing or decreasing')
        return QuantumState(self.state_matrix[sort_order], self.coeff_vector[sort_order])

    def sectors_present(self, symmetry):
        """ return the sectors present within the QuantumState w.r.t. a StabilizerOp
        """
        symmetry_copy = symmetry.copy()
        symmetry_copy.coeff_vec = np.ones(symmetry.n_terms)
        sector = np.array([self.conjugate*S*self for S in symmetry_copy])
        return sector

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