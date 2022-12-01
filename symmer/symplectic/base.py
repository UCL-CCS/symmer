import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from copy import deepcopy
from itertools import product
from functools import reduce
from typing import Dict, List, Tuple, Union
from numbers import Number
from cached_property import cached_property
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from symmer.symplectic.utils import *
from openfermion import QubitOperator, count_qubits
import matplotlib.pyplot as plt
from qiskit.opflow import PauliOp, PauliSumOp
from scipy.stats import unitary_group
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
        symp_matrix = np.asarray(symp_matrix)
        if symp_matrix.dtype == int:
            # initialization is slow if not boolean array
            assert(set(np.unique(symp_matrix)).issubset({0,1})), 'symplectic matrix not defined with 0 and 1 only'
            symp_matrix = symp_matrix.astype(bool)
        assert(symp_matrix.dtype == bool), 'Symplectic matrix must be defined over integers'
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
    def random(cls, 
            n_qubits: int, 
            n_terms:  int, 
            diagonal: bool = False, 
            complex_coeffs: bool = True
        ) -> "PauliwordOp":
        """ Generate a random PauliwordOp with normally distributed coefficients
        """
        symp_matrix = random_symplectic_matrix(n_qubits, n_terms, diagonal)
        coeff_vec = np.random.randn(n_terms).astype(complex)
        if complex_coeffs:
            coeff_vec += 1j * np.random.randn(n_terms)
        return cls(symp_matrix, coeff_vec)

    @classmethod
    def haar_random(cls,
            n_qubits: int,
        ) -> "PauliwordOp":
        """ Generate a Haar random U(N) matrix (N^n_qubits) as a linear combination of Pauli operators.
        aka generate a uniform random unitary from a Hilbert space.

        Args:
            n_qubits: number of qubits
        Returns:
            p_random (PauliwordOp): Haar random matrix in Pauli basis
        """
        haar_matrix = unitary_group.rvs(2**n_qubits)
        p_random = cls.from_matrix(haar_matrix)
        return p_random

    @classmethod
    def from_list(cls, 
            pauli_terms :List[str], 
            coeff_vec:   List[complex] = None
        ) -> "PauliwordOp":
        """ Initialize a PauliwordOp from its Pauli terms and coefficients stored as lists
        """
        n_rows = len(pauli_terms)
        if coeff_vec is None:
            coeff_vec = np.ones(n_rows)
        else:
            coeff_vec = np.array(coeff_vec)
            if len(coeff_vec.shape)==2:
                # if coeff_vec supplied as list of tuples (real, imag) 
                # then converts to single complex vector
                assert(coeff_vec.shape[1]==2), 'Only tuples of size two allowed (real and imaginary components)'
                coeff_vec = coeff_vec[:,0] + 1j*coeff_vec[:,1]
        
        if pauli_terms:
            n_qubits = len(pauli_terms[0])
            symp_matrix = np.zeros((n_rows, 2 * n_qubits), dtype=int)
            for row_ind, pauli_str in enumerate(pauli_terms):
                symp_matrix[row_ind] = string_to_symplectic(pauli_str, n_qubits)
        else:
            n_qubits = 0
            symp_matrix = np.array([[]], dtype=bool)
        return cls(symp_matrix, coeff_vec)

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
    def from_openfermion(cls, 
            openfermion_op: QubitOperator
        ) -> "PauliwordOp":
        """ Initialize a PauliwordOp from OpenFermion's QubitOperator representation
        """
        assert(isinstance(openfermion_op, QubitOperator)), 'Must supply a QubitOperator'
        operator_dict = QubitOperator_to_dict(
            openfermion_op, count_qubits(openfermion_op)
        )
        return cls.from_dictionary(operator_dict)

    @classmethod
    def from_qiskit(cls,
            qiskit_op: PauliSumOp
        ) -> "PauliwordOp":
        """ Initialize a PauliwordOp from Qiskit's PauliSumOp representation
        """
        assert(isinstance(qiskit_op, PauliSumOp)), 'Must supply a PauliSumOp'
        operator_dict = PauliSumOp_to_dict(
            qiskit_op
        )
        return cls.from_dictionary(operator_dict)

    @classmethod
    def empty(cls, 
            n_qubits: int
        ) -> "PauliwordOp":
        """ Initialize an empty PauliwordOp of the form 0 * I...I
        """
        return cls.from_dictionary({'I'*n_qubits:0})

    @classmethod
    def _from_matrix_full_basis(cls, 
            matrix: Union[np.array, csr_matrix], 
            n_qubits: int,
            operator_basis: "PauliwordOp" = None
        ) -> "PauliwordOp":
        if operator_basis is None:
            # fast method to build all binary assignments
            int_list = np.arange(4 ** (n_qubits))
            XZ_block = (((int_list[:, None] & (1 << np.arange(2 * n_qubits))[::-1])) > 0).astype(int)
            op_basis = cls(XZ_block, np.ones(XZ_block.shape[0]))
        else:
            op_basis = operator_basis

        denominator = 2 ** n_qubits
        decomposition = cls.empty(n_qubits)
        for op in tqdm(op_basis, desc='Building operator via full basis', total=op_basis.n_terms):
            if isinstance(matrix, np.ndarray):
                const = np.einsum(
                    'ij,ij->', 
                    op.to_sparse_matrix.toarray(), 
                    matrix, 
                    optimize=True
                ) / denominator
            else:
                const = (op.to_sparse_matrix.multiply(matrix)).sum() / denominator
            decomposition += op.multiply_by_constant(const)

        operator_out = decomposition.cleanup()
        if operator_basis is not None:
            if not np.all(operator_out.to_sparse_matrix.toarray() == matrix):
                warnings.warn('Basis not sufficiently expressive, output operator projected onto basis supplied.')

        return operator_out

    @classmethod
    def _from_matrix_projector(cls, 
            matrix: Union[np.array, csr_matrix],
            n_qubits: int
        ) -> "PauliwordOp":
        """
        """
        if isinstance(matrix, np.ndarray):
            row, col = np.where(matrix)
        elif isinstance(matrix, (csr_matrix, csc_matrix, coo_matrix)):
            row, col = matrix.nonzero()
        else:
            raise ValueError('Unrecognised matrix type, must be one of np.array or sp.sparse.csr_matrix')
        
        binary_vec = (
            (
                np.arange(2 ** n_qubits).reshape([-1, 1]) & 
                (1 << np.arange(n_qubits))[::-1]
            ) > 0
        ).astype(bool)

        P_out = cls.empty(n_qubits)
        for i,j in tqdm(zip(row, col), desc='Building operator via projectors', total=len(row)):
            ij_op = get_ij_operator(i,j,n_qubits,binary_vec=binary_vec) 
            P_out += ij_op * matrix[i,j]

        return P_out

    @classmethod
    def from_matrix(cls, 
            matrix: Union[np.array, csr_matrix], 
            operator_basis: "PauliwordOp" = None,
            strategy: str = 'projector'
        ) -> "PauliwordOp":
        """
        --------------
        | strategies |
        --------------
        
        - full_basis
        
        If user doesn't define an operator basis then builds the full 4^N Hilbert space - this can be costly!.
        The user can avoid this by specificying a reduced basis that targets a subspace; there is a check 
        to assess whether the basis is sufficiently expressible to represent the input matrix in this case.
        
        - projector

        Scales as O(M*2^N) where M the number of nonzero elements in the matrix.
        """
        if isinstance(matrix, np.matrix):
            # summing over numpy matrices does not function correctly
            matrix = np.array(matrix)

        n_qubits = int(np.ceil(np.log2(max(matrix.shape))))

        if n_qubits > 30 and operator_basis is None:
            # could change XZ_block builder to use numpy objects (allows above 64-bit integers) but very slow and matrix will be too large to build 
            raise ValueError('Matrix too large! Will run into memory limitations.')

        if not (2**n_qubits, 2**n_qubits) == matrix.shape:
            # padding matrix with zeros so correct size
            temp_mat = np.zeros((2 ** n_qubits, 2 ** n_qubits))
            temp_mat[:matrix.shape[0],
                :matrix.shape[1]] = matrix
            matrix = temp_mat

        if strategy == 'full_basis' or operator_basis is not None:
            operator_out = cls._from_matrix_full_basis(
                matrix=matrix, n_qubits=n_qubits, operator_basis=operator_basis
            )
        elif strategy == 'projector':
            operator_out = cls._from_matrix_projector(
                matrix=matrix, n_qubits=n_qubits
            )
        else:
            raise ValueError('Unrecognised strategy, must be one of full_basis or projector')
        
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

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "PauliwordOp":
        """ 
        Create a carbon copy of the class instance
        """
        return deepcopy(self)

    def sort(self, 
            by:  str = 'magnitude',
            key: str = 'decreasing'
        ) -> "PauliwordOp":
        """
        Sort the terms either by magnitude, weight X, Y or Z
        """
        if by=='magnitude':
            sort_order = np.argsort(-abs(self.coeff_vec))
        elif by=='weight':
            sort_order = np.argsort(-np.sum(self.symp_matrix.astype(int), axis=1))
        elif by=='Z':
            sort_order = np.argsort(np.sum((self.n_qubits+1)*self.X_block.astype(int) + self.Z_block.astype(int), axis=1))
        elif by=='X':
            sort_order = np.argsort(np.sum(self.X_block.astype(int) + (self.n_qubits+1)*self.Z_block.astype(int), axis=1))
        elif by=='Y':
            sort_order = np.argsort(np.sum(abs(self.X_block.astype(int) - self.Z_block.astype(int)), axis=1))
        else:
            raise ValueError('Only permitted sort by values are magnitude, weight, X, Y or Z')
        if key=='increasing':
            sort_order = sort_order[::-1]
        elif key!='decreasing':
            raise ValueError('Only permitted sort by values are increasing or decreasing')
        return PauliwordOp(self.symp_matrix[sort_order], self.coeff_vec[sort_order])

    def basis_reconstruction(self, 
            operator_basis: "PauliwordOp"
        ) -> np.array:
        """ Simultaneously reconstruct every operator term in the supplied basis.
        With B and M the symplectic form of the supplied basis and the internal 
        Pauli operator, respectively, we perform columnwise Gaussian elimination 
        to yield the matrix

                [ B ]     [ I | 0 ]
                |---| ->  |-------|
                [ M ]     [ R | F ]

        where R is the reconstruction matrix, i.e. M = RB, and F indicates which
        terms were succesfully reconstructed in the basis. If F is a zero matrix
        this means the basis is sufficiently expressible to reconstruct M. However,
        if any rows of F contain a non-zero entry, the corresponding row in R is
        not fully constructed.

        Since we only need to reduce columns, the algorithm scales with the number of
        qubits N, not the number of terms M, and is therefore at worst O(N^2).
        """
        dim = operator_basis.n_terms
        basis_op_stack = np.vstack([operator_basis.symp_matrix, self.symp_matrix])
        reduced = cref_binary(basis_op_stack)
        mask_successfully_reconstructed = np.all(~reduced[dim:,dim:], axis=1)
        op_reconstruction = reduced[dim:,:dim]
        return op_reconstruction, mask_successfully_reconstructed

    @cached_property
    def Y_count(self) -> np.array:
        """ 
        Count the qubit positions of each term set to Pauli Y

        cached_property means this only runs once and then is stored
        as self.Y_count

        Returns:
            numpy array of Y counts over terms of PauliwordOp
        """
        return np.sum(np.bitwise_and(self.X_block, self.Z_block), axis=1)

    def cleanup(self, 
            zero_threshold:float=1e-15
        ) -> "PauliwordOp":
        """ Apply symplectic_cleanup and delete terms with negligible coefficients
        """
        if self.n_qubits == 0:
            return PauliwordOp([], [np.sum(self.coeff_vec)])
        elif self.n_terms == 0:
            return PauliwordOp(np.zeros((1, self.symp_matrix.shape[1]), dtype=bool), [0])
        else:
            return PauliwordOp(
                *symplectic_cleanup(
                    self.symp_matrix, self.coeff_vec, zero_threshold=zero_threshold
                )
            )

    def __eq__(self, 
            Pword: "PauliwordOp"
        ) -> bool:
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
                not np.sum(np.logical_xor(check_1.symp_matrix, check_2.symp_matrix)) and 
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
            add_obj: Union[int, "PauliwordOp"]
        ) -> "PauliwordOp":
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
        Y_count_out = np.sum(np.bitwise_and(*np.hsplit(phaseless_prod,2)), axis=1)
        sign_change = (-1) ** (
            np.sum(np.bitwise_and(self.X_block, np.hsplit(symp_vec,2)[1]), axis=1) % 2
        ) # mod 2 as only care about parity
        # final phase modification
        phase_mod = sign_change * (1j) ** ((3*Y_count_in + Y_count_out) % 4) # mod 4 as roots of unity
        coeff_vec = phase_mod * self.coeff_vec * coeff
        return phaseless_prod, coeff_vec

    def _multiply_by_operator(self, 
            PwordOp: Union["PauliwordOp", "QuantumState", complex],
            zero_threshold: float = 1e-15
        ) -> "PauliwordOp":
        """ Right-multiplication of this PauliwordOp by another PauliwordOp or QuantumState ket.
        """
        assert (self.n_qubits == PwordOp.n_qubits), 'PauliwordOps defined for different number of qubits'

        if PwordOp.n_terms == 1:
            # no cleanup if multiplying by a single term (faster)
            symp_stack, coeff_stack = self._mul_symplectic(
                symp_vec=PwordOp.symp_matrix, 
                coeff=PwordOp.coeff_vec, 
                Y_count_in=self.Y_count+PwordOp.Y_count
            )
            pauli_mult_out = PauliwordOp(symp_stack, coeff_stack)
        else:
            # multiplication is performed at the symplectic level, before being stacked and cleaned
            symp_stack, coeff_stack = zip(
                *[self._mul_symplectic(symp_vec=symp_vec, coeff=coeff, Y_count_in=self.Y_count+Y_count) 
                for symp_vec, coeff, Y_count in zip(PwordOp.symp_matrix, PwordOp.coeff_vec, PwordOp.Y_count)]
            )
            pauli_mult_out = PauliwordOp(
                *symplectic_cleanup(
                    np.vstack(symp_stack), np.hstack(coeff_stack), zero_threshold=zero_threshold
                )
            )
        return pauli_mult_out

    def _multiply_by_operator_parallel(self, 
            PwordOp: Union["PauliwordOp", "QuantumState", complex],
            zero_threshold: float = 1e-15
        ) -> "PauliwordOp":
        """ Right-multiplication of this PauliwordOp by another PauliwordOp or QuantumState ket.
        """
        assert (self.n_qubits == PwordOp.n_qubits), 'PauliwordOps defined for different number of qubits'

        # multiplication is performed at the symplectic level, before being stacked and cleaned
        
        symp_matrix, coeff_vec = collect_multiplication_stack(self, PwordOp)
        pauli_mult_out = PauliwordOp(
            *symplectic_cleanup_parallel(
                symp_matrix, coeff_vec, zero_threshold=zero_threshold
            )
        )
        return pauli_mult_out

    def __mul__(self, 
            mul_obj: Union["PauliwordOp", "QuantumState", complex],
            zero_threshold: float = 1e-15
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

        # more efficient to multiply the larger operator from the right
        if self.n_terms < PwordOp.n_terms:
            pauli_mult_out = PwordOp.dagger._multiply_by_operator(
                self.dagger, zero_threshold=zero_threshold).dagger
        else:
            pauli_mult_out = self._multiply_by_operator(
                PwordOp, zero_threshold=zero_threshold)

        if isinstance(mul_obj, QuantumState):
            coeff_vec = pauli_mult_out.coeff_vec*(1j**pauli_mult_out.Y_count)
            # need to run a separate cleanup since identities are all mapped to Z, i.e. II==ZZ in QuantumState
            return QuantumState(pauli_mult_out.X_block.astype(int), coeff_vec).cleanup()
        else:
            return pauli_mult_out
        

    def __imul__(self, 
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
        """ in-place multiplication behaviour
        """
        return self.__mul__(PwordOp)

    def __pow__(self, 
            exponent:int
        ) -> "PauliwordOp":
        assert(isinstance(exponent, int)), 'the exponent is not an integer'
        if exponent == 0:
            return PauliwordOp.from_list(['I'*self.n_qubits],[1])
        else:
            factors = [self.copy()]*exponent
            return reduce(lambda x,y:x*y, factors)

    def __getitem__(self, 
            key: Union[slice, int]
        ) -> "PauliwordOp":
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
        elif isinstance(key, (list, np.ndarray)):
            mask = np.asarray(key)
        else:
            raise ValueError('Unrecognised input, must be an integer, slice, list or np.array')
        
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
        Omega_PwordOp_symp = np.hstack((PwordOp.Z_block,  PwordOp.X_block)).astype(int)
        return (self.symp_matrix @ Omega_PwordOp_symp.T) % 2 == 0

    def anticommutes_termwise(self,
            PwordOp: "PauliwordOp"
        ) -> np.array:
        return ~self.commutes_termwise(PwordOp)

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

    def commutator(self, 
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
        """ Computes the commutator [A, B] = AB - BA
        """
        return self * PwordOp - PwordOp * self

    def anticommutator(self, 
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
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
    def adjacency_matrix(self) -> np.array:
        """ Checks which terms of self commute within itself
        """
        return self.commutes_termwise(self)

    @cached_property
    def adjacency_matrix_qwc(self) -> np.array:
        """ Checks which terms of self qubitwise commute within itself
        """
        return self.qubitwise_commutes_termwise(self)

    @cached_property
    def is_noncontextual(self) -> bool:
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

    def tensor(self, right_op: "PauliwordOp") -> "PauliwordOp":
        """ Tensor current Pauli operator with another on the right (cannot interlace currently)
        """
        identity_block_right = np.zeros([right_op.n_terms, self.n_qubits]).astype(int)
        identity_block_left  = np.zeros([self.n_terms,  right_op.n_qubits]).astype(int)
        padded_left_symp = np.hstack([self.X_block, identity_block_left, self.Z_block, identity_block_left])
        padded_right_symp = np.hstack([identity_block_right, right_op.X_block, identity_block_right, right_op.Z_block])
        left_factor = PauliwordOp(padded_left_symp, self.coeff_vec)
        right_factor = PauliwordOp(padded_right_symp, right_op.coeff_vec)
        return left_factor * right_factor

    def get_graph(self, 
            edge_relation = 'C'
        ) -> nx.graph:
        """ Build a graph based on edge relation C (commuting), 
        AC (anticommuting) or QWC (qubitwise commuting).
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
        graph = nx.from_numpy_array(adjmat)
        return graph

    def largest_clique(self,
            edge_relation='C'
        ) -> "PauliwordOp":
        """ Return the largest clique w.r.t. the specified edge relation
        """
        # build graph
        graph = self.get_graph(edge_relation=edge_relation)
        pauli_indices = sorted(nx.find_cliques(graph), key=lambda x:-len(x))[0]
        return sum([self[i] for i in pauli_indices])

    def clique_cover(self, 
            edge_relation = 'C', 
            strategy='independent_set',
            colouring_interchange=False
        ) -> Dict[int, "PauliwordOp"]:
        """ Perform a graph colouring to identify a clique partition

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

           ------------------------
        | NON - colouring strategies |
        ------------------------
        'sorted_insertion' https://quantum-journal.org/papers/q-2021-01-20-385/pdf/

        """
        if strategy == 'sorted_insertion':
            ### not a graph approach
            if colouring_interchange is not False:
                warnings.warn(f'{strategy} is not a graph colouring method, so colouring_interchange flag is ignored')

            sorted_op_list = list(self.sort(by='magnitude', key='decreasing'))

            check_dic = {
                'C': lambda x, y: np.all(x.commutes_termwise(y)),
                'AC': lambda x, y: np.all(~x.commutes_termwise(y)),
                'QWC': lambda x, y: np.all(x.qubitwise_commutes_termwise(y))}

            cliques = {0: sorted_op_list[0]}
            new_clique_ind = 1
            for selected_op in sorted_op_list[1:]:
                term_added = False
                for key in cliques.keys():
                    clique = cliques[key]
                    if check_dic[edge_relation](selected_op, clique):
                        cliques[key] += selected_op
                        term_added = True
                        break
                if term_added is False:
                    cliques[new_clique_ind] = selected_op
                    new_clique_ind += 1
            return cliques
        else:
            # build graph and invert
            graph = self.get_graph(edge_relation=edge_relation)
            inverted_graph = nx.complement(graph)
            col_map = nx.greedy_color(inverted_graph, strategy=strategy, interchange=colouring_interchange)
            # invert the resulting colour map to identify cliques
            cliques = {}
            for p_index, colour in col_map.items():
                cliques[colour] = cliques.get(
                    colour,
                    PauliwordOp.from_list(['I'*self.n_qubits],[0])
                ) + self[p_index]
            return cliques

    @cached_property
    def dagger(self) -> "PauliwordOp":
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
        if len(pauli_terms) == 1:
            return pauli_terms[0]
        else:
            return sum(pauli_terms)

    @cached_property
    def to_qiskit(self) -> PauliSumOp:
        """ convert to Qiskit Pauli operator representation
        """
        return PauliSumOp.from_list(self.to_dictionary.items())

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
    def to_dataframe(self) -> pd.DataFrame:
        """ Convert operator to pd.DataFrame for easy conversion to LaTeX
        """
        paulis = list(self.to_dictionary.keys())
        DF_out = pd.DataFrame.from_dict({
            'Pauli terms': paulis, 
            'Coefficients (real)': self.coeff_vec.real
            }
        )
        if np.any(self.coeff_vec.imag):
            DF_out['Coefficients (imaginary)'] = self.coeff_vec.imag
        return DF_out



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

    def conjugate_op(self, R: 'PauliwordOp') -> 'PauliwordOp':
        """
        For a defined linear combination of pauli operators : R = ∑_{𝑖} ci Pi ... (note each P self-adjoint!)

        perform the adjoint rotation R self R† =  R [∑_{a} ca Pa] R†

        Args:
            R (PauliwordOp): operator to rotate self by
        Returns:
            rot_H (PauliwordOp): rotated operator

        ### Notes
        R = ∑_{𝑖} ci Pi
        R^{†} = ∑_{j}  cj^{*} Pj
        note i and j here run over the same indices!
        apply R H R^{†} where H is self (current Pauli defined in class object)

        ### derivation:

        = (∑_{𝑖} ci Pi ) * (∑_{a} ca Pa ) * ∑_{j} cj^{*} Pj

        = ∑_{a}∑_{i}∑_{j} (ci ca cj^{*}) Pi  Pa Pj

        # can write as case for when i==j and i!=j

        = ∑_{a}∑_{i=j} (ci ca ci^{*}) Pi  Pa Pi + ∑_{a}∑_{i}∑_{j!=i} (ci ca cj^{*}) Pi  Pa Pj

        # let C by the termwise commutator matrix between H and R
        = ∑_{a}∑_{i=j} (-1)^{C_{ia}} (ci ca ci^{*}) Pa  + ∑_{a}∑_{i}∑_{j!=i} (ci ca cj^{*}) Pi  Pa Pj

        # next write final term over upper triange (as i and j run over same indices)
        ## so add common terms for i and j and make j>i

        = ∑_{a}∑_{i=j} (-1)^{C_{ia}} (ci ca ci^{*}) Pa
          + ∑_{a}∑_{i}∑_{j>i} (ci ca cj^{*}) Pi  Pa Pj + (cj ca ci^{*}) Pj  Pa Pi

        # then need to know commutation relation betwen terms in R
        ## given by adjaceny matrix of R... here A


        = ∑_{a}∑_{i=j} (-1)^{C_{ia}} (ci ca ci^{*}) Pa
         + ∑_{a}∑_{i}∑_{j>i} (ci ca cj^{*}) Pi  Pa Pj + (-1)^{C_{ia}+A_{ij}+C_{ja}}(cj ca ci^{*}) Pi  Pa Pj


        = ∑_{a}∑_{i=j} (-1)^{C_{ia}} (ci ca ci^{*}) Pa
         + ∑_{a}∑_{i}∑_{j>i} (ci ca cj^{*} + (-1)^{C_{ia}+A_{ij}+C_{ja}}(cj ca ci^{*})) Pi  Pa Pj


        """

        # see from symmer.symplectic.anticommuting_op import conjugate_Pop_with_R
        raise NotImplementedError('not done yet. Full function at: from symmer.symplectic.anticommuting_op.conjugate_Pop_with_R')


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
        state_matrix = state_matrix.astype(int) # in case input is boolean
        assert(set(state_matrix.flatten()).issubset({0,1})) # must be binary, does not support N-ary qubits
        self.n_terms, self.n_qubits = state_matrix.shape
        self.state_matrix = state_matrix
        if coeff_vector is None:
            # if no coefficients specified produces a uniform superposition
            coeff_vector = np.ones(self.n_terms)/np.sqrt(self.n_terms)
        self.vec_type = vec_type
        # the quantum state is manipulated via the state_op PauliwordOp
        symp_matrix = np.hstack([state_matrix, 1-state_matrix])
        self.state_op = PauliwordOp(symp_matrix, coeff_vector)

    def copy(self) -> "QuantumState":
        """ 
        Create a carbon copy of the class instance
        """
        return deepcopy(self)

    @classmethod
    def haar_random(cls,
                    n_qubits: int,
                    vec_type: str='ket') -> "QuantumState":
        """
        Generate a Haar random quantum state - (uniform random quantum state).

        Args:
            n_qubits: number of qubits
            vec_type (str): bra or ket

        Returns:
            qstate_random (QuantumState): Haar random quantum state

        """
        if vec_type=='ket':
            haar_vec = (unitary_group.rvs(2**n_qubits)[:,0]).reshape([-1, 1])
        elif vec_type == 'bra':
            haar_vec = (unitary_group.rvs(2**n_qubits)[0,:]).reshape([1, -1])
        else:
            raise ValueError(f'vector type: {vec_type} unkown')

        qstate_random = cls.from_array(haar_vec)
        return qstate_random
    
    @classmethod
    def random_state(cls, num_qubits: int, num_terms: int, vec_type: str='ket') -> "QuantumState":
        """ Generates a random normalized QuantumState, but not from Haar distribution
        """
        # random binary array with N columns, M rows
        random_state = np.random.randint(0,2,(num_terms,num_qubits))
        # random vector of coefficients
        coeff_vec = (
            np.random.rand(num_terms) + 
            np.random.rand(num_terms)*1j
        )
        return QuantumState(random_state, coeff_vec, vec_type=vec_type).normalize
    
    @classmethod
    def zero_state(cls,
                    n_qubits: int,
                    vec_type: str='ket') -> "QuantumState":
        """
        Generate the all zero state on N qubits

        Args:
            n_qubits: number of qubits
            vec_type (str): bra or ket

        Returns:
            q_zero_state (QuantumState): zero ket or bra quantum state

        """
        binary_zero = np.zeros(n_qubits).reshape(1,-1)
        q_zero_state = QuantumState(binary_zero, coeff_vector=np.array([1]), vec_type=vec_type)
        return q_zero_state

    def __str__(self) -> str:
        """ 
        Defines the print behaviour of QuantumState - differs depending on vec_type

        Returns:
            out_string (str): human-readable QuantumState string
        """
        out_string = ''
        for basis_vec, coeff in zip(self.state_matrix, self.state_op.coeff_vec):
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

    def __eq__(self, 
            Qstate: "QuantumState"
        ) -> bool:
        return self.state_op == Qstate.state_op
    
    def __add__(self, 
            Qstate: "QuantumState"
        ) -> "QuantumState":
        """ Add to this QuantumState another QuantumState by summing 
        the respective state_op (PauliwordOp representing the state)
        """
        new_state = self.state_op + Qstate.state_op
        return QuantumState(new_state.X_block, new_state.coeff_vec)

    def __radd__(self, 
            add_obj: Union[int, "QuantumState"]
        ) -> "QuantumState":
        """ Allows use of sum() over a list of PauliwordOps
        """
        if add_obj == 0:
            return self
        else:
            return self + add_obj
    
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
        if isinstance(mul_obj, Number):
            return QuantumState(self.state_matrix, self.state_op.coeff_vec*mul_obj)
        
        assert(self.n_qubits == mul_obj.n_qubits), 'Multiplication object defined for different number of qubits'
        assert(self.vec_type=='bra'), 'Cannot multiply a ket from the right'
        
        if isinstance(mul_obj, QuantumState):
            assert(mul_obj.vec_type=='ket'), 'Cannot multiply a bra with another bra'
            inner_product=0
            for (bra_string, bra_coeff),(ket_string, ket_coeff) in product(
                    zip(self.state_matrix, self.state_op.coeff_vec), 
                    zip(mul_obj.state_matrix, mul_obj.state_op.coeff_vec)
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
        coeff_items = self.state_op.coeff_vec[mask]
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
            sort_order = np.argsort(-abs(self.state_op.coeff_vec))
        elif key=='support':
            sort_order = np.argsort(-np.sum(self.state_matrix, axis=1))
        else:
            raise ValueError('Only permitted sort key values are magnitude or support')
        if by=='increasing':
            sort_order = sort_order[::-1]
        elif by!='decreasing':
            raise ValueError('Only permitted sort by values are increasing or decreasing')
        return QuantumState(self.state_matrix[sort_order], self.state_op.coeff_vec[sort_order])

    def sectors_present(self, symmetry):
        """ return the sectors present within the QuantumState w.r.t. a StabilizerOp
        """
        symmetry_copy = symmetry.copy()
        symmetry_copy.coeff_vec = np.ones(symmetry.n_terms)
        sector = np.array([self.dagger*S*self for S in symmetry_copy])
        return sector

    @cached_property
    def normalize(self):
        """ Normalize a state by dividing through its norm.

        Returns:
            self (QuantumState)
        """
        coeff_vector = self.state_op.coeff_vec/np.linalg.norm(self.state_op.coeff_vec)
        return QuantumState(self.state_matrix, coeff_vector, vec_type=self.vec_type)

    @cached_property
    def normalize_counts(self):
        """ Normalize a state by dividing through by the sum of coefficients and taking its square 
        root. This normalization is faithful to the probability distribution one might obtain from
        quantum circuit sampling. A subtle difference, but important!

        Returns:
            self (QuantumState)
        """
        coeff_vector = np.sqrt(self.state_op.coeff_vec/np.sum(self.state_op.coeff_vec))
        return QuantumState(self.state_matrix, coeff_vector, vec_type=self.vec_type)
        
    @cached_property
    def dagger(self) -> "QuantumState":
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
            coeff_vector = self.state_op.coeff_vec.conjugate(),
            vec_type     = new_type
        )
        return conj_state

    @cached_property
    def to_sparse_matrix(self):
        """
        Returns:
            sparse_Qstate (csr_matrix): sparse matrix representation of the statevector
        """
        # nonzero_indices = [int(''.join([str(i) for i in row]),2) for row in self.state_matrix]
        if self.n_qubits<64:
            nonzero_indices = self.state_matrix @ (1 << np.arange(self.state_matrix.shape[1])[::-1])
        else:
            nonzero_indices = self.state_matrix @ (1 << np.arange(self.state_matrix.shape[1], dtype=object)[::-1])

        sparse_Qstate = csr_matrix(
            (self.state_op.coeff_vec, (nonzero_indices, np.zeros_like(nonzero_indices))),
            shape = (2**self.n_qubits, 1), 
            dtype=np.complex128
        )
        if self.vec_type == 'bra':
            # conjugate has already taken place, just need to make into row vector
            sparse_Qstate= sparse_Qstate.reshape([1,-1])
        return sparse_Qstate

    def _is_normalized(self) -> bool:
        """
        check if state is normalized

        Returns:
            True or False depending on if state is normalized

        """

        if not np.isclose(np.linalg.norm(self.state_op.coeff_vec), 1):
            return False
        else:
            return True

    def sample_state(self, n_samples: int, return_normalized: bool=False) -> "QuantumState":
        """
        Method to sample given quantum state in computational basis. Get an array of bitstrings and counts as output.

        Note if other basis measurement required perform change of basis on state first.

        Args:
            n_samples (int): how many bitstring samples to take
            return_normalized (bool): whether to normalize sampled state (defaults to False)

        Returns:
            samples_as_coeff_state (QuantumState): state approximated via sampling (normalized or not depending on optional arg)
        """

        if not self._is_normalized():
            raise ValueError('should not sample state that is not normalized')

        counter = np.random.multinomial(n_samples, np.abs(self.state_op.coeff_vec)**2)
        if return_normalized:
            # normalize counter (note counter will be real and positive)
            counter = np.sqrt(counter /n_samples)
            # NOTE this is NOT the same as normalizing the state using np.linalg.norm!

        samples_as_coeff_state = QuantumState(self.state_matrix,
                                              counter,
                                              vec_type=self.vec_type)  ## gives counts as coefficients!
        return samples_as_coeff_state

    @cached_property
    def to_dictionary(self) -> Dict[str, complex]:
        """ Return the QuantumState as a dictionary
        """
        state_dict = dict(
            zip(
                [''.join([str(i) for i in row]) for row in self.state_matrix], 
                self.state_op.coeff_vec
            )
        )
        return state_dict

    @classmethod
    def from_dictionary(cls, 
            state_dict: Dict[str, complex]
        ) -> "QuantumState":
        """ Initialize a QuantumState from a dictionary of the form {'1101':a, '0110':b, '1010':c, ...}. This is useful for
        converting the measurement output of a quantum circuit to a QuantumState object for further manipulation/bootstrapping.
        """
        bin_strings, coeff_vector = zip(*state_dict.items())
        coeff_vector = np.array(coeff_vector)
        state_matrix = np.array([[int(i) for i in bstr] for bstr in bin_strings])
        return cls(state_matrix, coeff_vector)

    @classmethod
    def from_array(cls,
            statevector: np.array,
            threshold: float =1e-15,
        ) -> "QuantumState":
        """ Initialize a QubitState from a vector of 2^N elements over N qubits
        Args:
            statevector (np.array): numpy array of quantum state (size 2^N by 1)
            threshold (float): threshold to determine zero amplitudes (absolute value)
        Returns:
            Qstate (QuantumState): a QuantumState object

        **example
            statevector = array([0.57735027,0,0,0,0,0.81649658,0,0]).reshape([-1,1])
            Qstate = QuantumState.from_array(statevector)
            print(Qstate)
            >>  0.5773502692 |000> +
                0.8164965809 |101>
        """
        assert(((len(statevector.shape)==2) and (1 in statevector.shape))), 'state must be a bra (row) or ket (column) vector'

        vec_type = 'ket'
        if statevector.shape[0] == 1:
            vec_type= 'bra'

        statevector = statevector.reshape([-1])

        N = np.log2(statevector.shape[0])
        assert (N - int(N) == 0), 'the statevector dimension is not a power of 2'

        if not np.isclose(np.linalg.norm(statevector), 1):
            warnings.warn(f'statevector is not normalized')

        N = int(N)
        non_zero = np.where(abs(statevector) >= threshold)[0]

        # build binary states of non_zero terms
        if N<64:
            state_matrix = (((non_zero[:, None] & (1 << np.arange(N))[::-1])) > 0).astype(int)
        else:
            state_matrix = (((non_zero[:, None] & (1 << np.arange(N, dtype=object))[::-1])) > 0).astype(int)

        coeff_vector = statevector[non_zero]
        Qstate = cls(state_matrix, coeff_vector, vec_type=vec_type)
        return Qstate

    def plot_state(self, 
            logscale:bool = False, 
            probability_threshold:float=None,
            dpi:int=100
        ):

        assert self._is_normalized(), 'should only plot normalized quantum states'

        # clean duplicate states and set amplitdue threshold
        if probability_threshold is not None:
            assert probability_threshold>=0 and probability_threshold<=1, 'Probability threshold is a number between 0 and 1.'
            zero_threshold = np.sqrt(probability_threshold)
        else:
            zero_threshold = None
        q_state = self.cleanup(zero_threshold=zero_threshold)
        prob = np.abs(q_state.state_op.coeff_vec) ** 2

        fig, ax = plt.subplots(1, 1, dpi=dpi)

        if q_state.state_op.n_qubits<64:
            x_binary_ints = q_state.state_matrix @ (1 << np.arange(q_state.state_matrix.shape[1])[::-1])
        else:
            x_binary_ints = q_state.state_matrix @ (1 << np.arange(q_state.state_matrix.shape[1], dtype=object)[::-1])

        if prob.shape[0]<2**8:
            # bar chart
            ax.bar(x_binary_ints, prob, width=1, edgecolor="white", linewidth=0.8)
            ax.set_xticks(x_binary_ints, labels=x_binary_ints.astype(str))
        else:
            # line plot
            sort_inds = np.argsort(x_binary_ints)
            x_data = x_binary_ints[sort_inds]
            y_data = prob[sort_inds]
            ax.plot(x_data, y_data)
        ax.set(xlabel='binary output', ylabel='probability amplitude')
        
        if logscale:
            ax.set_yscale('log')

        return (ax)


def get_PauliwordOp_projector(projector: Union[str, List[str], np.array]) -> "PauliwordOp":
    """
    Build PauliwordOp projector onto different qubit states. Using I to leave state unchanged and 0,1,+,-,*,% to fix
    qubit.

    key:
        I leaves qubit unchanged
        0,1 fixes qubit as |0>, |1> (Z basis)
        +,- fixes qubit as |+>, |-> (X basis)
        *,% fixes qubit as |i+>, |i-> (Y basis)

    e.g.
     'I+0*1II' defines the projector the state I ⊗ [ |+ 0 i+ 1>  <+ 0 i+ 1| ]  ⊗ II

    TODO: could be used to develop a control version of PauliWordOp

    Args:
        projector (str, list) : either string or list of strings defininng projector

    Returns:
        projector (PauliwordOp): operator that performs projection
    """
    if isinstance(projector, str):
        projector = np.array(list(projector))
    else:
        projector = np.asarray(projector)
    basis_dict = {'I':1,
                  '0':0, '1':1,
                  '+':0, '-':1,
                  '*':0, '%':1}
    assert len(projector.shape) == 1, 'projector can only be defined over a single string or single list of strings (each a single letter)'
    assert set(projector).issubset(list(basis_dict.keys())), 'unknown qubit state (must be I,X,Y,Z basis)'


    N_qubits = len(projector)
    qubit_inds_to_fix = np.where(projector!='I')[0]
    N_qubits_fixed = len(qubit_inds_to_fix)
    state_sign = np.array([basis_dict[projector[active_ind]] for active_ind in qubit_inds_to_fix])

    if N_qubits_fixed < 64:
        binary_vec = (((np.arange(2 ** N_qubits_fixed).reshape([-1, 1]) & (1 << np.arange(N_qubits_fixed))[
                                                                          ::-1])) > 0).astype(int)
    else:
        binary_vec = (((np.arange(2 ** N_qubits_fixed, dtype=object).reshape([-1, 1]) & (1 << np.arange(N_qubits_fixed,
                                                                                                        dtype=object))[
                                                                                        ::-1])) > 0).astype(int)

    # # assign a sign only to 'active positions' (0 in binary not relevent)
    # sign_from_binary = binary_vec * state_sign
    #
    # # need to turn 0s in matrix to 1s before taking product across rows
    # sign_from_binary = sign_from_binary + (sign_from_binary + 1) % 2
    #
    # sign = np.product(sign_from_binary, axis=1)

    sign = (-1)**((binary_vec@state_sign.T)%2)

    coeff = 1 / 2 ** (N_qubits_fixed) * np.ones(2 ** N_qubits_fixed)
    sym_arr = np.zeros((coeff.shape[0], 2 * N_qubits))

    # assumed in Z basis
    sym_arr[:, qubit_inds_to_fix + N_qubits] = binary_vec
    sym_arr = sym_arr.astype(bool)

    ### fix for Y and X basis

    X_inds_fixed = np.where(np.logical_or(projector == '+', projector == '-'))[0]
    # swap Z block and X block
    (sym_arr[:, X_inds_fixed],
     sym_arr[:,  X_inds_fixed+N_qubits]) = (sym_arr[:, X_inds_fixed+N_qubits],
                                            sym_arr[:, X_inds_fixed].copy())

    # copy Z block into X block
    Y_inds_fixed = np.where(np.logical_or(projector == '*', projector == '%'))[0]
    sym_arr[:, Y_inds_fixed] = sym_arr[:, Y_inds_fixed + N_qubits]

    projector = PauliwordOp(sym_arr, coeff * sign)
    return projector


def get_ij_operator(i,j,n_qubits,binary_vec=None):
    """
    """
    if n_qubits > 30:
        raise ValueError('Too many qubits, might run into memory limitations.')

    if binary_vec is None:
        binary_vec = (
            ((np.arange(2 ** n_qubits).reshape([-1, 1]) & 
            (1 << np.arange(n_qubits))[::-1])) > 0
        ).astype(bool)

    left  = np.array([int(i) for i in np.binary_repr(i, width=n_qubits)]).astype(bool)
    right = np.array([int(i) for i in np.binary_repr(j, width=n_qubits)]).astype(bool)

    AND = left & right # AND where -1 sign
    XZX_sign_flips = (-1) ** np.sum(AND & binary_vec, axis=1) # XZX = -X multiplications
        
    if i != j:
        XOR = left ^ right # XOR where +-i phase

        XZ_mult = left & binary_vec
        ZX_mult = binary_vec & right

        XZ_phase = (-1j) ** np.sum(XZ_mult & ~ZX_mult, axis=1) # XZ=-iY multiplications
        ZX_phase = (+1j) ** np.sum(ZX_mult & ~XZ_mult, axis=1) # ZX=+iY multiplications
        phase_mod = XZX_sign_flips * XZ_phase * ZX_phase
        
        ij_symp_matrix = np.hstack([np.tile(XOR, [2**n_qubits, 1]), binary_vec])
        ij_operator= PauliwordOp(ij_symp_matrix, phase_mod/2**n_qubits)
    else:
        ij_symp_matrix = np.hstack([np.zeros_like(binary_vec), binary_vec])
        ij_operator= PauliwordOp(ij_symp_matrix, XZX_sign_flips/2**n_qubits)
    
    return ij_operator
