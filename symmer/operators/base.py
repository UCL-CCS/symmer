import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from symmer import process
from symmer.operators.utils import (
    matmul_GF2, random_symplectic_matrix, string_to_symplectic, QubitOperator_to_dict, SparsePauliOp_to_dict,
    symplectic_to_string, cref_binary, check_independent, check_jordan_independent, symplectic_cleanup,
    check_adjmat_noncontextual, symplectic_to_openfermion, binary_array_to_int, count1_in_int_bitstring
)
from tqdm.auto import tqdm
from copy import deepcopy
from functools import reduce
from typing import List, Union, Optional, Dict, Tuple
from numbers import Number
from cached_property import cached_property
from scipy.stats import unitary_group
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, dok_matrix
from openfermion import QubitOperator, count_qubits
from qiskit.quantum_info import SparsePauliOp
warnings.simplefilter('always', UserWarning)

from qiskit._accelerate.sparse_pauli_op import (
    ZXPaulis,
    to_matrix_sparse,
)

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

class PauliwordOp:
    """ 
    A class thats represents an operator defined over the Pauli group in the symplectic representation.

    Attributes:
        sigfig (int): The number of significant figures for printing.
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

        Args:
            symp_matrix (Union[List[str], Dict[str, float], np.array]): The symplectic matrix representing the operator.
            coeff_vec (Union[List[complex], np.array]): The coefficients of the operator.
        """
        symp_matrix = np.asarray(symp_matrix)
        if symp_matrix.dtype == int:
            # initialization is slow if not boolean array
            assert(set(np.unique(symp_matrix)).issubset({0,1})), 'symplectic matrix not defined with 0 and 1 only'
            symp_matrix = symp_matrix.astype(bool)
        assert(symp_matrix.dtype == bool), 'Symplectic matrix must be defined over bools'
        if len(symp_matrix.shape)==1:
            symp_matrix = symp_matrix.reshape([1, len(symp_matrix)])
        self.symp_matrix = symp_matrix
        assert self.symp_matrix.shape[-1]%2 == 0, 'symplectic matrix must have even number of columns'
        assert len(self.symp_matrix.shape) == 2, 'symplectic matrix must be 2 dimensional only'
        self.n_qubits = self.symp_matrix.shape[1]//2
        self.coeff_vec = np.asarray(coeff_vec, dtype=complex)
        self.n_terms = self.symp_matrix.shape[0]
        assert(self.n_terms==len(self.coeff_vec)), 'coeff list and Pauliwords not same length'
        self.X_block = self.symp_matrix[:, :self.n_qubits]
        self.Z_block = self.symp_matrix[:, self.n_qubits:]

    def set_processing_method(self, method):
        """ Set the method to use when running parallelizable processes. 
        Valid options are: mp, ray, single_thread.
        """
        process.method = method
        
    @classmethod
    def random(cls, 
            n_qubits: int, 
            n_terms:  int, 
            diagonal: bool = False, 
            complex_coeffs: bool = True,
            density: float = 0.3
        ) -> "PauliwordOp":
        """ 
        Generate a random PauliwordOp with normally distributed coefficients.

        Args:
            n_qubits (int): The number of qubits.
            n_terms (int): The number of terms in the operator.
            diagonal (bool): Whether to generate a diagonal operator (default: False).
            complex_coeffs (bool): Whether to generate complex coefficients (default: True).
            density (float): The density of non-zero elements in the symplectic matrix (default: 0.3).

        Returns:
            PauliwordOp: A random PauliwordOp object.
        """
        symp_matrix = random_symplectic_matrix(n_qubits, n_terms, diagonal, density=density)
        coeff_vec = np.random.randn(n_terms).astype(complex)
        if complex_coeffs:
            coeff_vec += 1j * np.random.randn(n_terms)
        return cls(symp_matrix, coeff_vec)

    @classmethod
    def haar_random(cls,
            n_qubits: int,
            strategy: Optional[str] = 'projector',
            disable_loading_bar: Optional[bool] = False
        ) -> "PauliwordOp":
        """ 
        Generate a Haar random U(N) matrix (N^n_qubits) as a linear combination of Pauli operators.
        aka generate a uniform random unitary from a Hilbert space.

        Args:
            n_qubits: number of qubits
        Returns:
            p_random (PauliwordOp): Haar random matrix in Pauli basis
        """
        haar_matrix = unitary_group.rvs(2**n_qubits)
        p_random = cls.from_matrix(haar_matrix, strategy=strategy, disable_loading_bar=disable_loading_bar)
        return p_random

    @classmethod
    def from_list(cls, 
            pauli_terms :List[str], 
            coeff_vec:   List[complex] = None
        ) -> "PauliwordOp":
        """ 
        Initialize a PauliwordOp from its Pauli terms and coefficients stored as lists.

        Args:
            operator_dict (Dict[str, complex]): A dictionary representing the PauliwordOp.

        Returns:
            PauliwordOp: A new PauliwordOp object.
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
            symp_matrix = np.array([[]], dtype=bool)
        return cls(symp_matrix, coeff_vec)

    @classmethod
    def from_dictionary(cls,
            operator_dict: Dict[str, complex]
        ) -> "PauliwordOp":
        """ 
        Initialize a PauliwordOp from its dictionary representation {pauli:coeff, ...}

        Args:
            operator_dict (Dict[str, complex]): A dictionary representing the PauliwordOp.

        Returns:
            PauliwordOp: A new PauliwordOp object.
        """
        pauli_terms, coeff_vec = zip(*operator_dict.items())
        pauli_terms = list(pauli_terms)
        return cls.from_list(pauli_terms, coeff_vec)

    @classmethod
    def from_openfermion(cls, 
            openfermion_op: QubitOperator,
            n_qubits = None
        ) -> "PauliwordOp":
        """ 
        Initialize a PauliwordOp from OpenFermion's QubitOperator representation.

        Args:
            openfermion_op (QubitOperator): The QubitOperator to initialize from.
            n_qubits (int, optional): The number of qubits. If not provided, it is determined
                from the QubitOperator. Defaults to None.

        Returns:
            PauliwordOp: A new PauliwordOp object.
        """
        assert(isinstance(openfermion_op, QubitOperator)), 'Must supply a QubitOperator'
        if n_qubits is None:
            n_qubits = count_qubits(openfermion_op)
        
        operator_dict = QubitOperator_to_dict(
            openfermion_op, n_qubits
        )
        return cls.from_dictionary(operator_dict)

    @classmethod
    def from_qiskit(cls,
            qiskit_op: SparsePauliOp
        ) -> "PauliwordOp":
        """ 
        Initialize a PauliwordOp from Qiskit's SparsePauliOp representation.

        Args:
            qiskit_op (SparsePauliOp): The SparsePauliOp to initialize from.

        Returns:
            PauliwordOp: A new PauliwordOp object.
        """
        assert(isinstance(qiskit_op, SparsePauliOp)), 'Must supply a SparsePauliOp'
        operator_dict = SparsePauliOp_to_dict(
            qiskit_op
        )
        return cls.from_dictionary(operator_dict)

    @classmethod
    def empty(cls, 
            n_qubits: int
        ) -> "PauliwordOp":
        """ 
        Initialize an empty PauliwordOp of the form 0 * I...I

        Args:
            n_qubits (int): The number of qubits.

        Returns:
            PauliwordOp: A new PauliwordOp object.
        """
        return cls.from_dictionary({'I'*n_qubits:0})

    @classmethod
    def _from_matrix_full_basis(cls, 
            matrix: Union[np.array, csr_matrix], 
            n_qubits: int,
            operator_basis: "PauliwordOp" = None,
            disable_loading_bar: Optional[bool] = False
        ) -> "PauliwordOp":
        """
        Args:
            n_qubits (int): The number of qubits.
            operator_basis (PauliwordOp, optional): The operator basis to use. Default is 'None'.
            disable_loading_bar (bool) : whether to have loading bar when constructing operator

        Returns:
            PauliwordOp: A new PauliwordOp object.
        """
        if operator_basis is None:
            # fast method to build all binary assignments
            int_list = np.arange(4 ** (n_qubits))
            XZ_block = (((int_list[:, None] & (1 << np.arange(2 * n_qubits))[::-1])) > 0).astype(bool)
            op_basis = cls(XZ_block, np.ones(XZ_block.shape[0]))
        else:
            op_basis = operator_basis.copy().cleanup()
            op_basis.coeff_vec = np.ones(op_basis.coeff_vec.shape)

        denominator = 2 ** n_qubits
        coeffs = []
        for op in tqdm(op_basis, desc='Building operator via full basis', total=op_basis.n_terms, disable=disable_loading_bar):
            coeffs.append((op.to_sparse_matrix.multiply(matrix)).sum() / denominator)

        ### fix ZX Y phases generated!
        # Y_sign = (op_basis.Y_count % 2 * -2) + 1
        op_basis.coeff_vec = np.array(coeffs) * ((op_basis.Y_count % 2 * -2) + 1)

        if operator_basis is not None:
            warnings.warn('Basis supplied MAY not be sufficiently expressive, output operator projected onto basis supplied.')
        #     if isinstance(matrix, csr_matrix):
        #         tol=1e-15
        #         max_diff = np.abs(matrix - operator_out.to_sparse_matrix).max()
        #         flag = not (max_diff <= tol)
        #     else:
        #         flag = not np.allclose(operator_out.to_sparse_matrix.toarray(), matrix)
        #
        #     if flag:
        #         warnings.warn('Basis not sufficiently expressive, output operator projected onto basis supplied.')

        return op_basis[op_basis.coeff_vec.nonzero()[0]]

    @classmethod
    def _from_matrix_projector(cls, 
            matrix: Union[np.array, csr_matrix],
            n_qubits: int,
            disable_loading_bar: Optional[bool] = False
        ) -> "PauliwordOp":
        """
        Args:
            matrix (Union[np.array, csr_matrix]): The matrix to decompose.
            n_qubits (int): The number of qubits.
            disable_loading_bar (bool) : whether to have loading bar when constructing operator

        Returns:
            PauliwordOp: A new PauliwordOp object representing the decomposition of the matrix using projectors.
        """
        assert n_qubits <= 32, 'cannot decompose matrices above 32 qubits'

        if isinstance(matrix, np.ndarray):
            row, col = np.where(matrix)
        elif isinstance(matrix, (csr_matrix, csc_matrix, coo_matrix)):
            row, col = matrix.nonzero()
        else:
            raise ValueError('Unrecognised matrix type, must be one of np.array or sp.sparse.csr_matrix')

        sym_operator = dok_matrix((4 ** n_qubits, 2 * n_qubits),
                                  dtype=bool)

        coeff_operator = dok_matrix((4 ** n_qubits, 1),
                                    dtype=complex)

        binary_vec = (
                (
                        np.arange(2 ** n_qubits).reshape([-1, 1]) &
                        (1 << np.arange(n_qubits))[::-1]
                ) > 0).astype(bool)

        binary_convert = 1 << np.arange(2 * n_qubits)[::-1]

        constant = 2 ** n_qubits
        ij_same = (row == col)
        for i in tqdm(row[ij_same], desc='Building operator via projectors diag elements', total=sum(ij_same),
                 disable=disable_loading_bar):
            j = i
            ij_symp_matrix = np.hstack([np.zeros_like(binary_vec), binary_vec])
            proj_coeffs = ((-1) ** np.sum(np.logical_and(binary_vec[i], binary_vec[j]) & binary_vec, axis=1)) / constant
            int_list = np.einsum('j, ij->i', binary_convert, ij_symp_matrix)

            # populate sparse mats
            sym_operator[int_list, :] = ij_symp_matrix
            coeff_operator[int_list] += proj_coeffs.reshape(-1, 1) * matrix[i, j]
            del ij_symp_matrix, proj_coeffs, int_list


        for i, j in tqdm(zip(row[~ij_same], col[~ij_same]), desc='Building operator via projectors off-diag elements',
                         total=sum(~ij_same), disable=disable_loading_bar):
            proj_coeffs = (((-1) ** np.sum(np.logical_and(binary_vec[i], binary_vec[j]) & binary_vec, axis=1))
                      * ((-1j) ** np.sum((binary_vec[i] & binary_vec) & ~(binary_vec & binary_vec[j]), axis=1))
                      * ((+1j) ** np.sum((binary_vec & binary_vec[j]) & ~(binary_vec[i] & binary_vec),
                                         axis=1))) / constant

            ij_symp_matrix = np.hstack([np.tile((binary_vec[i] ^ binary_vec[j]), [2 ** n_qubits, 1]),
                                        binary_vec])

            ### find location in symp matrix
            int_list = np.einsum('j, ij->i', binary_convert, ij_symp_matrix)

            # populate sparse mats
            sym_operator[int_list, :] = ij_symp_matrix
            coeff_operator[int_list] += proj_coeffs.reshape(-1, 1) * matrix[i, j]
            del ij_symp_matrix, proj_coeffs, int_list

        ### only keep nonzero coeffs! (skips expensive cleanup)
        nonzero = coeff_operator.nonzero()[0]
        P_out = PauliwordOp(sym_operator[nonzero, :].toarray(),
                            coeff_operator[nonzero].toarray()[:, 0])

        # P_out = PauliwordOp(sym_operator.toarray(),
        #                    coeff_operator.toarray()[:,0]).cleanup()
        return P_out

    @classmethod
    def from_matrix(cls, 
            matrix: Union[np.array, csr_matrix], 
            operator_basis: "PauliwordOp" = None,
            strategy: str = 'projector',
            disable_loading_bar: Optional[bool] = False
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

        Args:
            matrix (Union[np.array, csr_matrix]): The matrix to construct the PauliwordOp from.
            operator_basis (PauliwordOp, optional): The operator basis to use for decomposition. Defaults to None.
            strategy (str, optional): The decomposition strategy. Options are 'full_basis' and 'projector'. Defaults to 'projector'.
            disable_loading_bar (bool, optional): whether to have loading bar that gives time estimate for decompostion time

        Returns:
            PauliwordOp: A new PauliwordOp object representing the matrix.
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
                matrix=matrix, n_qubits=n_qubits, operator_basis=operator_basis, disable_loading_bar=disable_loading_bar
            )
        elif strategy == 'projector':
            operator_out = cls._from_matrix_projector(
                matrix=matrix, n_qubits=n_qubits, disable_loading_bar=disable_loading_bar
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

        Args:
            by (str, optional): The criterion for sorting. Options are 'magnitude', 'weight', 'support', 'X', 'Y', 'Z'. Defaults to 'magnitude'.
            key (str, optional): The sorting order. Options are 'increasing' and 'decreasing'. Defaults to 'decreasing'.

        Returns:
            PauliwordOp: A new PauliwordOp object with sorted terms.    
        """
        if by == 'magnitude':
            sort_order = np.argsort(-abs(self.coeff_vec))
        elif by == 'lex':
            sort_order = np.lexsort(self.symp_matrix.T)
        elif by == 'weight':
            sort_order = np.argsort(-np.sum(self.symp_matrix.astype(int), axis=1))
        elif by == 'support':
            pos_terms_occur = np.logical_or(self.symp_matrix[:, :self.n_qubits], self.symp_matrix[:, self.n_qubits:])
            symp_matrix_view = np.ascontiguousarray(pos_terms_occur).view(
                np.dtype((np.void, pos_terms_occur.dtype.itemsize * pos_terms_occur.shape[1]))
            )
            sort_order = np.argsort(symp_matrix_view.ravel())[::-1]
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

    def reindex(self, qubit_map: Union[List[int], Dict[int, int]]):
        """ 
        Re-index qubit labels
        For example, can specify a dictionary {0:2, 2:3, 3:0} mapping qubits 
        to their new positions or a list [2,3,0] will achieve the same result.

        Args:
            qubit_map (Union[List[int], Dict[int, int]]): A mapping of qubit indices to their new positions. 
                It can be specified as a list [2, 3, 0] or a dictionary {0: 2, 2: 3, 3: 0}.

        Returns:
            PauliwordOp: A new PauliwordOp object with re-indexed qubit labels.
        """
        if isinstance(qubit_map, list):
            old_indices, new_indices = sorted(qubit_map), qubit_map
        elif isinstance(qubit_map, dict):
            old_indices, new_indices = zip(*qubit_map.items())
        old_set, new_set = set(old_indices), set(new_indices)
        setdiff = old_set.difference(new_set)
        assert len(new_indices) == len(new_set), 'Duplicated index'
        assert len(setdiff) == 0, f'Assignment conflict: indices {setdiff} cannot be mapped.'
        
        # map corresponding columns in the symplectic matrix to their new positions
        new_X_block = self.X_block.copy()
        new_Z_block = self.Z_block.copy()
        new_X_block[:,old_indices] = new_X_block[:,new_indices]
        new_Z_block[:,old_indices] = new_Z_block[:,new_indices]
        
        return PauliwordOp(np.hstack([new_X_block, new_Z_block]), self.coeff_vec)

    def generator_reconstruction(self, 
            generators: "PauliwordOp",
            override_independence_check: bool = False
        ) -> np.array:
        """ 
        Simultaneously reconstruct every operator term in the supplied basis.
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

        Args:
            generators (PauliwordOp): The basis operators for reconstruction.
            override_independence_check (bool): Whether to override the algebraic independence check of the generators.

        Returns:
            Tuple[np.array, np.array]: A tuple containing the reconstruction matrix and a mask indicating which terms were successfully reconstructed.
        """
        if not override_independence_check:
            assert check_independent(generators), 'Supplied generators are algebraically dependent'
        dim = generators.n_terms
        basis_op_stack = np.vstack([generators.symp_matrix, self.symp_matrix])
        reduced = cref_binary(basis_op_stack)
        mask_successfully_reconstructed = np.all(~reduced[dim:,dim:], axis=1)
        op_reconstruction = reduced[dim:,:dim]
        return op_reconstruction.astype(int), mask_successfully_reconstructed

    def jordan_generator_reconstruction(self, generators: "PauliwordOp"):
        """ 
        Reconstruct this PauliwordOp under the Jordan product PQ = {P,Q}/2
        with respect to the supplied generators

        Args:
            generators (PauliwordOp): The basis operators for reconstruction.

        Returns:
            Tuple[np.array, np.array]: A tuple containing the reconstruction matrix and a mask indicating which terms were successfully reconstructed.
        """
        assert check_jordan_independent(generators), 'The non-symmetry elements do not pairwise anticommute.'

        # first, separate symmetry elements  from anticommuting ones
        symmetry_mask = np.all(generators.commutes_termwise(generators), axis=1)

        if np.all(symmetry_mask):
            # If not anticommuting component, return standard generator recon over symmetries
            return self.generator_reconstruction(generators)
        else:
            # empty reconstruction matrix to be updated in loop over anticommuting elements
            op_reconstruction = np.zeros([self.n_terms, generators.n_terms])
            successfully_reconstructed = np.zeros(self.n_terms, dtype=bool)

            ac_terms = generators[~symmetry_mask]

            # loop over anticommuting elements to enforce Jordan condition (no two anticommuting elements multiplied)
            for _, clq in ac_terms.clique_cover(edge_relation='C').items():
                clq_indices = [np.where(np.all(generators.symp_matrix == t, axis=1))[0][0] for t in clq.symp_matrix]
                mask_symmetries_with_P = symmetry_mask.copy()
                mask_symmetries_with_P[np.array(clq_indices)] = True
                # reconstruct this PauliwordOp in the augemented symmetry + single anticommuting term generating set
                augmented_symmetries = generators[mask_symmetries_with_P]
                recon_mat_P, successful_P = self.generator_reconstruction(augmented_symmetries)
                # np.ix_ needed to correctly slice op_reconstruction as mask method does not work
                row, col = np.ix_(successful_P, mask_symmetries_with_P)
                op_reconstruction[row, col] = recon_mat_P[successful_P]
                # will have duplicate succesful reconstruction of symmetries, so only sets True once in logical OR
                successfully_reconstructed = np.logical_or(successfully_reconstructed, successful_P)

            return op_reconstruction.astype(int), successfully_reconstructed

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
        """ 
        Apply symplectic_cleanup and delete terms with negligible coefficients.

        Args:
            zero_threshold (float): Threshold below which coefficients are considered negligible.

        Returns:
            PauliwordOp: A new PauliwordOp object after cleanup.
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
        """ 
        In theory should use logical XNOR to check symplectic matrix match, however
        can use standard logical XOR and look for False indices instead (implementation
        skips an additional NOT operation)

        Args:
            Pword (PauliwordOp): The PauliwordOp object to compare with.

        Returns:
            bool: True if the two PauliwordOp objects are equal, False otherwise.
        """
        check_1 = self.cleanup().sort('lex')
        check_2 = Pword.cleanup().sort('lex')
        if check_1.n_qubits != check_2.n_qubits:
            raise ValueError('Operators defined over differing numbers of qubits.')
        elif check_1.n_terms != check_2.n_terms:
            return False
        else:
            return (not np.sum(np.logical_xor(check_1.symp_matrix, check_2.symp_matrix)) and
                         np.allclose(check_1.coeff_vec, check_2.coeff_vec))
            # if eq_flag is True:
            #     assert hash(check_1) == hash(check_2), 'equal objects have different hash values'
            # return eq_flag

    def __hash__(self) -> int:
        """ 
        Build unique hash from dictionary of PauliwordOp.

        Returns:
            int: The hash value of the PauliwordOp object.
        """

        self.cleanup()  # self.cleanup(zero_threshold=1e-15)

        # tuples are immutable
        tuple_of_tuples = tuple(self.to_dictionary.items())
        hash_val = hash(tuple_of_tuples)
        return hash_val

    def append(self, 
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
        """ 
        Append another PauliwordOp onto this one - duplicates allowed.

        Args:
            PwordOp (PauliwordOp): The PauliwordOp object to append.

        Returns:
            PauliwordOp: The new PauliwordOp object after appending.
        """
        assert (self.n_qubits == PwordOp.n_qubits), 'Pauliwords defined for different number of qubits'
        P_symp_mat_new = np.vstack((self.symp_matrix, PwordOp.symp_matrix))
        P_new_coeffs = np.hstack((self.coeff_vec, PwordOp.coeff_vec))
        return PauliwordOp(P_symp_mat_new, P_new_coeffs) 

    def __add__(self, 
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
        """ 
        Add to this PauliwordOp another PauliwordOp by stacking the
        respective symplectic matrices and cleaning any resulting duplicates.

        Args:
            PwordOp (PauliwordOp): The PauliwordOp object to add.

        Returns:
            PauliwordOp: The result of adding the two PauliwordOp objects.
        """
        # cleanup run to remove duplicate rows (Pauliwords)
        return self.append(PwordOp).cleanup()

    def __radd__(self, 
            add_obj: Union[int, "PauliwordOp"]
        ) -> "PauliwordOp":
        """ 
        Allows use of sum() over a list of PauliwordOps.

        Args:
            add_obj (Union[int, PauliwordOp]): The object to add to this PauliwordOp.

        Returns:
            PauliwordOp: The result of adding the object to this PauliwordOp.
        """
        if add_obj == 0:
            return self
        else:
            return self + add_obj

    def __sub__(self,
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
        """ 
        Subtract from this PauliwordOp another PauliwordOp 
        by negating the coefficients and summing.

        Args:
            PwordOp (PauliwordOp): The PauliwordOp object to subtract.

        Returns:
            PauliwordOp: The result of subtracting the PauliwordOp object from this PauliwordOp object.
        """     
        op_copy = PwordOp.copy()
        op_copy.coeff_vec*=-1
        
        return self+op_copy

    def multiply_by_constant(self, 
            const: complex
        ) -> "PauliwordOp":
        """
        Multiply the PauliwordOp by a complex coefficient.

        Args:
            const (complex): The complex constant to multiply by.

        Returns:
            PauliwordOp: The result of multiplying the PauliwordOp by the constant.
        """
        return PauliwordOp(self.symp_matrix, self.coeff_vec*const)

    def _multiply_by_operator(self, 
            PwordOp: Union["PauliwordOp", "QuantumState", complex],
            zero_threshold: float = 1e-15
        ) -> "PauliwordOp":
        """ 
        Right-multiplication of this PauliwordOp by another PauliwordOp or QuantumState ket.

        Performs Pauli multiplication with phases at the level of the symplectic 
        matrices to avoid superfluous PauliwordOp initializations. The phase compensation 
        is implemented as per https://doi.org/10.1103/PhysRevA.68.042318.

        Args:
            PwordOp (Union["PauliwordOp", "QuantumState", complex]): The operator or ket to multiply by.
            zero_threshold (float): The threshold below which coefficients are considered negligible.

        Returns:
            PauliwordOp: The resulting PauliwordOp after multiplication.
        """
        assert (self.n_qubits == PwordOp.n_qubits), 'PauliwordOps defined for different number of qubits'
        Q_symp_matrix = PwordOp.symp_matrix.reshape([PwordOp.n_terms, 1, 2*PwordOp.n_qubits])
        termwise_phaseless_prod = self.symp_matrix ^ Q_symp_matrix
        Y_count_in  = self.Y_count + PwordOp.Y_count.reshape(-1,1)
        Y_count_out = np.sum(termwise_phaseless_prod[:,:,:PwordOp.n_qubits] & termwise_phaseless_prod[:,:,PwordOp.n_qubits:], axis=2)
        sign_change = (-1) ** (np.sum(self.X_block & Q_symp_matrix[:,:,PwordOp.n_qubits:], axis=2) % 2)
        phase_mod = sign_change * (1j) ** ((3*Y_count_in + Y_count_out) % 4)
        return PauliwordOp(
            *symplectic_cleanup(
                termwise_phaseless_prod.reshape(-1,2*self.n_qubits), 
                (phase_mod * np.outer(self.coeff_vec, PwordOp.coeff_vec).T).reshape(-1), zero_threshold=zero_threshold
            )
        )
    
    def expval(self, psi: "QuantumState") -> complex:
        """ 
        Efficient (linear) expectation value calculation using projectors
        See single_term_expval function below for further details.

        Args:
            psi (QuantumState): The quantum state for which to calculate the expectation value.

        Returns:
            complex: The expectation value.
        """
        if self.n_terms > psi.n_terms and psi.n_terms > 10:
            return (psi.dagger * self * psi).real
        else:
            if self.n_terms > 1:
                @process.parallelize
                def f(P, psi):
                    return single_term_expval(P, psi)

                expvals = np.array(f(self, psi))
            else:
                expvals = np.array(single_term_expval(self, psi))

            return np.sum(expvals * self.coeff_vec).real

    def __mul__(self, 
            mul_obj: Union["PauliwordOp", "QuantumState", complex],
            zero_threshold: float = 1e-15
        ) -> "PauliwordOp":
        """ 
        Right-multiplication of this PauliwordOp by another PauliwordOp or QuantumState ket.

        Args:
            mul_obj (Union["PauliwordOp", "QuantumState", complex]): The object to multiply with.
            zero_threshold (float, optional): Threshold for cleaning up resulting PauliwordOp. Defaults to 1e-15.

        Returns:
            PauliwordOp: The result of the multiplication.
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
        """ 
        In-place multiplication behaviour.

        Args:
            PwordOp (PauliwordOp): The `PauliwordOp` object to right-multiply with.

        Returns:
            PauliwordOp: The result of the in-place multiplication.
        """
        return self.__mul__(PwordOp)

    def __pow__(self, 
            exponent:int
        ) -> "PauliwordOp":
        """
        Exponentiation behavior.

        Args:
            exponent (int): The exponent to raise the `PauliwordOp` to.

        Returns:
            PauliwordOp: The result of the exponentiation.
        """
        assert(isinstance(exponent, int)), 'the exponent is not an integer'
        if exponent == 0:
            return PauliwordOp.from_list(['I'*self.n_qubits],[1])
        else:
            factors = [self.copy()]*exponent
            return reduce(lambda x,y:x*y, factors)

    def __getitem__(self, 
            key: Union[slice, int]
        ) -> "PauliwordOp":
        """ 
        Makes the PauliwordOp subscriptable - returns a PauliwordOp constructed
        from the indexed row and coefficient from the symplectic matrix .

        Args:
            key (Union[slice, int]): The index or slice to select the rows.

        Returns:
            PauliwordOp: A new PauliwordOp object constructed from the selected rows and coefficients.
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
            raise ValueError(f'Unrecognised input {type(key)}, must be an integer, slice, list or np.array')
        
        symp_items = self.symp_matrix[mask]
        coeff_items = self.coeff_vec[mask]
        return PauliwordOp(symp_items, coeff_items)

    def __iter__(self):
        """ 
        Makes a PauliwordOp instance iterable.

        Returns:
            Iterator: An iterator that generates each term of the PauliwordOp.
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

        Args:
            PwordOp (PauliwordOp): The PauliwordOp to check for term-wise commutation.

        Returns:
            np.array: A Boolean array indicating the term-wise commutation.
        """
        assert (self.n_qubits == PwordOp.n_qubits), 'Pauliwords defined for different number of qubits'

        ### sparse code
        # adjacency_matrix = (
        #             csr_matrix(self.symp_matrix.astype(int)) @ csr_matrix(np.hstack((PwordOp.Z_block, PwordOp.X_block)).astype(int)).T)
        # adjacency_matrix.data = ((adjacency_matrix.data % 2) != 0)
        # return np.logical_not(adjacency_matrix.toarray())

        ### dense code
        # Omega_PwordOp_symp = np.hstack((PwordOp.Z_block,  PwordOp.X_block)).astype(int)
        # return (self.symp_matrix @ Omega_PwordOp_symp.T) % 2 == 0
        
        return ~matmul_GF2(self.symp_matrix, np.hstack((PwordOp.Z_block,  PwordOp.X_block)).T)

    def anticommutes_termwise(self,
            PwordOp: "PauliwordOp"
        ) -> np.array:
        """
        Args:
            PwordOp (PauliwordOp): The PauliwordOp to check for term-wise anticommutation.

        Returns:
            np.array: A Boolean array indicating the term-wise anticommutation.
        """
        return ~self.commutes_termwise(PwordOp)

    def qubitwise_commutes_termwise(self,
            PwordOp: "PauliwordOp"
        ) -> np.array:
        """ 
        Given the symplectic representation of a single Pauli operator,
        determines which operator terms of the internal PauliwordOp qubitwise commute

        Args:
            PwordOp (PauliwordOp): The PauliwordOp to check for qubitwise term-wise commutation.

        Returns:  
            QWC_matrix (np.array): 
                An array whose elements are True if the corresponding term
                qubitwise commutes with the input PwordOp.
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
        """ 
        Computes the commutator [A, B] = AB - BA.

        Args:
            PwordOp (PauliwordOp): The PauliwordOp to compute the commutator with.

        Returns:
            PauliwordOp: The commutator [A, B] = AB - BA.
        """
        return self * PwordOp - PwordOp * self

    def anticommutator(self, 
            PwordOp: "PauliwordOp"
        ) -> "PauliwordOp":
        """ 
        Computes the anticommutator {A, B} = AB + BA.

        Args:
            PwordOp (PauliwordOp): The PauliwordOp to compute the anticommutator with.

        Returns:
            PauliwordOp: The anticommutator {A, B} = AB + BA.
        """
        return self * PwordOp + PwordOp * self

    def commutes(self, 
            PwordOp: "PauliwordOp"
        ) -> bool:
        """ 
        Checks if every term of self commutes with every term of PwordOp.

        Args:
            PwordOp (PauliwordOp): The PauliwordOp to check for commutation.

        Returns:
            bool: True if all terms commute, False otherwise.
        """
        commutator = self.commutator(PwordOp).cleanup()
        return (commutator.n_terms == 0 or np.all(commutator.coeff_vec[0] == 0))
        
    @cached_property
    def adjacency_matrix(self) -> np.array:
        """ 
        Checks which terms of self commute within itself.

        Returns:
            np.array: Adjacency matrix.
        """
        return self.commutes_termwise(self)

    @cached_property
    def adjacency_matrix_qwc(self) -> np.array:
        """ 
        Checks which terms of self qubitwise commute within itself.

        Returns:
            np.array: Adjacency matrix.
        """
        return self.qubitwise_commutes_termwise(self)

    @cached_property
    def is_noncontextual(self) -> bool:
        """ 
        Returns True if the operator is noncontextual, False if contextual
        Scales as O(M^2), compared with the O(M^3) algorithm of https://doi.org/10.1103/PhysRevLett.123.200501
        where M is the number of terms in the operator.
        Constructing the adjacency matrix is by far the most expensive part - very fast once that has been built.

        Returns:
            bool: True if the operator is noncontextual, False if contextual.
        """
        if self.n_terms < 4:
            # all operators with 3 or less P are noncontextual
            return True
        return check_adjmat_noncontextual(self.adjacency_matrix)

    def _rotate_by_single_Pword(self,
            Pword: "PauliwordOp", 
            angle: float = None,
            threshold: float = 1e-18
        ) -> "PauliwordOp":
        """ 
        Let R(t) = e^{i t/2 Q} = cos(t/2)*I + i*sin(t/2)*Q, then one of the following can occur:
        R(t) P R^\dag(t) = P when [P,Q] = 0
        R(t) P R^\dag(t) = cos(t) P + sin(t) (-iPQ) when {P,Q} = 0

        This operation is Clifford when t=pi/2, since cos(pi/2) P - sin(pi/2) iPQ = -iPQ.
        For t!=pi/2 an increase in the number of terms can be observed (non-Clifford unitary).
        
        <!> Please note the definition of the angle in R(t)...
            different implementations could be out by a factor of 2!

        Args:
            Pword (PauliwordOp): The Pauliword to rotate by.
            angle (float): The rotation angle in radians. If None, a Clifford rotation (angle=pi/2) is assumed.
            threshold (float): Angle threshold for Clifford rotation (precision to which the angle is a multiple of pi/2)
        Returns:
            PauliwordOp: The rotated operator.
        """
        if angle is None: # for legacy compatibility
            angle = np.pi/2

        if angle.imag != 0:
            warnings.warn('Complex component in angle: this will be ignored.')
        angle = angle.real

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

            multiple = angle * 2 / np.pi
            int_part = round(multiple)
            if abs(int_part - multiple)<=threshold:
                if int_part % 2 == 0:
                    # no rotation for angle congruent to 0 or pi disregarding sign, fixed in next line
                    anticom_part = anticom_self
                else:
                    # rotation of -pi/2 disregarding sign, fixed in next line
                    anticom_part = (anticom_self*Pword_copy).multiply_by_constant(-1j)
                if int_part in [2,3]:
                    anticom_part = anticom_part.multiply_by_constant(-1)
                # if rotation is Clifford cannot produce duplicate terms so cleanup not necessary
                return PauliwordOp(
                    np.vstack([anticom_part.symp_matrix, commute_self.symp_matrix]), 
                    np.hstack([anticom_part.coeff_vec, commute_self.coeff_vec])
                )
            else:
                if abs(angle)>1e6:
                    warnings.warn('Large angle can lead to precision errors: recommend using high-precision math library such as mpmath or redefine angle in range [-pi, pi]')
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

        If no angles are given then rotations are assumed to be pi/2 (Clifford).

        Args:
            rotations (List[Tuple[PauliwordOp, float]]): A list of tuples, where each tuple contains a Pauliword
                to rotate by and the rotation angle in radians. If no angle is given, a Clifford rotation (angle=pi/2) is assumed.

        Returns:
            PauliwordOp: The operator after performing the rotations.
        """
        op_copy = self.copy()
        if rotations == []:
            return op_copy.cleanup()
        else:
            for pauli_rotation, angle in rotations:
                op_copy = op_copy._rotate_by_single_Pword(pauli_rotation, angle).cleanup()
            return op_copy

    def tensor(self, right_op: "PauliwordOp") -> "PauliwordOp":
        """ 
        Tensor current Pauli operator with another on the right (cannot interlace currently).

        Args:
            right_op (PauliwordOp): The Pauli operator to tensor with.

        Returns:
            PauliwordOp: The resulting Pauli operator after the tensor product.
        """
        identity_block_right = np.zeros([right_op.n_terms, self.n_qubits], dtype=bool)#.astype(int)
        identity_block_left  = np.zeros([self.n_terms,  right_op.n_qubits], dtype=bool)#.astype(int)
        padded_left_symp = np.hstack([self.X_block, identity_block_left, self.Z_block, identity_block_left])
        padded_right_symp = np.hstack([identity_block_right, right_op.X_block, identity_block_right, right_op.Z_block])
        left_factor = PauliwordOp(padded_left_symp, self.coeff_vec)
        right_factor = PauliwordOp(padded_right_symp, right_op.coeff_vec)
        return left_factor * right_factor

    def get_graph(self, 
            edge_relation: Optional[str]='C',
             label_nodes: Optional[bool]=False
        ) -> nx.graph:
        """
        Build a graph based on edge relation C (commuting), AC (anticommuting) or QWC (qubitwise commuting).
        Note if label_nodes set to True then node names are pauli operators.

        To draw:
        import networkx as nx
        H = PauliwordOp.random(3, 10)
        graph = H.get_graph(edge_relation='C', label_nodes=True)
        nx.draw(graph,
                with_labels = True,
                alpha=0.75,
                node_color="skyblue",
                width=0.1,
                node_size=750
        )

        Args:
            edge_relation (str): The edge relation to consider. Options are 'C' for commuting, 'AC' for anticommuting, and 'QWC' for qubitwise commuting. Defaults to 'C'.
            label_nodes (bool): flag to label nodes of graph
        Returns:
            nx.Graph: The graph representing the edge relation.
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

        if label_nodes:
            node_list = np.apply_along_axis(symplectic_to_string, 1, self.symp_matrix).tolist()
            mapping = dict(zip(range(len(node_list)), node_list))
            graph = nx.relabel_nodes(graph, mapping)

        return graph

    def largest_clique(self,
            edge_relation='C'
        ) -> "PauliwordOp":
        """ 
        Return the largest clique w.r.t. the specified edge relation.

        Args:
            edge_relation (str): The edge relation to consider. Options are 'C' for commuting, 'AC' for anticommuting, and 'QWC' for qubitwise commuting. Defaults to 'C'.

        Returns:
            PauliwordOp: The PauliwordOp representing the largest clique.
        """
        # build graph
        graph = self.get_graph(edge_relation=edge_relation)
        pauli_indices = sorted(nx.find_cliques(graph), key=lambda x:-len(x))[0]
        return sum([self[i] for i in pauli_indices])

    def clique_cover(self, 
            edge_relation = 'C', 
            strategy='largest_first',
            colouring_interchange=False
        ) -> Dict[int, "PauliwordOp"]:
        """ 
        Perform a graph colouring to identify a clique partition.

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

        Args:
            edge_relation (str, optional): The edge relation used for building the graph. 
                Must be one of the following:
                - 'C': Commuting relation.
                - 'AC': Anticommuting relation.
                - 'QWC': Qubitwise commuting relation.
                Defaults to 'C'.

            strategy (str, optional): The coloring strategy to be used. Must be one of the following:
                - 'largest_first': Nodes are colored by the number of their colored neighbors in decreasing order.
                - 'random_sequential': Nodes are colored in a random order.
                - 'smallest_last': Nodes are colored starting from the smallest degree.
                - 'independent_set': Independent set coloring strategy.
                - 'connected_sequential_bfs': Nodes are colored in a connected order using BFS.
                - 'connected_sequential_dfs': Nodes are colored in a connected order using DFS.
                - 'connected_sequential': Alias for 'connected_sequential_bfs'.
                - 'saturation_largest_first': Nodes are colored by their saturation degree in decreasing order.
                - 'DSATUR': Alias for 'saturation_largest_first'.
                Defaults to 'largest_first'.

            colouring_interchange (bool, optional): Specifies whether to use interchange optimization 
                during coloring. This can improve the quality of the coloring but may take more time.
                Defaults to False.

        Returns:
            Dict[int, "PauliwordOp"]: A dictionary where the keys represent the clique index and 
            the values represent the PauliwordOp objects corresponding to each clique.

        Raises:
            TypeError: If the edge_relation argument is not one of 'C', 'AC', or 'QWC'.
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
            Pword_conj (PauliwordOp): The Hermitian conjugated operator.
        """
        Pword_conj = PauliwordOp(
            symp_matrix = self.symp_matrix, 
            coeff_vec   = self.coeff_vec.conjugate()
        )
        return Pword_conj

    @cached_property
    def to_openfermion(self) -> QubitOperator:
        """ 
        Convert to OpenFermion Pauli operator representation.

        Returns:
            open_f (QubitOperator): The QubitOperator representation of the PauliwordOp.
        """
        open_f = QubitOperator()
        for P_sym, coeff in zip(self.symp_matrix, self.coeff_vec):
            open_f+=symplectic_to_openfermion(P_sym, coeff)
        return open_f

    @cached_property
    def to_qiskit(self) -> SparsePauliOp:
        """ 
        Convert to Qiskit Pauli operator representation.

        Returns:
            PauliSumOp: The PauliSumOp representation of the PauliwordOp.
        """
        Pstr_list = np.apply_along_axis(symplectic_to_string, 1, self.symp_matrix).tolist()

        return SparsePauliOp(Pstr_list, coeffs=self.coeff_vec.tolist())

    @cached_property
    def to_dictionary(self) -> Dict[str, complex]:
        """
        Method for converting the operator from the symplectic representation 
        to a dictionary of the form {P_string:coeff, ...}

        Returns:
            dict: The dictionary representation of the operator in the form {P_string: coeff, ...}.
        """
        # clean the operator since duplicated terms will be overwritten in the conversion to a dictionary
        op_to_convert = self.cleanup()
        out_dict = {symplectic_to_string(symp_vec):coeff for symp_vec, coeff
                    in zip(op_to_convert.symp_matrix, op_to_convert.coeff_vec)}
        return out_dict

    @cached_property
    def to_dataframe(self) -> pd.DataFrame:
        """ 
        Convert operator to pd.DataFrame for easy conversion to LaTeX.

        Returns:
            pd.DataFrame: The DataFrame representation of the operator.
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
    def generators(self) -> "PauliwordOp":
        """ Find an independent generating set for input Pauli operator

        Args:
            op (PauliwordOp): operator to find symmetry basis for

        Returns:
            generators (PauliwordOp): independet generating set for op
        """
        from symmer.operators.utils import _rref_binary

        row_red = _rref_binary(self.symp_matrix)
        non_zero_rows = row_red[np.sum(row_red, axis=1).astype(bool)]
        generators = PauliwordOp(non_zero_rows,
                          np.ones(non_zero_rows.shape[0], dtype=complex))

        assert check_independent(generators), 'generators are not independent'
        assert generators.n_terms <= 2*self.n_qubits, 'cannot have an independent generating set of size greaterthan 2 time num qubits'

        return generators

    @cached_property
    def to_sparse_matrix(self) -> csr_matrix:
        """
        Returns (2**n x 2**n) matrix of PauliwordOp where each Pauli operator has been kronector producted together

        This follows because tensor products of Pauli operators are one-sparse: they each have only
        one nonzero entry in each row and column

        Returns:
            sparse_matrix (csr_matrix): Sparse matrix of PauliOp.
        """
        if self.n_qubits == 0:
            return csr_matrix(self.coeff_vec)

        # if self.n_qubits>15:
        #     from symmer.utils import get_sparse_matrix_large_pauliwordop
        #     sparse_matrix = get_sparse_matrix_large_pauliwordop(self)
        #     return sparse_matrix
        # else:
        #     x_int = binary_array_to_int(self.X_block).reshape(-1, 1)
        #     z_int = binary_array_to_int(self.Z_block).reshape(-1, 1)

        #     Y_number = np.sum(np.bitwise_and(self.X_block, self.Z_block), axis=1)
        #     global_phase = (-1j) ** Y_number

        #     dimension = 2 ** self.n_qubits
        #     row_ind = np.repeat(np.arange(dimension).reshape(1, -1), self.X_block.shape[0], axis=0)
        #     col_ind = np.bitwise_xor(row_ind, x_int)

        #     row_inds_and_Zint = np.bitwise_and(row_ind, z_int)
        #     vals = global_phase.reshape(-1, 1) * (-1) ** (
        #                 count1_in_int_bitstring(row_inds_and_Zint) % 2)  # .astype(complex))

        #     values_and_coeff = np.einsum('ij,i->ij', vals, self.coeff_vec)

        #     sparse_matrix = csr_matrix(
        #         (values_and_coeff.flatten(), (row_ind.flatten(), col_ind.flatten())),
        #         shape=(dimension, dimension),
        #         dtype=complex
        #     )
        #     return sparse_matrix
        
        phase = np.zeros(self.n_terms, dtype=np.uint8)
        zx = ZXPaulis(
                    self.X_block[:,::-1],
                    self.Z_block[:,::-1],
                    phase,
                    self.coeff_vec,
                )
        
        data, indices, indptr = to_matrix_sparse(zx, force_serial=False)
        side = 1 << self.n_qubits
        return csr_matrix((data, indices, indptr), shape=(side, side))

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

        # see from symmer.operators.anticommuting_op import conjugate_Pop_with_R
        raise NotImplementedError('not done yet. Full function at: from symmer.operators.anticommuting_op.conjugate_Pop_with_R')


class QuantumState:
    """ 
    Class to represent quantum states.
    
    This is achieved by identifying the state with a 
    state_op (PauliwordOp), namely |0> --> Z, |1> --> X. 
    
    For example, the 2-qubit Bell state is mapped as follows: 
        1/sqrt(2) (|00> + |11>) --> 1/sqrt(2) (ZZ + XX)
    Observe the state is recovered by applying the state_op to the 
    zero vector |00>, which will be the X_block of state_op.
    
    This ensures correct phases when multiplying the quantum state by a PauliwordOp.

    QuantumState is defined in base.py to avoid circular imports since multiplication
    behaviour is defined between QuantumState and PauliwordOp.

     Attributes:
        sigfig (int): The number of significant figures for printing.
    """
    sigfig = 3 # specifies the number of significant figures for printing
    
    def __init__(self, 
            state_matrix: Union[List[List[int]], np.array], 
            coeff_vector: Union[List[complex], np.array] = None,
            vec_type: str = 'ket'
        ) -> None:
        """ 
        The state is not normalized by default, since this would result
        in incorrect behaviour when perfoming non-unitary multiplications,
        e.g. for evaluating expectation values of Hamiltonians. However, if
        one wishes to normalize the state, it is stored as a cached propoerty
        as QuantumState.normalize.

        Args:
            state_matrix (Union[List[List[int]], np.array]): The state matrix representing the quantum state.
            coeff_vector (Union[List[complex], np.array], optional): The coefficient vector for the quantum state. Defaults to None.
            vec_type (str, optional): The type of vector representation (e.g., 'ket', 'bra'). Defaults to 'ket'.
        """
        if isinstance(state_matrix, list):
            state_matrix = np.array(state_matrix)
        if isinstance(coeff_vector, list):
            coeff_vector = np.array(coeff_vector)
        if len(state_matrix.shape)==1: # incase a single basis state given
            state_matrix = state_matrix.reshape([1,-1])
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
        Create a carbon copy of the class instance.

        Returns:
            QuantumState: A copy of the class instance.
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
    def random(cls, num_qubits: int, num_terms: int, vec_type: str='ket') -> "QuantumState":
        """ 
        Generates a random normalized QuantumState, but not from Haar distribution.

        Args:
            num_qubits (int): The number of qubits.
            num_terms (int): The number of terms. Note duplicate bitstrings mean the produced state can have slightly fewer terms
            vec_type (str, optional): The vector type. Defaults to 'ket'.

        Returns:
            QuantumState: A random normalized QuantumState instance.
        """
        # random binary array with N columns, M rows
        random_state = np.random.randint(0,2,(num_terms,num_qubits))
        # random vector of coefficients
        coeff_vec = (
            np.random.rand(num_terms) + 
            np.random.rand(num_terms)*1j
        )
        return QuantumState(random_state, coeff_vec, vec_type=vec_type).cleanup().normalize
    
    @classmethod
    def zero(cls,
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
        """
        Returns a string representation of the QuantumState object.
        """
        return str(self)

    def __eq__(self, 
            Qstate: "QuantumState"
        ) -> bool:
        """
        Check if the current QuantumState object is equal to another QuantumState object.

        Args:
            Qstate (QuantumState): The QuantumState object to compare with.

        Returns:
            bool: True if the QuantumState objects are equal, False otherwise.
        """
        return self.state_op == Qstate.state_op
    
    def __add__(self, 
            Qstate: "QuantumState"
        ) -> "QuantumState":
        """ 
        Add to this QuantumState another QuantumState by summing 
        the respective state_op (PauliwordOp representing the state).

        Args:
            Qstate (QuantumState): The QuantumState object to add.

        Returns:
            QuantumState: A new QuantumState object representing the sum of the two QuantumStates.
        """
        new_state = self.state_op + Qstate.state_op
        return QuantumState(new_state.X_block, new_state.coeff_vec)

    def __radd__(self, 
            add_obj: Union[int, "QuantumState"]
        ) -> "QuantumState":
        """ 
        Allows use of sum() over a list of PauliwordOps.

        Args:
            add_obj (Union[int, QuantumState]): The object to add.

        Returns:
            QuantumState: A new QuantumState object representing the sum.
        """
        if add_obj == 0:
            return self
        else:
            return self + add_obj
    
    def __sub__(self, 
            Qstate: "QuantumState"
        ) -> "QuantumState":
        """ 
        Subtract from this QuantumState another QuantumState by subtracting 
        the respective state_op (PauliwordOp representing the state).

        Args:
            Qstate (QuantumState): The QuantumState object to subtract.

        Returns:
            QuantumState: A new QuantumState object representing the difference.
        """
        new_state_op = self.state_op - Qstate.state_op
        return QuantumState(new_state_op.X_block, new_state_op.coeff_vec)
    
    def __mul__(self,
        mul_obj: Union["QuantumState", PauliwordOp]
        ) -> Union["QuantumState", complex]:
        """
        Right multiplication of a bra QuantumState by either a ket QuantumState or PauliwordOp.
        
        Args:
            mul_obj (Union["QuantumState", PauliwordOp]): The object to multiply with.

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

            # set left state to be smallest in number of bitstrings making loop short!
            if self.state_op.n_terms < mul_obj.n_terms:
                left_state = self
                right_state = mul_obj
            else:
                left_state = mul_obj
                right_state = self

            # O(length_smallest_statevector) runtime. (Note linear rather than quadratic!)
            for bstring, left_coeff in left_state.to_dictionary.items():
                right_coeff = right_state.to_dictionary.get(bstring, 0)
                inner_product += left_coeff * right_coeff

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
        """ 
        Makes the QuantumState subscriptable - returns a QuantumState 
        constructed from the indexed rows and coefficients of the state matrix.

        Args:
            key (Union[slice, int]): The index or slice object used for subscripting.

        Returns:
            QuantumState: The QuantumState object constructed from the indexed rows and coefficients.
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
        """ 
        Makes a QuantumState instance iterable.

        Returns:
            iter: An iterator that generates elements by iterating over the QuantumState object.
        """
        return iter([self[i] for i in range(self.n_terms)])

    def cleanup(self, zero_threshold=1e-15) -> "QuantumState":
        """ 
        Combines duplicate basis states, summing their coefficients.

        Args:
            zero_threshold (float): Threshold below which coefficients are considered zero.

        Returns:
            QuantumState: A new QuantumState object with combined duplicate basis states.
        """
        clean_state_op = self.state_op.cleanup(zero_threshold=zero_threshold)
        return QuantumState(
            clean_state_op.X_block, 
            clean_state_op.coeff_vec, 
            vec_type=self.vec_type
        )

    def sort(self, by='decreasing', key='magnitude') -> "QuantumState":
        """
        Sort the terms by some key, either magnitude, weight X, Y or Z.

        Args:
            by (str): Sort order, either 'increasing' or 'decreasing'.
            key (str): Sort key, either 'magnitude' or 'support'.

        Returns:
            QuantumState: A new QuantumState object with sorted terms.
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

    def reindex(self, qubit_map: Union[List[int], Dict[int, int]]):
        """ 
        Re-index qubit labels.
        For example, can specify a dictionary {0:2, 2:3, 3:0} mapping qubits 
        to their new positions or a list [2,3,0] will achieve the same result.

        Args:
            qubit_map (Union[List[int], Dict[int, int]]): A mapping of qubits to their new positions. It can be specified
                as a list [2, 3, 0] or a dictionary {0: 2, 2: 3, 3: 0}.

        Returns:
            QuantumState: A new QuantumState object with re-indexed qubit labels.
        """
        if isinstance(qubit_map, list):
            old_indices, new_indices = sorted(qubit_map), qubit_map
        elif isinstance(qubit_map, dict):
            old_indices, new_indices = zip(*qubit_map.items())
        old_set, new_set = set(old_indices), set(new_indices)
        setdiff = old_set.difference(new_set)
        assert len(new_indices) == len(new_set), 'Duplicated index'
        assert len(setdiff) == 0, f'Assignment conflict: indices {setdiff} cannot be mapped.'
        
        # map corresponding columns in the state matrix to their new positions
        new_state_matrix = self.state_matrix.copy()
        new_state_matrix[:,old_indices] = new_state_matrix[:,new_indices]
        
        return QuantumState(new_state_matrix, self.state_op.coeff_vec, vec_type=self.vec_type)

    def sectors_present(self, symmetry):
        """ 
        Return the sectors present within the QuantumState w.r.t. a IndependentOp.

        Args:
            symmetry (IndependentOp): The IndependentOp object representing the symmetry.

        Returns:
            numpy.ndarray: An array representing the sectors present within the QuantumState.
        """
        symmetry_copy = symmetry.copy()
        symmetry_copy.coeff_vec = np.ones(symmetry.n_terms)
        sector = np.array([S.expval(self) for S in symmetry_copy])
        return sector

    @cached_property
    def normalize(self):
        """ 
        Normalize a state by dividing through its norm.

        Returns:
            self (QuantumState): The normalized QuantumState.
        """
        coeff_vector = self.state_op.coeff_vec/np.linalg.norm(self.state_op.coeff_vec)
        return QuantumState(self.state_matrix, coeff_vector, vec_type=self.vec_type)

    @cached_property
    def normalize_counts(self):
        """ 
        Normalize a state by dividing through by the sum of coefficients and taking its square 
        root. This normalization is faithful to the probability distribution one might obtain from
        quantum circuit sampling. A subtle difference, but important!

        Returns:
            self (QuantumState): The normalized QuantumState.
        """
        coeff_vector = np.sqrt(self.state_op.coeff_vec/np.sum(self.state_op.coeff_vec))
        return QuantumState(self.state_matrix, coeff_vector, vec_type=self.vec_type)
        
    @cached_property
    def dagger(self) -> "QuantumState":
        """
        Returns:
            conj_state (QuantumState): The Hermitian conjugated state i.e. bra -> ket, ket -> bra.
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
            sparse_Qstate (csr_matrix): sparse matrix representation of the statevector.
        """
        # nonzero_indices = [int(''.join([str(i) for i in row]),2) for row in self.state_matrix]
        # if self.n_qubits<64:
        #     nonzero_indices = self.state_matrix @ (1 << np.arange(self.state_matrix.shape[1])[::-1])
        # else:
        #     nonzero_indices = self.state_matrix @ (1 << np.arange(self.state_matrix.shape[1], dtype=object)[::-1])
        nonzero_indices = binary_array_to_int(self.state_matrix)

        sparse_Qstate = csr_matrix(
            (self.state_op.coeff_vec, (nonzero_indices, np.zeros_like(nonzero_indices))),
            shape = (2**self.n_qubits, 1), 
            dtype=np.complex128
        )
        if self.vec_type == 'bra':
            # conjugate has already taken place, just need to make into row vector
            sparse_Qstate= sparse_Qstate.reshape([1,-1])
        return sparse_Qstate
    
    @cached_property
    def to_dense_matrix(self):
        """
        Returns:
            dense_Qstate (ndarray): dense matrix representation of the statevector
        """
        return self.to_sparse_matrix.toarray()
    
    def partial_trace_over_qubits(self, qubits: List[int] = []) -> np.ndarray:
        """
        Perform a partial trace over the specified qubit positions, 
        yielding the reduced density matrix of the remaining subsystem.

        Args:
            qubits (List[int]): qubit indicies to trace over

        Returns:
            rho_reduced (ndarray): Reduced density matrix over the remaining subsystem
        """
        rho_reduced = self.to_dense_matrix.reshape([2]*self.n_qubits)
        rho_reduced = np.tensordot(rho_reduced, rho_reduced.conj(), axes=(qubits, qubits))
        d = int(np.sqrt(np.product(rho_reduced.shape)))
        return rho_reduced.reshape(d, d)

    def get_rdm(self, qubits: List[int] = []) -> np.ndarray:
        """
        Return the reduced density matrix of the specified qubit positions, 
        corresponding with a partial trace over the complementary qubit indices
        
        Args:
            qubits (List[int]): qubit indicies to preserve

        Returns:
            rho_reduced (ndarray): Reduced density matrix over the chosen subsystem
        """
        trace_over_indices = list(set(range(self.n_qubits)).difference(set(qubits)))
        rho_reduced = self.partial_trace_over_qubits(trace_over_indices)
        return rho_reduced

    def _is_normalized(self) -> bool:
        """
        Check if state is normalized.

        Returns:
            bool: True or False depending on if state is normalized.

        """
        # clean-up needed to avoid duplicated terms messing with normalization calculation
        if not np.isclose(np.linalg.norm(self.state_op.cleanup().coeff_vec), 1):
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
        """ 
        Returns:
            dict: The QuantumState represented as a dictionary.
        """
        state_to_convert = self.cleanup()
        state_dict = dict(
            zip(
                [''.join([str(i) for i in row]) for row in state_to_convert.state_matrix], 
                state_to_convert.state_op.coeff_vec
            )
        )
        return state_dict

    @classmethod
    def from_dictionary(cls, 
            state_dict: Dict[str, Union[complex, Tuple[float, float]]]
        ) -> "QuantumState":
        """ 
        Initialize a QuantumState from a dictionary of the form {'1101':a, '0110':b, '1010':c, ...}. This is useful for
        converting the measurement output of a quantum circuit to a QuantumState object for further manipulation/bootstrapping.

        Args:
            state_dict (Dict[str, Union[complex, Tuple[float, float]]]): The dictionary representation of the QuantumState.

        Returns:
            QuantumState: The initialized QuantumState object.
        """
        bin_strings, coeff_vector = zip(*state_dict.items())

        coeff_vector = np.array(coeff_vector)
        if len(coeff_vector.shape)==2:
            # if coeff_vec supplied as list of tuples (real, imag) then converts to single complex vector
            assert(coeff_vector.shape[1]==2), 'Only tuples of size two allowed (real and imaginary components)'
            coeff_vector = coeff_vector[:,0] + 1j*coeff_vector[:,1]

        coeff_vector = np.array(coeff_vector)
        state_matrix = np.array([[int(i) for i in bstr] for bstr in bin_strings])
        return cls(state_matrix, coeff_vector)

    @classmethod
    def from_array(cls,
            statevector: np.array,
            threshold: float =1e-15,
        ) -> "QuantumState":
        """ 
        Initialize a QubitState from a vector of 2^N elements over N qubits
        
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

    def measure_state_in_computational_basis(self, P_op: PauliwordOp) -> Tuple["QuantumState", PauliwordOp]:
        """
        Perform change of basis to measure input Pauli operator in the computational basis

        <self| P_op |self> == <psi_new_basis | Z_new | psi_new_basis>

        due to:  <self|U† U P_op U† U |self> --> <psi_new_basis | Z_new | psi_new_basis>

        where U |self> = | psi_new_basis>
        and  U P_op U† = Z_new

        Args:
            P_op (PauliwordOp): PauliwordOp to measure

        Returns:
            psi_new_basis (QuantumState): quantum state in new basis
            Z_new (PauliwordOp): operator to measure in new basis (composed of only I,Z pauli matrices)
        """
        assert self.vec_type == 'ket', 'cannot perform change of basis on bra'

        U = change_of_basis_XY_to_Z(P_op)
        Z_new = U * P_op * U.dagger
        psi_new_basis = U*self

        return psi_new_basis, Z_new

    def plot_state(self, 
            logscale:bool = False, 
            probability_threshold:float=None,
            binary_xlabels = False,
            dpi:int=100
        ):
        """
        Plot the probabilities of the quantum state.

        Args:
            logscale (bool): Whether to use a logarithmic scale for the y-axis.
            probability_threshold (float): Threshold for considering probabilities as zero.
            binary_xlabels (bool): Whether to use binary strings as x-axis labels.
            dpi (int): Dots per inch of the plot.

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
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

        # if q_state.state_op.n_qubits<64:
        #     x_binary_ints = q_state.state_matrix @ (1 << np.arange(q_state.state_matrix.shape[1])[::-1])
        # else:
        #     x_binary_ints = q_state.state_matrix @ (1 << np.arange(q_state.state_matrix.shape[1], dtype=object)[::-1])
        x_binary_ints = binary_array_to_int(q_state.state_matrix)

        if prob.shape[0]<2**8:
            # bar chart
            ax.bar(x_binary_ints, prob, width=1, edgecolor="white", linewidth=0.8)
            if binary_xlabels:
                ax.set_xticks(x_binary_ints, labels=[np.binary_repr(x, self.n_qubits) for x in x_binary_ints])
                plt.xticks(rotation = 90)
            else:
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

def get_ij_operator(i:int, j:int, n_qubits:int,
                    binary_vec:np.ndarray=None,
                    return_operator:bool=True) -> Union["PauliwordOp", Tuple[np.ndarray, np.ndarray]]:
    """
    Get the Pauli operator for the projector: |i> <j|

    Args:
        i (int): ket of projector
        j (int): bra of projector
        n_qubits (int): number of qubits
        binary_vec (optional): bool array of all bitstrings on n_qubits
        return_operator (bool): whether to return PauliWordOp or a the symplectic matrix and coeff_vec
    
    Returns:
        Union["PauliwordOp", Tuple[np.ndarray, np.ndarray]]: Pauli operator or the symplectic matrix and coeff_vec
    """
    if n_qubits > 30:
        raise ValueError('Too many qubits, might run into memory limitations.')

    if binary_vec is None:
        binary_vec = (
                ((np.arange(2 ** n_qubits).reshape([-1, 1]) &
                  (1 << np.arange(n_qubits))[::-1])) > 0
        ).astype(bool)


    #### LONG form below
    # left = binary_vec[i]
    # right = binary_vec[j]

    # AND = left & right  # AND where -1 sign
    # XZX_sign_flips = (-1) ** np.sum(AND & binary_vec, axis=1)  # XZX = -X multiplications
    #
    # if i != j:
    #     XOR = left ^ right  # XOR where +-i phase
    #
    #     XZ_mult = left & binary_vec
    #     ZX_mult = binary_vec & right
    #
    #     XZ_phase = (-1j) ** np.sum(XZ_mult & ~ZX_mult, axis=1)  # XZ=-iY multiplications
    #     ZX_phase = (+1j) ** np.sum(ZX_mult & ~XZ_mult, axis=1)  # ZX=+iY multiplications
    #     phase_mod = XZX_sign_flips * XZ_phase * ZX_phase
    #
    #     ij_symp_matrix = np.hstack([np.tile(XOR, [2 ** n_qubits, 1]), binary_vec])
    #     coeffs = phase_mod / 2 ** n_qubits
    #
    #     if return_operator:
    #         ij_operator = PauliwordOp(ij_symp_matrix, phase_mod / 2 ** n_qubits)
    #         return ij_operator
    # else:
    #     ij_symp_matrix = np.hstack([np.zeros_like(binary_vec), binary_vec])
    #     coeffs = XZX_sign_flips / 2 ** n_qubits
    #
    #     if return_operator:
    #         ij_operator = PauliwordOp(ij_symp_matrix, XZX_sign_flips / 2 ** n_qubits)
    #         return ij_operator


    if i != j:
        coeffs = (((-1) ** np.sum(np.logical_and(binary_vec[i], binary_vec[j]) & binary_vec, axis=1))
               * ((-1j) ** np.sum((binary_vec[i] & binary_vec) & ~(binary_vec & binary_vec[j]), axis=1))
               * ((+1j) ** np.sum((binary_vec & binary_vec[j]) & ~(binary_vec[i] & binary_vec), axis=1))) / 2 ** n_qubits

        # # use broadcasting over tile
        # ij_symp_matrix = np.hstack(((binary_vec[i] ^ binary_vec[j]) * np.ones([2 ** n_qubits, n_qubits], dtype=bool),
        #
        #                              binary_vec))
        # ij_symp_matrix = np.hstack([np.repeat((binary_vec[i] ^ binary_vec[j])[np.newaxis, :], repeats=2**n_qubits, axis=0)
        #                            , binary_vec])

        ij_symp_matrix = np.hstack([np.tile((binary_vec[i] ^ binary_vec[j]),[2 ** n_qubits, 1]),
                                    binary_vec])

    else:
        ij_symp_matrix = np.hstack([np.zeros_like(binary_vec), binary_vec])
        coeffs = ((-1) ** np.sum(np.logical_and(binary_vec[i], binary_vec[j]) & binary_vec, axis=1)) / 2 ** n_qubits

    if return_operator:
        ij_operator = PauliwordOp(ij_symp_matrix, coeffs)
        return ij_operator
    else:
        return ij_symp_matrix, coeffs


def single_term_expval(P_op: PauliwordOp, psi: QuantumState) -> float:
    """ 
    Expectation value calculation for a single Pauli operator given a QuantumState psi

    Scales linearly in the number of basis states of psi, versus the quadratic cost of
    evaluating <psi|P|psi> directly, taking into consideration all of the cross-terms.

    Works by decomposing P = P(+) - P(-) where P(±) = (I±P)/2 projects onto the ±1-eigensapce of P. 

    Args:
        P_op (PauliwordOp): The Pauli operator for which to calculate the expectation value.
        psi (QuantumState): The quantum state on which to calculate the expectation value.

    Returns:
        float: The expectation value of the Pauli operator.

    Raises:
        AssertionError: If the supplied Pauli operator has multiple terms.
    """
    assert P_op.n_terms == 1, 'Supplied multiple Pauli terms.'
    
    # symplectic form of the projection operator
    proj_symplectic = np.vstack([np.zeros(P_op.n_qubits*2, dtype=bool), P_op.symp_matrix])

    # function that applies the projector onto the ±1 eigenspace of P
    # (given by the operator (I±P)/2) and returns norm of the resulting state
    norm_ev = lambda ev:np.linalg.norm( 
        ( 
            PauliwordOp(proj_symplectic, [.5,.5*ev]) * psi
        ).state_op.coeff_vec
    )
    # difference of norms provides a metric for which eigenvalue is dominant within
    # the provided reference state (e.g. if inputting a ±1 eigenvector then diff=±1)
    return (norm_ev(+1)**2 - norm_ev(-1)**2).real


def change_of_basis_XY_to_Z(P_op: PauliwordOp) -> PauliwordOp:
    """
    Get PauliwordOp representing H and Sdagger dates required to measure a PauliwordOp in the
    computational basis

    Args:
        P_op (PauliwordOp): PauliwordOp to measure in computational basis
    Returns:
        change_basis (PauliwordOp): PauliwordOp to implement change of basis

    example:
        P_op = PauliwordOp.from_list(['XYZI'])
        U = change_of_basis_XY_to_Z(P_op)
        print(U * P_op * U.dagger)
        >>  1.000+0.000j ZZZI

    Here Udagger represents a H gate on 1st qubit and H Sdagger on second qubit

    """

    # find Y terms (to palace Sdagger gates)

    Y_inds = np.logical_and(P_op.X_block, P_op.Z_block)[0]
    n_Sdag = np.sum(Y_inds)

    if n_Sdag == 0:
        s_dag_op = PauliwordOp.from_list(['I' * P_op.n_qubits])

    else:
        Z_block = (
                (
                        np.arange(2 ** n_Sdag).reshape([-1, 1]) &
                        (1 << np.arange(n_Sdag))[::-1]
                ) > 0).astype(bool)

        zblock = np.zeros((2 ** n_Sdag, P_op.n_qubits), dtype=bool)
        zblock[:, Y_inds] = Z_block

        xblock = np.zeros((2 ** n_Sdag, P_op.n_qubits), dtype=bool)

        symp = np.hstack((xblock, zblock))
        n_Sz = np.sum(zblock, axis=1)

        s_dag_op = PauliwordOp(symp, ((1 - 1j) ** (n_Sdag - n_Sz) * (1 + 1j) ** n_Sz) / 2 ** n_Sdag)

    ### Measure XY terms (to place Hadamard gates)
    X_inds = np.logical_and(P_op.X_block, ~P_op.Z_block)[0]
    XY_inds = X_inds ^ Y_inds

    n_hadamards = np.sum(XY_inds)
    if n_hadamards == 0:
        xy_measure = PauliwordOp.from_list(['I' * P_op.n_qubits])
    else:
        constant_H = (1 / np.sqrt(2)) ** n_hadamards * np.ones(2 ** n_hadamards)

        X_block = (
                (
                        np.arange(2 ** n_hadamards).reshape([-1, 1]) &
                        (1 << np.arange(n_hadamards))[::-1]
                ) > 0).astype(bool)

        xblock = np.zeros((2 ** n_hadamards, P_op.n_qubits), dtype=bool)
        xblock[:, XY_inds] = X_block

        zblock = np.zeros((2 ** n_hadamards, P_op.n_qubits), dtype=bool)
        zblock[:, XY_inds] = ~X_block

        symp = np.hstack((xblock, zblock))

        # operator represents apply H gate on X qubit indices
        xy_measure = PauliwordOp(symp, constant_H)

    ## change of basis
    change_basis = xy_measure * s_dag_op
    del xy_measure, s_dag_op

    return change_basis
