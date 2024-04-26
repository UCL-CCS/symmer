import numpy as np
import warnings
from functools import reduce
from typing import Dict, List, Tuple, Union
from symmer import process
from symmer.operators.utils import _rref_binary, _cref_binary, check_independent
from symmer.operators import PauliwordOp, QuantumState, symplectic_to_string, single_term_expval

class IndependentOp(PauliwordOp):
    """ 
    Special case of PauliwordOp, in which the operator terms must
    by algebraically independent, with all coefficients set to integers +/-1.

    - stabilizer_rotations
        This method determines a sequence of Clifford rotations mapping the
        provided stabilizers onto single-qubit Paulis (sqp), either X or Z

    <!> Note the target_sqp must be chosen BEFORE generating 
        the stabilizer rotations, since these will be cached.
    """
    def __init__(self,
            symp_matrix:   np.array,
            coeff_vec:  Union[List[complex], np.array] = None,
            target_sqp: str = 'Z'):
        """
        Args:
            symp_matrix (np.array): Symplectic matrix.
            coeff_vec (np.array): Coefficient Vector. By default, it's set to 'None'.
            target_sqp (str): Target SQP (Single-Qubit Pauli). By default it is to 'Z'.
        """
        if coeff_vec is None:
            coeff_vec = np.ones(symp_matrix.shape[0], dtype=complex)
        super().__init__(symp_matrix, coeff_vec)
        self._check_stab()
        self.coeff_vec = self.coeff_vec.real.astype(int)
        self._check_independent()
        if target_sqp in ['X', 'Z', 'Y']:
            self.target_sqp = target_sqp
        else:
            raise ValueError('Target single-qubit Pauli not recognised - must be X or Z')
        # set up these attributes to later track rotations mapping stabilizers to single-qubit Pauli operators
        self.stabilizer_rotations = None
        self.used_indices = None

    @classmethod
    def from_PauliwordOp(cls,
            PwordOp: PauliwordOp
        ) -> "IndependentOp":
        """
        Args:
            PwordOp (PauliwordOp): Pauli operator to be used to form the Independent Operator.

        Return:
            Independent Operator
        """
        return cls(PwordOp.symp_matrix, PwordOp.coeff_vec)

    @classmethod
    def from_list(cls, 
            pauli_terms :List[str], 
            coeff_vec:   List[complex] = None
        ) -> "IndependentOp":
        """
        Args:
            pauli_terms (List[str]): List of Pauli Terms.
            coeff_vec (List[complex]): List of complex coeffecients.

        Return:
            Independent Operator
        """
        PwordOp = super().from_list(pauli_terms, coeff_vec)
        return cls.from_PauliwordOp(PwordOp)

    @classmethod
    def from_dictionary(cls,
            operator_dict: Dict[str, complex]
        ) -> "IndependentOp":
        """ 
        Initialize a PauliwordOp from its dictionary representation {pauli:coeff, ...}

        Args:
            operator_dict (Dict[str, complex]): Dictionary representation of IndependentOp
                    
        Return:
            Independent Operator
        """
        PwordOp = super().from_dictionary(operator_dict)
        return cls.from_PauliwordOp(PwordOp)

    @classmethod
    def symmetry_generators(cls, 
            PwordOp: PauliwordOp, 
            commuting_override:bool=False,
            largest_clique = False
        ) -> "IndependentOp":
        """ 
        Identify a symmetry basis for the supplied Pauli operator with
        symplectic representation  M = [ X | Z ]. We perform columnwise 
        Gaussian elimination to yield the matrix

                [ Z | X ]     [ R ]
                |-------| ->  |---|
                [   I   ]     [ Q ]

        Indexing the zero columns of R with i, we form the matrix
        
                S^T = [ Q_i1 | ... | Q_iM ] 
                
        and conclude that S is the symplectic representation of the symmetry basis.
        This holds since MΩS^T=0 by construction, which implies commutativity.

        Since we only need to reduce columns, the algorithm scales with the number of
        qubits N, not the number of terms M, and is therefore at worst O(N^2).

        Args:
            PwordOp (PauliwordOp): Pauli operator.
            commuting_override (bool): If 'True', the check for commuting symmetries is overriden. By default, it's set to 'False'.
            largest_clique (bool): If 'True', the largest commuting subset is determined using clique cover or a greedy method. By default, it's set to 'False'.

        Returns:
            Symplectic representation of the symmetry basis.
        """
        # swap order of XZ blocks in symplectic matrix to ZX
        to_reduce = np.vstack([np.hstack([PwordOp.Z_block, PwordOp.X_block]), np.eye(2*PwordOp.n_qubits, dtype=bool)])
        cref_matrix = _cref_binary(to_reduce)
        S_symp = cref_matrix[PwordOp.n_terms:,np.all(~cref_matrix[:PwordOp.n_terms], axis=0)].T
        S = cls(S_symp, np.ones(S_symp.shape[0]))
        if S.n_terms==0:
            warnings.warn('The input PauliwordOp has no Z2 symmetries.')
            return S
            # raise RuntimeError('The input PauliwordOp has no Z2 symmetries.')
        if np.all(S.adjacency_matrix) or commuting_override:
            return S
        else:
            # if any of the stabilizers are not mutually commuting, take the largest commuting subset
            if S.n_terms < 10 or largest_clique:
                # expensive clique cover finding optimal commuting subset
                S_commuting = S.largest_clique(edge_relation='C')    
            else:
                # greedy graph-colouring approach when symmetry basis is large
                S_commuting = S.clique_cover(edge_relation='C', strategy='independent_set')[0]
                warnings.warn('Greedy method may identify non-optimal commuting symmetry terms; might be able to taper again.')
            
            return cls(S_commuting.symp_matrix, np.ones(S_commuting.n_terms, dtype=complex))

    def _check_stab(self) -> None:
        """ 
        Checks the stabilizer coefficients are +/-1
        """
        if not set(self.coeff_vec).issubset({0, +1,-1}):
            raise ValueError(f'Stabilizer coefficients not +/-1: {self.coeff_vec}')
        
    def _check_independent(self) -> None:
        """ 
        Check the supplied stabilizers are algebraically independent
        """
        if not check_independent(self):
            # there is a dependent row
            raise ValueError('The supplied stabilizers are not independent')

    def __str__(self) -> str:
        """ 
        Defines the print behaviour of IndependentOp - 
        returns the operator in an easily readable format

        Returns:
            out_string (str): human-readable IndependentOp string
        """
        out_string = ''
        for pauli_vec, coeff in zip(self.symp_matrix, self.coeff_vec):
            p_string = symplectic_to_string(pauli_vec)
            out_string += (f'{coeff} {p_string} \n')
        return out_string[:-2]

    def __repr__(self) -> str:
        return str(self)
    
    def __add__(self, Pword: "IndependentOp") -> "IndependentOp":
        summed = super().__add__(Pword)
        return self.from_PauliwordOp(summed)

    def _rotate_by_single_Pword(self, 
            Pword: "PauliwordOp", 
            angle: float = None
        ) -> "IndependentOp":
        rotated_stabilizers = super()._rotate_by_single_Pword(Pword, angle)
        return self.from_PauliwordOp(rotated_stabilizers)

    def perform_rotations(self, 
        rotations: List[Tuple["PauliwordOp", float]]
        ) -> "PauliwordOp":
        """ 
        Overwrite PauliwordOp.perform_rotations to return a IndependentOp.

        Args:
            rotations (List[Tuple["PauliwordOp", float]]): List of (PauliwordOp, rotation).

        Returns:
            Rotated Stabilizers of PauliwordOp Instance.
        """
        rotated_stabilizers = super().perform_rotations(rotations)
        return self.from_PauliwordOp(rotated_stabilizers)

    def _recursive_rotations(self, basis: "IndependentOp") -> None:
        """ 
        Recursively rotate terms of the IndependentOp to single-qubit Pauli operators.
        This is only possible when the basis is mutually commuting! Else, such rotations do
        not exist (there is a check for this in generate_stabilizer_rotations, that wraps this method).

        Args:
            basis (IndependentOp): Basis
        """
        # drop any term(s) that are single-qubit Pauli operators
        non_sqp = np.where(np.sum(basis.symp_matrix, axis=1)!=1)
        basis_non_sqp = IndependentOp(basis.symp_matrix[non_sqp], basis.coeff_vec[non_sqp])
        sqp_indices = np.where((basis - basis_non_sqp).symp_matrix)[1]%self.n_qubits
        self.used_indices += np.append(sqp_indices, sqp_indices+self.n_qubits).tolist()
    
        if basis_non_sqp.n_terms == 0:
            # once the basis has been fully rotated onto single-qubit Paulis, return the rotations
            return None
        else:
            # identify the lowest-weight Pauli operator from the commuting basis
            row_sum = np.sum(basis_non_sqp.symp_matrix, axis=1)
            sort_rows_by_weight = np.argsort(row_sum)
            pivot_row = basis_non_sqp.symp_matrix[sort_rows_by_weight][0]
            non_I = np.setdiff1d(np.where(pivot_row)[0], np.array(self.used_indices))
            # once a Pauli operator has been selected, the least-supported qubit is chosen as pivot
            col_sum = np.sum(basis_non_sqp.symp_matrix, axis=0)
            support = pivot_row*col_sum
            pivot_point = non_I[np.argmin(support[non_I])]
            # define (in the symplectic form) the single-qubit Pauli we aim to rotate onto
            target = np.zeros(2*self.n_qubits, dtype=int)
            target[pivot_point+self.n_qubits*(-1)**(pivot_point//self.n_qubits)]=1
            # the rotation mapping onto the target Pauli is given by (target + pivot_row)%2...
            # this is identicial to performing a bitwise XOR operation
            pivot_rotation = PauliwordOp(np.bitwise_xor(target, pivot_row), [1])
            self.stabilizer_rotations.append((pivot_rotation, None))
            # perform the rotation on the full basis (the ordering of rotations is important because of this!)
            rotated_basis = basis_non_sqp._rotate_by_single_Pword(pivot_rotation)
            return self._recursive_rotations(rotated_basis)
        
    def generate_stabilizer_rotations(self) -> None:
        """ 
        Find the full list of pi/2 Pauli rotations (Clifford operations) mapping this IndependentOp 
        to single-qubit Pauli operators, for use in stabilizer subsapce projection schemes.
        """
        assert(self.n_terms <= self.n_qubits), 'Too many terms in basis to reduce to single-qubit Paulis'
        assert(np.all(self.adjacency_matrix)), 'The basis is not commuting, hence the rotation is not possible'
        
        # ensure stabilizer_rotations and used_indices are empty before generating rotations
        self.stabilizer_rotations = []
        self.used_indices = []

        basis = self.copy()
        # generate the rotations mapping onto single-qubit Pauli operators (not necessarily the target_sqp)
        self._recursive_rotations(basis)
        rotated_basis = basis.perform_rotations(self.stabilizer_rotations)

        # now that the basis consists of single-qubit Paulis we may map to the target X,Y or Z
        for P in rotated_basis:
            # index in the X block:
            sqp_index = np.where(P.symp_matrix[0])[0][0]%self.n_qubits
            target = np.zeros(2*self.n_qubits, dtype=int)
            if self.target_sqp in ['X','Y']:
                target[sqp_index] = 1
            if self.target_sqp in ['Y','Z']:
                target[sqp_index+self.n_qubits] = 1
            R_symp = np.bitwise_xor(target, P.symp_matrix[0])
            # the rotation will be identity if already the target_sqp
            if np.any(R_symp):
                # therefore, only append nontrivial rotations
                self.stabilizer_rotations.append((PauliwordOp(R_symp, [1]),None))

    def update_sector(self, 
            ref_state: Union[List[int], np.array, QuantumState],
            threshold: float = 0.5
        ) -> None:
        """ 
        Given the specified reference state, e.g. Hartree-Fock |1...10...0>, 
        determine the corresponding sector by measuring the stabilizers.
        will also accept a superposition of basis states, in which case it will 
        identify the dominant sector therein, but note it will ascribe a zero
        assignment if there is not sufficient evidence to fix a +-1 eigenvalue.

        Args:
             ref_state (Union[List[int], np.array, QuantumState]): Reference State.
             threshold (float): Threshold Value for  expectation value. It's default value is 0.5
        """
        if not isinstance(ref_state, QuantumState):
            ref_state = QuantumState(ref_state)
        assert ref_state._is_normalized(), 'Reference state is not normalized.'
        
        ### update the stabilizers assignments in parallel
        self.coeff_vec = np.array(assign_value(self, ref_state))
        # raise a warning if any stabilizers are assigned a zero value
        if np.any(self.coeff_vec==0):
            S_zero = self[self.coeff_vec==0]; S_zero.coeff_vec[:]=1
            S_zero = list(S_zero.to_dictionary.keys())
            warnings.warn(f'The stabilizers {S_zero} were assigned zero values - bad reference state.')
        
    def rotate_onto_single_qubit_paulis(self) -> "IndependentOp":
        """ 
        Returns the rotated single-qubit Pauli stabilizers.

        Returns: 
            Rotated single-qubit Pauli stabilizers.
        """
        self.generate_stabilizer_rotations()
        if self.stabilizer_rotations != []:
            return IndependentOp.from_PauliwordOp(
                reduce(
                    lambda x,y:x.append(y), 
                    [s.perform_rotations(self.stabilizer_rotations) for s in self]
                )
            ) # ensures stabilizer ordering is correct
        else:
            return self
          
    def __getitem__(self,
                    key: Union[slice, int]
                    ) -> "IndependentOp":
        """ 
        Makes the IndependentOp subscriptable - returns a IndependentOp constructed
        from the indexed row and coefficient from the symplectic matrix.

        Args:
            key (Union[slice, int]): Key.

        Returns:
            IndependentOp constructed from the indexed row and coefficient from the symplectic matrix.
        """
        if isinstance(key, int):
            if key < 0:
                # allow negative subscript
                key += self.n_terms
            assert (key < self.n_terms), 'Index out of range'
            mask = [key]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.n_terms
            mask = np.arange(start, stop, key.step)
        elif isinstance(key, (list, np.ndarray)):
            mask = np.asarray(key)
        else:
            raise ValueError('Unrecognised input, must be an integer, slice, list or np.array')

        symp_items = self.symp_matrix[mask]
        coeff_items = self.coeff_vec[mask]
        return IndependentOp(symp_items, coeff_items)

    def __iter__(self):
        """ 
        Makes a PauliwordOp instance iterable.

        Returns:
            Iterator of PauliwordOp instance.
        """
        return iter([self[i] for i in range(self.n_terms)])

@process.parallelize
def assign_value(S: PauliwordOp, ref_state: QuantumState) -> int:
    """
    Measure expectation value of stabilizer on input reference state.
    Args: 
        S (PauliwordOp): Stabilizer.
        ref_state (QuantumState): Reference State.
        threshold (float): Threshold Value of expectation value.

    Returns:
        Expectation Value (int) of stabilizer on input reference state.
    """
    threshold = 0.5
    expval = single_term_expval(S, ref_state)
    # if this expval exceeds some predefined threshold then assign the corresponding 
    # ±1 eigenvalue. Otherwise, return 0 as insufficient evidence to fix the value.
    if abs(expval) > threshold:
        return int(np.sign(expval))
    else:
        return 0
