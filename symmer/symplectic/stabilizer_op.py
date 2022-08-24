import numpy as np
from typing import Dict, List, Tuple, Union
from functools import reduce
from cached_property import cached_property
from symmer.utils import gf2_gaus_elim, gf2_basis_for_gf2_rref
from symmer.symplectic import PauliwordOp, ObservableGraph, symplectic_to_string

def find_symmetry_basis(operator, commuting_override=False):
    """ Find an independent symmetry basis for the input operator,
    i.e. a basis that commutes universally within the operator
    """
    # swap order of XZ blocks in symplectic matrix to ZX
    ZX_symp = np.hstack([operator.Z_block, operator.X_block])
    reduced = gf2_gaus_elim(ZX_symp)
    kernel  = gf2_basis_for_gf2_rref(reduced)
    stabilizers = ObservableGraph(kernel, np.ones(kernel.shape[0]))
    if not commuting_override and np.any(~stabilizers.adjacency_matrix):
        # if any of the stabilizers are not mutually commuting, take the largest commuting subset
        stabilizers = stabilizers.clique_cover(clique_relation='C', colouring_strategy='largest_first')[0]

    return StabilizerOp(stabilizers.symp_matrix, np.ones(stabilizers.n_terms))

class StabilizerOp(PauliwordOp):
    """ Special case of PauliwordOp, in which the operator terms must
    by algebraically independent, with all coefficients set to integers +/-1.

    - stabilizer_rotations
        This method determines a sequence of Clifford rotations mapping the
        provided stabilizers onto single-qubit Paulis (sqp), either X or Z

    <!> Note the target_sqp must be chosen BEFORE generating 
        the stabilizer rotations, since these will be cached
    """
    def __init__(self,
            operator:   Union[List[str], Dict[str, float], np.array],
            coeff_vec:  Union[List[complex], np.array] = None,
            target_sqp: str = 'Z'):
        """
        """
        super().__init__(operator, coeff_vec)
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

    def __str__(self) -> str:
        """ 
        Defines the print behaviour of StabilizerOp - 
        returns the operator in an easily readable format

        Returns:
            out_string (str): human-readable StabilizerOp string
        """
        out_string = ''
        for pauli_vec, coeff in zip(self.symp_matrix, self.coeff_vec):
            p_string = symplectic_to_string(pauli_vec)
            out_string += (f'{coeff: d} {p_string} \n')
        return out_string[:-2]

    def __repr__(self):
        return str(self)
    
    def __add__(self, Pword: "PauliwordOp") -> "PauliwordOp":
        summed = super().__add__(Pword)
        return StabilizerOp(summed.symp_matrix, summed.coeff_vec)

    def _rotate_by_single_Pword(self, 
            Pword: "PauliwordOp", 
            angle: float = None
        ) -> "PauliwordOp":
        rotated_stabilizers = super()._rotate_by_single_Pword(Pword, angle)
        return StabilizerOp(rotated_stabilizers.symp_matrix, rotated_stabilizers.coeff_vec)

    def perform_rotations(self, 
        rotations: List[Tuple["PauliwordOp", float]]
        ) -> "PauliwordOp":
        """ Overwrite PauliwordOp.perform_rotations to return a StabilizerOp
        """
        rotated_stabilizers = super().perform_rotations(rotations)
        return StabilizerOp(rotated_stabilizers.symp_matrix, rotated_stabilizers.coeff_vec)

    def _recursive_rotations(self, basis: "StabilizerOp"):
        """ Recursively rotate terms of the StabilizerOp to single-qubit Pauli operators.
        This is only possible when the basis is mutually commuting! Else, such rotations do
        not exist (there is a check for this in generate_stabilizer_rotations, that wraps this method).
        """
        # drop any term(s) that are single-qubit Pauli operators
        non_sqp = np.where(np.einsum('ij->i', basis.symp_matrix)!=1)
        basis_non_sqp = StabilizerOp(basis.symp_matrix[non_sqp], basis.coeff_vec[non_sqp])
        sqp_indices = np.where((basis - basis_non_sqp).symp_matrix)[1]%self.n_qubits
        self.used_indices += np.append(sqp_indices, sqp_indices+self.n_qubits).tolist()
    
        if basis_non_sqp.n_terms == 0:
            # once the basis has been fully rotated onto single-qubit Paulis, return the rotations
            return None
        else:
            # identify the lowest-weight Pauli operator from the commuting basis
            row_sum = np.einsum('ij->i',basis_non_sqp.symp_matrix)
            sort_rows_by_weight = np.argsort(row_sum)
            pivot_row = basis_non_sqp.symp_matrix[sort_rows_by_weight][0]
            non_I = np.setdiff1d(np.where(pivot_row)[0], np.array(self.used_indices))
            # once a Pauli operator has been selected, the least-supported qubit is chosen as pivot
            col_sum = np.einsum('ij->j',basis_non_sqp.symp_matrix)
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
        
    def generate_stabilizer_rotations(self):
        """ Find the full list of pi/2 Pauli rotations (Clifford operations) mapping this StabilizerOp 
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
            ref_state: Union[List[int], np.array]
        ) -> None:
        """ Given the specified reference state, e.g. Hartree-Fock |1...10...0>, 
        determine the corresponding sector by measuring the stabilizers

        TODO: currently only measures in Z basis
        """
        ref_state = np.array(ref_state)
        self.coeff_vec = (-1)**np.count_nonzero(self.Z_block & ref_state, axis=1)

    def rotate_onto_single_qubit_paulis(self) -> "StabilizerOp":
        """ Returns the rotated single-qubit Pauli stabilizers
        """
        self.generate_stabilizer_rotations()
        if self.stabilizer_rotations != []:
            return self.perform_rotations(self.stabilizer_rotations)
        else:
            return self