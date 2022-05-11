import numpy as np
from typing import Dict, List, Tuple, Union
from functools import reduce
from cached_property import cached_property
from symred.utils import gf2_gaus_elim, gf2_basis_for_gf2_rref
from symred.symplectic import PauliwordOp, ObservableGraph, symplectic_to_string

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
        self._check_independent()
        self.coeff_vec = self.coeff_vec.astype(int)
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

            rotations.append((PauliwordOp(np.array(base_vector), [1]), None))
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
            rotated_op = self.perform_rotations(rotations)
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
        """
        ref_state = np.array(ref_state)
        self.coeff_vec = (-1)**np.count_nonzero(self.Z_block & ref_state, axis=1)

    def rotate_onto_single_qubit_paulis(self) -> "StabilizerOp":
        """ Returns the rotated single-qubit Pauli stabilizers
        """
        if self.stabilizer_rotations != []:
            return self.perform_rotations(self.stabilizer_rotations)
        else:
            return self