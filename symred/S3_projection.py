from functools import cached_property
from shutil import ExecError
from symred.symplectic_form import PauliwordOp, symplectic_to_string
from typing import Dict, List, Tuple, Union
import numpy as np

def gf2_gaus_elim(gf2_matrix: np.array) -> np.array:
    """
    Function that performs Gaussian elimination over GF2(2)
    GF is the initialism of Galois field, another name for finite fields.

    GF(2) may be identified with the two possible values of a bit and to the boolean values true and false.

    pseudocode: http://dde.binghamton.edu/filler/mct/hw/1/assignment.pdf

    Args:
        gf2_matrix (np.array): GF(2) binary matrix to preform Gaussian elimination over
    Returns:
        gf2_matrix_rref (np.array): reduced row echelon form of M
    """
    gf2_matrix_rref = gf2_matrix.copy()
    m_rows, n_cols = gf2_matrix_rref.shape

    row_i = 0
    col_j = 0

    while row_i < m_rows and col_j < n_cols:

        if sum(gf2_matrix_rref[row_i:, col_j]) == 0:
            # case when col_j all zeros
            # No pivot in this column, pass to next column
            col_j += 1
            continue

        # find index of row with first "1" in the vector defined by column j (note previous if statement removes all zero column)
        k = np.argmax(gf2_matrix_rref[row_i:, col_j]) + row_i
        # + row_i gives correct index (as we start search from row_i!)

        # swap row k and row_i (row_i now has 1 at top of column j... aka: gf2_matrix_rref[row_i, col_j]==1)
        gf2_matrix_rref[[k, row_i]] = gf2_matrix_rref[[row_i, k]]
        # next need to zero out all other ones present in column j (apart from on the i_row!)
        # to do this use row_i and use modulo addition to zero other columns!

        # make a copy of j_th column of gf2_matrix_rref, this includes all rows (0 -> M)
        Om_j = np.copy(gf2_matrix_rref[:, col_j])

        # zero out the i^th position of vector Om_j (this is why copy needed... to stop it affecting gf2_matrix_rref)
        Om_j[row_i] = 0
        # note this was orginally 1 by definition...
        # This vector now defines the indices of the rows we need to zero out
        # by setting ith position to zero - it stops the next steps zeroing out the i^th row (which we need as our pivot)


        # next from row_i of rref matrix take all columns from j->n (j to last column)
        # this is vector of zero and ones from row_i of gf2_matrix_rref
        i_jn = gf2_matrix_rref[row_i, col_j:]
        # we use i_jn to zero out the rows in gf2_matrix_rref[:, col_j:] that have leading one (apart from row_i!)
        # which rows are these? They are defined by that Om_j vector!

        # the matrix to zero out these rows is simply defined by the outer product of Om_j and i_jn
        # this creates a matrix of rows of i_jn terms where Om_j=1 otherwise rows of zeros (where Om_j=0)
        Om_j_dependent_rows_flip = np.einsum('i,j->ij', Om_j, i_jn, optimize=True)
        # note flip matrix is contains all m rows ,but only j->n columns!

        # perfrom bitwise xor of flip matrix to zero out rows in col_j that that contain a leading '1' (apart from row i)
        gf2_matrix_rref[:, col_j:] = np.bitwise_xor(gf2_matrix_rref[:, col_j:], Om_j_dependent_rows_flip)

        row_i += 1
        col_j += 1

    return gf2_matrix_rref


def gf2_basis_for_gf2_rref(gf2_matrix_in_rreform: np.array) -> np.array:
    """
    Function that gets the kernel over GF2(2) of ow reduced  gf2 matrix!

    uses method in: https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Basis

    Args:
        gf2_matrix_in_rreform (np.array): GF(2) matrix in row reduced form
    Returns:
        basis (np.array): basis for gf2 input matrix that was in row reduced form
    """
    rows_to_columns = gf2_matrix_in_rreform.T
    eye = np.eye(gf2_matrix_in_rreform.shape[1], dtype=int)

    # do column reduced form as row reduced form
    rrf = gf2_gaus_elim(np.hstack((rows_to_columns, eye.T)))

    zero_rrf = np.where(~rrf[:, :gf2_matrix_in_rreform.shape[0]].any(axis=1))[0]
    basis = rrf[zero_rrf, gf2_matrix_in_rreform.shape[0]:]

    return basis


class StabilizerOp(PauliwordOp):
    def __init__(self,
                 operator:   Union[List[str], Dict[str, float], np.array],
                 coeff_list: Union[List[complex], np.array] = None):
        super().__init__(operator, coeff_list)
        self._check_stab()
        self._check_independent()

    def _check_stab(self):
        """ Checks the coefficient is +1
        """
        assert(set(self.coeff_vec).issubset([+1])), 'Stabilizer coefficients not +1'

    def _check_independent(self):
        """ Check the supplied stabilizers are algebraically independent
        """
        check_independent = gf2_gaus_elim(self.symp_matrix)
        for row in check_independent:
            if np.all(row==0):
                # there is a dependent row
                raise ValueError('The supplied stabilizers are not independent')



class S3_projection:
    """ Base class for enabling qubit reduction techniques derived from
    the Stabilizer SubSpace (S3) projection framework, such as tapering
    and Contextual-Subspace VQE. The methods defined herein serve the 
    following purposes:

    - stabilizer_rotations
        This method determines a sequence of Clifford rotations mapping the
        provided stabilizers onto single-qubit Paulis (sqp), either X or Z
    - _perform_projection
        Assuming the input operator has been rotated via the Clifford operations 
        found in the above stabilizer_rotations method, this will effect the 
        projection onto the corresponding stabilizer subspace. This involves
        droping any operator terms that do not commute with the rotated generators
        and fixing the eigenvalues of those that do consistently.
    - perform_projection
        This method wraps _perform_projection but provides the facility to insert
        auxiliary rotations (that need not be Clifford). This is used in CS-VQE
        to implement unitary partitioning where necessary. 
    """
    
    rotated_flag = False

    def __init__(self,
                stabilizers: "StabilizerOp", 
                target_sqp: str,
                fix_qubits: List[int] = None
                ) -> None:
        """
        - stabilizers
            a list of stabilizers that should be enforced, given as Pauli strings
        - eigenvalues
            the list of eigenvalue assignments to complement the stabilizers
        - target_sqp
            the target single-qubit Pauli (X or Z) that we wish to rotate onto
        - fix_qubits
            Manually overrides the qubit positions selected in stabilizer_rotations, 
            although the rotation procedure can be a bit unpredictable so take care!
        """
        # store stabilizers and their assignments as PauliwordOp object
        # this facilitates various manipulations such as Pauli rotations
      
        self.stabilizers = stabilizers
        self.target_sqp= target_sqp
        self.rotations, self.angles = zip(*self.stabilizer_rotations)
        self.rotated_stabilizers = self.stabilizers.recursive_rotate_by_Pword(
            pauli_rot_list=self.rotations,
            angles=self.angles
        )
        
        if fix_qubits is None:
            fix_qubits = [None for S in range(self.stabilizers.n_terms)]
        self.fix_qubits = fix_qubits
    
    
    @cached_property
    def stabilizer_rotations(self):
        """ 
        Implementation of procedure described in https://doi.org/10.22331/q-2021-05-14-456 (Lemma A.2)
        
        Returns 
        - a dictionary of stabilizers with the rotations mapping each to a 
          single Pauli in the formList[Tuple[rotation, angle, gen_rot]], 
        
        - a dictionary of qubit positions that we have rotated onto and 
          the eigenvalues post-rotation
        """
        stabilizer_ref = self.stabilizers.copy()
        rotations=[]

        def append_rotation(base_pauli: np.array, index: int) -> str:
            """ force the indexed qubit to a Pauli Y in the base Pauli
            """
            X_index = index % self.stabilizers.n_qubits # index in the X block
            base_pauli[np.array([X_index, X_index+self.stabilizers.n_qubits])]=1
            base_pauli = symplectic_to_string(base_pauli)
            # None angle defaults to pi/2 for Clifford rotation
            rotations.append((base_pauli, None))
            # return the pauli rotation to update stabilizer_ref as we go
            return base_pauli

        # This part produces rotations onto single-qubit Paulis (sqp) - might be a combination of X and Z
        # while loop active until each row of symplectic matrix contains a single non-zero element
        while np.any(~(np.count_nonzero(stabilizer_ref.symp_matrix, axis=1)==1)):
            unique_position = np.where(np.count_nonzero(stabilizer_ref.symp_matrix, axis=0)==1)[0]
            reduced = stabilizer_ref.symp_matrix[:,unique_position]
            unique_stabilizer = np.where(np.any(reduced, axis=1))
            for row in stabilizer_ref.symp_matrix[unique_stabilizer]:
                if np.count_nonzero(row) != 1:
                    # find the free indices and pick one (there is some freedom over this)
                    available_positions = np.intersect1d(unique_position, np.where(row))
                    pauli_rotation = PauliwordOp([append_rotation(row.copy(), available_positions[0])], [1])
                    # update the stabilizers by performing the rotation
                    stabilizer_ref = stabilizer_ref._rotate_by_single_Pword(pauli_rotation)

        # This part produces rotations onto the target sqp
        for row in stabilizer_ref.symp_matrix:
            sqp_index = np.where(row)[0]
            if ((self.target_sqp == 'Z' and sqp_index< self.stabilizers.n_qubits) or 
                (self.target_sqp == 'X' and sqp_index>=self.stabilizers.n_qubits)):
                pauli_rotation = append_rotation(np.zeros(2*self.stabilizers.n_qubits, dtype=int), sqp_index)

        return rotations


    def _perform_projection(self, 
                        operator: PauliwordOp,
                        sym_sector: List[int]
                        ) -> Dict[str, float]:
        """ method for projecting an operator over fixed qubit positions 
        stabilized by single Pauli operators (obtained via Clifford operations)
        """
        if not self.rotated_flag:
            raise ExecError('The operator has not been rotated - intended for use with perform_projection method')
        
        # overwrite the coefficient vector to the assigned eigenvalues defined by the symmetry sector
        rotated_stabilizers = self.rotated_stabilizers.copy()
        rotated_stabilizers.coeff_vec*=np.array(sym_sector)
        stab_positions = np.einsum("ij->j",rotated_stabilizers.symp_matrix)
        stab_q_indices = np.where(stab_positions)[0]
        assert(len(stab_q_indices)== rotated_stabilizers.n_terms), 'unique indices and stabilizers do not match'

        # remove terms that do not commute with the rotated stabilizers
        commutes_with_all_stabilizers = np.all(operator.commutes_termwise(rotated_stabilizers), axis=1)
        op_anticommuting_removed = operator.symp_matrix[commutes_with_all_stabilizers]
        cf_anticommuting_removed = operator.coeff_vec[commutes_with_all_stabilizers]

        # determine sign flipping from eigenvalue assignment
        # currently ill-defined for single-qubit Y stabilizers
        eigval_assignment = op_anticommuting_removed[:,stab_q_indices]*rotated_stabilizers.coeff_vec
        eigval_assignment[eigval_assignment==0]=1 # 0 entries are identity, so fix as 1 in product
        coeff_sign_flip = cf_anticommuting_removed*(np.prod(eigval_assignment, axis=1)).T

        # the projected Pauli terms:
        all_qubits = np.arange(operator.n_qubits)
        unfixed_positions = np.setdiff1d(all_qubits,stab_q_indices % operator.n_qubits)
        unfixed_positions = np.hstack([ unfixed_positions,
                                        unfixed_positions+operator.n_qubits])
        op_projected = op_anticommuting_removed[:,unfixed_positions]


        # there may be duplicate rows in op_projected - these are identified and
        # the corresponding coefficients collected in cleanup_symplectic function
        projected_operator = PauliwordOp(op_projected, coeff_sign_flip)

        return projected_operator.cleanup() 
        
    
    def perform_projection(self,
                    operator: PauliwordOp,
                    sym_sector: List[int],
                    insert_rotation:Tuple[str,float]=None
                    )->Dict[float, str]:
        """ Input a PauliwordOp and returns the reduced operator corresponding 
        with the specified stabilizers and eigenvalues.
        
        insert_rotation allows one to include supplementary Pauli rotations
        to be performed prior to the stabilizer rotations, for example 
        unitary partitioning in CS-VQE
        """
        stab_rotations = self.rotations
        angles = self.angles
        # ...and insert any supplementary ones coming from the child class
        if insert_rotation is not None:
            stab_rotations.insert(0, insert_rotation[0])
            angles.insert(0, insert_rotation[1])

        # perform the full list of rotations on the input operator...
        op_rotated = operator.recursive_rotate_by_Pword(pauli_rot_list=stab_rotations, angles=angles)
        self.rotated_flag = True
        # ...and finally perform the stabilizer subspace projection
        op_project = self._perform_projection(operator=op_rotated, sym_sector=sym_sector)

        return op_project


class taper(S3_projection):
    """
    """
    def __init__():
        pass

    @cached_property
    def symmetry_generators(self):
        """ Find an independent basis for the Hamiltonian symmetry
        This is carried out in the symplectic representation.
        """
        # swap order of XZ blocks in symplectic matrix to ZX
        ZX_symp = np.hstack([self.stabilizers.Z_block, self.stabilizers.X_block])
        reduced = gf2_gaus_elim(ZX_symp)
        kernel  = gf2_basis_for_gf2_rref(reduced)

        return PauliwordOp(kernel, np.ones(kernel.shape[0]))