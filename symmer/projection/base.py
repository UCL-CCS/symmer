import numpy as np
from typing import List, Tuple, Union
from symmer.symplectic import PauliwordOp, StabilizerOp

class S3_projection:
    """ Base class for enabling qubit reduction techniques derived from
    the Stabilizer SubSpace (S3) projection framework, such as tapering
    and Contextual-Subspace VQE. The methods defined herein serve the 
    following purposes:

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
                stabilizers: StabilizerOp
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
        self.stabilizers = stabilizers
        
    def _perform_projection(self, 
            operator: PauliwordOp,
            #sym_sector: Union[List[int], np.array]
        ) -> PauliwordOp:
        """ method for projecting an operator over fixed qubit positions 
        stabilized by single Pauli operators (obtained via Clifford operations)
        """
        assert(operator.n_qubits == self.stabilizers.n_qubits), 'The input operator does not have the same number of qubits as the stabilizers'
        assert(self.rotated_flag), 'The operator has not been rotated - intended for use with perform_projection method'
        self.rotated_flag = False
        
        # remove terms that do not commute with the rotated stabilizers
        commutes_with_all_stabilizers = np.all(operator.commutes_termwise(self.rotated_stabilizers), axis=1)
        op_anticommuting_removed = operator.symp_matrix[commutes_with_all_stabilizers]
        cf_anticommuting_removed = operator.coeff_vec[commutes_with_all_stabilizers]

        # determine sign flipping from eigenvalue assignment
        # currently ill-defined for single-qubit Y stabilizers
        stab_symp_indices  = np.where(self.rotated_stabilizers.symp_matrix)[1]
        eigval_assignment = op_anticommuting_removed[:,stab_symp_indices]*self.rotated_stabilizers.coeff_vec
        eigval_assignment[eigval_assignment==0]=1 # 0 entries are identity, so fix as 1 in product
        coeff_sign_flip = cf_anticommuting_removed*(np.prod(eigval_assignment, axis=1)).T

        # the projected Pauli terms:
        unfixed_XZ_indices = np.hstack([self.free_qubit_indices,
                                        self.free_qubit_indices+operator.n_qubits])
        projected_symplectic = op_anticommuting_removed[:,unfixed_XZ_indices]

        # there may be duplicate rows in op_projected - these are identified and
        # the corresponding coefficients collected in the cleanup method
        if projected_symplectic.shape[1]:
            return PauliwordOp(projected_symplectic, coeff_sign_flip).cleanup()
        else:
            return PauliwordOp(np.array([], dtype=bool), [np.sum(coeff_sign_flip)])
            
    def perform_projection(self,
            operator: PauliwordOp,
            ref_state: Union[List[int], np.array]=None,
            sector: Union[List[int], np.array]=None
        ) -> PauliwordOp:
        """ Input a PauliwordOp and returns the reduced operator corresponding 
        with the specified stabilizers and eigenvalues.
        
        insert_rotation allows one to include supplementary Pauli rotations
        to be performed prior to the stabilizer rotations, for example 
        unitary partitioning in CS-VQE
        """
        if sector is None and ref_state is not None:
            #assert(ref_state is not None), 'If no sector is provided then a reference state must be given instead'
            self.stabilizers.update_sector(ref_state)
        elif sector is not None:
            self.stabilizers.coeff_vec = np.array(sector, dtype=int)

        self.rotated_stabilizers = self.stabilizers.rotate_onto_single_qubit_paulis()
        self.stab_qubit_indices  = np.where(self.rotated_stabilizers.symp_matrix)[1] % operator.n_qubits
        self.free_qubit_indices  = np.setdiff1d(np.arange(operator.n_qubits),self.stab_qubit_indices)

        # perform the full list of rotations on the input operator...
        if len(self.stabilizers.stabilizer_rotations) > 0:
            op_rotated = operator.perform_rotations(self.stabilizers.stabilizer_rotations)
        else:
            op_rotated = operator
        
        self.rotated_flag = True
        # ...and finally perform the stabilizer subspace projection
        return self._perform_projection(operator=op_rotated)