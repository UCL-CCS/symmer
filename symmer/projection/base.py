import numpy as np
from typing import List, Tuple, Union
from symmer.operators import PauliwordOp, IndependentOp, QuantumState
from symmer.evolution import trotter, Had
from functools import reduce

class S3Projection:
    """ 
    Base class for enabling qubit reduction techniques derived from
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

    Attributes:
        rotated_flag (bool): If True, the operator is rotated. By default it is set to 'False'.
    """
    rotated_flag = False

    def __init__(self,
                stabilizers: IndependentOp
                ) -> None:
        """
        - eigenvalues: The list of eigenvalue assignments to complement the stabilizers.
        
        - target_sqp: The target single-qubit Pauli (X or Z) that we wish to rotate onto.
        - fix_qubits: Manually overrides the qubit positions selected in stabilizer_rotations, although the rotation procedure can be a bit unpredictable so take care!

        Args:
            stabilizers (IndependentOp): A list of stabilizers that should be enforced, given as Pauli strings.
        """
        self.stabilizers = stabilizers
        
    def _perform_projection(self, 
            operator: PauliwordOp,
            #sym_sector: Union[List[int], np.array]
        ) -> PauliwordOp:
        """ 
        Method for projecting an operator over fixed qubit positions 
        stabilized by single Pauli operators (obtained via Clifford operations).

        Args:
            operator (PauliwordOp): Operator to be projected over fixed qubit positions stabilized by single Pauli operators.

        Returns:
            PauliwordOp representing the projection of input operator.
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
        """ 
        Input a PauliwordOp and returns the reduced operator corresponding 
        with the specified stabilizers and eigenvalues.
        
        insert_rotation allows one to include supplementary Pauli rotations
        to be performed prior to the stabilizer rotations, for example 
        unitary partitioning in CS-VQE.

        Args: 
            operator (PauliwordOp): Operator projected over fixed qubit positions stabilized by single Pauli operators.
            ref_state (np.array): Reference State. By default, it is set to None.
            sector (np.array): Sector. By default it is set to none. If no sector is provided then a reference state must be given instead.
        Returns:
            Reduced operator corresponding to the given stabilizers and eigenvalues.
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
    
    def _project_state(self, state: QuantumState) -> QuantumState:
        """ 
        Project a state into the stabilizer subspace.

        Args:
            state (QuantumState): The state which has to be projected into the stabilizer subspace.

        Returns: 
            Projection of input QuantumState into the stabilizer subspace.
        """
        transformation_list = []
        # Hadamards where rotated onto Pauli X operators
        transformation_list += [
            Had(self.stabilizers.n_qubits, i) for i in np.where(
                np.sum(
                    self.stabilizers.rotate_onto_single_qubit_paulis().X_block & 
                    ~self.stabilizers.rotate_onto_single_qubit_paulis().Z_block,
                    axis=0
            )
                )[0]
        ]
        # Projections onto the stabilizer subspace
        transformation_list += list(map(lambda x:(x**2 + x)*.5,self.stabilizers.rotate_onto_single_qubit_paulis()))
        # Rotations mapping stabilizers onto single-qubit Pauli operators
        transformation_list += list(map(lambda s:trotter(s[0]*(np.pi/4*1j)), self.stabilizers.stabilizer_rotations))
        # Product over the transformation list yields final transformation operator
        transformation = reduce(lambda x,y:x*y, transformation_list)
        # apply transformation to the reference state
        transformed_state = transformation * state
        # drop stabilized qubit positions and sum over potential duplicates
        return QuantumState(
            transformed_state.state_matrix[:, self.free_qubit_indices], 
            transformed_state.state_op.coeff_vec
        ).cleanup(zero_threshold=1e-12)