from symmer.symplectic import PauliwordOp
import numpy as np
from cached_property import cached_property

class matrix_to_Pword():
    def __init__(self, matrix: np.array):

        self.n_qubits = None
        self.matrix = self._correct_size_m(matrix)

    def _correct_size_m(self, matrix: np.array) -> np.array:
        """
        method to make sure matrix is size 2^n by 2^n
        Args:
            matrix (np.array): matrix to decompose as linear combination of Pauli operators

        Returns:
            matrix (np.array): square matrix of 2^n by 2^n

        """

        self.n_qubits = int(np.ceil(np.log2(max(matrix.shape))))

        if (self.n_qubits, self.n_qubits) == matrix.shape:
            return matrix
        else:
            mat = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits))
            mat[:matrix.shape[0],
                :matrix.shape[1]] = matrix
            return mat

    @cached_property
    def operator(self):

        # XZ_block = np.eye(2*self.n_qubits, dtype=int)

        # fast method to build all binary assignments
        int_list = np.arange(2 ** (2 * self.n_qubits))
        XZ_block = (((int_list[:, None] & (1 << np.arange(2 * self.n_qubits))[::-1])) > 0).astype(int)

        operator_basis = PauliwordOp(XZ_block, np.ones(XZ_block.shape[0]))
        # print(operator_basis)

        denominator = 2 ** self.n_qubits
        decomposition = PauliwordOp({'I' * self.n_qubits: 0})
        for op in operator_basis:
            const = np.trace(op.to_sparse_matrix @ self.matrix) / denominator
            decomposition += op.multiply_by_constant(const)

        return decomposition.cleanup()

    def check_decomposition(self):
        if not np.allclose(self.operator.to_sparse_matrix.todense(), self.matrix):
            raise ValueError('decomposition is incorrect')
        else:
            return True