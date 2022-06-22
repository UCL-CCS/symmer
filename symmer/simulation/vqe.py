import numpy as np
from symmer.symplectic import PauliwordOp, ObservableGraph
from symmer.simulation import vqe_runtime
from qiskit import Aer, QuantumCircuit
from qiskit.providers.ibmq.runtime import UserMessenger

def VQE(
    operator: PauliwordOp, 
    ansatz: QuantumCircuit,
    init_params: np.array = None,
    optimizer: str = 'SLSQP',
    maxiter: int = 10,
    n_shots: int = 2**12
    ):
    QWC_decomposition = list(
            ObservableGraph(
                operator.symp_matrix, 
                operator.coeff_vec
            ).clique_cover(
                clique_relation='QWC', 
                colouring_strategy='largest_first'
            ).values()
        )
    assert(sum(QWC_decomposition)==operator), 'Decomposition into QWC groups failed'

    runtime_input = {
        "ansatz": ansatz,
        "operator_groups": [op.to_PauliSumOp for op in QWC_decomposition],
        "init_params": init_params,
        "optimizer": optimizer,
        "maxiter": maxiter,
        "n_shots": n_shots
    }

    vqe_result, interim_values = vqe_runtime.main(
        backend = Aer.get_backend('qasm_simulator'), 
        user_messenger = UserMessenger(), 
        **runtime_input
    )

    return vqe_result, interim_values
 