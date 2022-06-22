import numpy as np
from typing import List, Union
from scipy.optimize import minimize
from qiskit.quantum_info import Pauli, PauliTable
from qiskit.opflow import PauliSumOp, PauliOp, PauliBasisChange
from qiskit import QuantumCircuit, transpile

class VQE_Runtime:
    """
    """
    n_shots     = 2**12
    maxiter     = 10
    optimizer   = 'SLSQP'
    init_params = None

    def __init__(self,
        backend,
        user_messenger,
        ansatz: QuantumCircuit,
        operator_groups: List[Union[PauliSumOp, PauliOp]]
    ) -> None:
        """
        """
        self.backend = backend
        self.user_messenger = user_messenger
        self.ansatz = ansatz
        self.n_params = ansatz.num_parameters
        self.operator_groups = operator_groups
        self.circuits, self.group_data = self.prepare_measurement_groups()

    def prepare_measurement_groups(self):
        """
        """
        circuits = []
        group_data = []
        for opgroup in self.operator_groups:
            if isinstance(opgroup, PauliSumOp):
                X_block, Z_block = np.hsplit(opgroup.primitive.table.array.astype(int), 2)
                coeff_vec = opgroup.coeffs
            elif isinstance(opgroup, PauliOp):
                X_block, Z_block = np.hsplit(PauliTable(opgroup.primitive).array.astype(int), 2)
                coeff_vec = np.array([opgroup.coeff])
            else:
                raise ValueError('Unrecognized operator group type, must be PauliSumOp or PauliOp')

            new_Z_block = (X_block | Z_block)[:,::-1]  
            group_data.append([new_Z_block, coeff_vec])
            X_pos = np.einsum('ij->j', X_block)!=0
            Z_pos = np.einsum('ij->j', Z_block)!=0
            cob_group = PauliTable(np.hstack([X_pos, Z_pos])).to_labels()[0]
            cob_gates, target = PauliBasisChange().get_cob_circuit(Pauli(cob_group))
            qc = self.ansatz.compose(cob_gates.to_circuit())
            qc.measure_all()
            circuits.append(qc)

        circuits = transpile(circuits, self.backend)

        return circuits, group_data

    def get_counts(self, bound_circuits):
        """
        """
        # implement measurement error mitigation!
        job = self.backend.run(
            circuits = bound_circuits,
            shots=self.n_shots
        )
        result = job.result()
        return result.get_counts()

    def _estimate(self, countset):
        """
        """
        assert(len(countset)==len(self.group_data)), 'Incompatible number of counts and operators'
        expval = 0
        for (group_Z_block, coeff_vec), measurements in zip(self.group_data, countset):
            for binstr, freq in measurements.items():
                weight = freq/self.n_shots
                binarr = np.array([int(i) for i in binstr])
                signed = (-1)**np.einsum('ij->i', np.bitwise_and(binarr, group_Z_block))
                expval += weight*np.sum(signed*coeff_vec)
        return expval.real

    def estimate(self, x):
        """
        """
        countset = self.get_counts([qc.bind_parameters(x) for qc in self.circuits]) 
        return self._estimate(countset)
        
    def gradient(self, x):
        """
        """
        def bind(param_list):
            all_circuits = []
            for params in param_list:
                all_circuits+=[qc.bind_parameters(params) for qc in self.circuits]
            return all_circuits
        
        def split_list(alist, wanted_parts=1):
            length = len(alist)
            return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
                    for i in range(wanted_parts) ]
        
        upper_shift_params = [x+np.pi/4*np.eye(1,self.n_params,i)[0] for i in range(self.n_params)]
        lower_shift_params = [x-np.pi/4*np.eye(1,self.n_params,i)[0] for i in range(self.n_params)]
        
        all_counts = self.get_counts(bind(upper_shift_params+lower_shift_params))
        upper_shift_counts, lower_shift_counts = split_list(all_counts, 2)

        upper_shift_energy = np.array([self._estimate(countset) for countset in split_list(upper_shift_counts, self.n_params)])
        lower_shift_energy = np.array([self._estimate(countset) for countset in split_list(lower_shift_counts, self.n_params)])
        
        return upper_shift_energy - lower_shift_energy
    
    def run(self):
        """
        """
        interim_values = {'values':[], 'params':[], 'gradients':[], 'count':0}

        if self.init_params is None:
            self.init_params = np.zeros(self.n_params)
            
        def fun(x):
            interim_values['count']+=1
            countnum = interim_values['count']
            energy = self.estimate(x)
            self.user_messenger.publish(f'Optimization step #{countnum}: energy = {energy}')
            interim_values['params'].append((interim_values['count'], x))
            interim_values['values'].append((interim_values['count'], energy))
            return energy

        def jac(x):
            Delta = self.gradient(x)
            interim_values['gradients'].append((interim_values['count'], Delta))
            return Delta
            
        self.user_messenger.publish('Optimization commencing')
        vqe_result = minimize(
            fun=fun, 
            jac=jac,
            x0=self.init_params,
            method=self.optimizer,
            #tol=opt_tol
            options={'maxiter':self.maxiter}
        )
        vqe_result.success = bool(vqe_result.success)
        self.user_messenger.publish('VQE complete')
        
        return vqe_result, interim_values

def main(backend, user_messenger, **kwargs):
    """ The main runtime program entry-point.
    All of the heavy-lifting is handled by the VQE_Runtime class
    """
    ansatz          = kwargs.pop("ansatz", 1)
    operator_groups = kwargs.pop("operator_groups", 1)
    
    vqe = VQE_Runtime(
        backend=backend,
        user_messenger = user_messenger,
        ansatz=ansatz, 
        operator_groups=operator_groups
    )
    vqe.n_shots     = kwargs.pop("n_shots", 1)
    vqe.maxiter     = kwargs.pop("maxiter", 1)
    vqe.optimizer   = kwargs.pop("optimizer", 1)
    vqe.init_params = kwargs.pop("init_params", 1)
    
    return vqe.run()