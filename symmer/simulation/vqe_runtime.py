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
    n_groups    = 5
    grouping_scheme = 'classical_shadows'
    
    def __init__(self,
        backend,
        user_messenger,
        ansatz: QuantumCircuit,
        operator: PauliSumOp = None,
        operator_groups: List[Union[PauliSumOp, PauliOp]] = None
    ) -> None:
        """
        """
        self.backend = backend
        self.user_messenger = user_messenger
        self.ansatz = ansatz
        self.n_params = ansatz.num_parameters
        self.n_qubits = ansatz.num_qubits
        self.operator = operator
        self.operator_groups = operator_groups
        if self.grouping_scheme == 'classical_shadows':
            assert(self.operator is not None), 'No operator specified for classical shadow tomography'
            self.prepare_classical_shadows_measurement_groups()
        elif self.grouping_scheme == 'qubitwise_commuting':
            assert(self.operator_groups is not None), 'No qubitwise commuting operator_groups specified'
            self.prepare_qubitwise_commuting_measurement_groups()
        else:
            raise ValueError('Unrecognised grouping_scheme value, must be one of classical_shadows or qubitwise_commuting')
    
    def prepare_qubitwise_commuting_measurement_groups(self):
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

        self.circuits = circuits
        self.group_data = group_data

    def QWC_terms(self, basis_symp_vec):
        X_basis, Z_basis = np.hsplit(basis_symp_vec, 2)
        X_block, Z_block = np.hsplit(self.operator.primitive.table.array, 2)

        non_I = (X_block | Z_block) & (X_basis | Z_basis)
        X_match = np.all((X_block & non_I) == (X_basis & non_I), axis=1)
        Z_match = np.all((Z_block & non_I) == (Z_basis & non_I), axis=1)

        QWC_mask = X_match & Z_match

        return self.operator[QWC_mask]

    def prepare_classical_shadows_measurement_groups(self):
        """
        """
        circuits=[]
        group_data=[]
        random_bases = np.random.randint(0,2,(self.n_groups,2*self.n_qubits), dtype=bool)
        for basis_symp_vec in random_bases:

            operator_QWC = self.QWC_terms(basis_symp_vec)
            X_block, Z_block = np.hsplit(operator_QWC.primitive.table.array.astype(int), 2)
            coeff_vec = operator_QWC.coeffs
            new_Z_block = (X_block | Z_block)[:,::-1]  
            group_data.append([new_Z_block, coeff_vec])

            cob_group = PauliTable(basis_symp_vec).to_labels()[0]
            cob_gates, target = PauliBasisChange().get_cob_circuit(Pauli(cob_group))
            qc = self.ansatz.compose(cob_gates.to_circuit())
            qc.measure_all()
            circuits.append(qc)

        circuits = transpile(circuits, self.backend)

        self.circuits = circuits
        self.group_data = group_data

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
    ansatz          = kwargs.pop("ansatz", None)
    operator        = kwargs.pop("operator", None)
    operator_groups = kwargs.pop("operator_groups", None)
    
    vqe = VQE_Runtime(
        backend=backend,
        user_messenger = user_messenger,
        ansatz=ansatz,
        operator=operator,
        operator_groups=operator_groups
    )
    vqe.n_shots     = kwargs.pop("n_shots", None)
    vqe.maxiter     = kwargs.pop("maxiter", None)
    vqe.optimizer   = kwargs.pop("optimizer", None)
    vqe.init_params = kwargs.pop("init_params", None)
    vqe.n_groups    = kwargs.pop("n_groups", None)
    vqe.grouping_scheme = kwargs.pop("grouping_scheme", None)
    
    if vqe.grouping_scheme == 'classical_shadows':
        assert(vqe.operator is not None), 'No operator specified for classical shadow tomography'
        vqe.prepare_classical_shadows_measurement_groups()
    elif vqe.grouping_scheme == 'qubitwise_commuting':
        assert(vqe.operator_groups is not None), 'No qubitwise commuting operator_groups specified'
        vqe.prepare_qubitwise_commuting_measurement_groups()
    else:
        raise ValueError('Unrecognised grouping_scheme value, must be one of classical_shadows or qubitwise_commuting')
    
    return vqe.run()