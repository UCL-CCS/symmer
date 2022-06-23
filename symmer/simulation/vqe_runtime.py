from ast import operator
import numpy as np
from typing import List, Union
from scipy.optimize import minimize
from qiskit.quantum_info import Pauli, PauliTable
from qiskit.opflow import PauliSumOp, PauliOp, PauliBasisChange
from qiskit import QuantumCircuit, transpile
import mthree

class VQE_Runtime:
    """ Runtime program for performing VQE routines.

    Accepts a parametrized ansatz (QuantumCircuit) with an observable (PauliSumOp)
    or observable grouping (List[PauliSumOp]), e.g. qubitwise commuting groups that
    may be measured simultaneously. If no grouping is specified then a collecion
    of randomized bases is generated and the qubitwise commuting terms of the operator
    are decomposed in each measurement basis (classical shadows https://arxiv.org/abs/2002.08953)

    For each measurement group, a change-of-basis is implemented in-circuit so that
    only Pauli-Z measurements are necessary.

    This class facilitates the submission of batch circuit jobs to the QPU, with the
    returned measurement outcomes being used to estimate expectaton values in each
    observable group, before being summed to yield an energy estimate of the full observable.
    In the classical shadows scheme, the coefficients are corrected to allow for duplicated
    terms accross the measurement groups.

    Gradients are computed analytically via the parameter shift rule (eq.15 https://arxiv.org/abs/1906.08728).
    Such gradient calculations are expensive for large numbers of parameters; each gradient requires 2*n_params
    observable expectation value estimations, although they facilitate rapid convergence to the minimum energy 
    within the limitations of the provided ansatz.

    The VQE_Runtime.run() wraps all this within a scipy.optimize.minimize routine, returning the optimizer
    output, in addition to the interim energy, parameter and gradient values.
    """
    n_shots     = 2**12
    n_realize   = 1
    n_groups    = 5
    maxiter     = 10
    optimizer   = 'SLSQP'
    init_params = None
    mitigate_errors = True
    
    def __init__(self,
        backend,
        user_messenger,
        ansatz: QuantumCircuit,
        observable: PauliSumOp = None,
        observable_groups: List[Union[PauliSumOp, PauliOp]] = None
    ) -> None:
        """
        Args:
            - backend (): 
                The target QPU for circuit jobs
            - user_messenger (): 
                Allows readout during computation
            - ansatz (QuantumCircuit): 
                The parametrized QuantumCircuit used in state preparation
            - observable (PauliSumOp): 
                The observable for expecation value estimation
            - observable_groups (List[PauliSumOp]): 
                Some Pauli operators may be estimated simultaneously. The grouping 
                specifies this - if None, will determine a random grouping.
        """
        self.backend = backend
        self.user_messenger = user_messenger
        self.ansatz = ansatz
        self.n_params = ansatz.num_parameters
        self.n_qubits = ansatz.num_qubits
        self.observable = observable
        # if no observable grouping is specified, will group by classical shadows (i.e. random measurement bases)
        if observable_groups is None:
            assert(observable is not None), 'Must provide an observable or observable grouping'
            self.observable_groups = self.prepare_classical_shadows_observable_groups()
        else:
            self.observable_groups = observable_groups

        self.circuits,self.group_data = self.prepare_qubitwise_commuting_measurement_groups()

        #if self.mitigate_errors:
        #    maps = mthree.utils.final_measurement_mapping(self.circuits)
        #    mit  = mthree.M3Mitigation(self.backend)
        #    mit.cals_from_system(maps)
            
    def QWC_terms(self, basis_symp_vec):
        """ Given the symplectic representation of a measurement basis,
        determines which operator terms may be measured in this basis

        Returns:  
            QWC_observable (PauliSumOp): 
                an observable consisting of the self.observable terms 
                that qubitwise commute with the measurement basis
        """
        X_basis, Z_basis = np.hsplit(basis_symp_vec, 2)
        X_block, Z_block = np.hsplit(self.observable.primitive.table.array, 2)
        # identify the qubit positions on which there is at least one non-identity operation
        non_I = (X_block | Z_block) & (X_basis | Z_basis)
        # identift matches between the operator and basis, these indicate qubitwise commutation
        X_match = np.all((X_block & non_I) == (X_basis & non_I), axis=1)
        Z_match = np.all((Z_block & non_I) == (Z_basis & non_I), axis=1)
        # mask the terms of self.observable that qubitwise commute with the basis
        QWC_mask = X_match & Z_match
        QWC_observable = self.observable[QWC_mask]
        
        return QWC_observable

    def prepare_classical_shadows_observable_groups(self):
        """ If no grouping is explicitly specified, we generate a grouping at random.
        This is synonymous with classical shadows and can facilitate convergence on
        the energy estimate using fewer quantum measurements than full state tomography.
        
        Returns:
            - corrected_observable_groups (List[PauliSumOp]):
                An observable groupingl each group contains terms that may be measured
                simulataneously. Note duplications can occur between measurement groups,
                hence this is corrected in the coefficients of the correpsonding terms.
        """
        random_bases = np.random.randint(0,2,(self.n_groups,2*self.n_qubits), dtype=bool)
        observable_groups = [
            self.QWC_terms(basis_symp_vec) for basis_symp_vec in random_bases
        ]
        # need to account for duplicated terms accross shadow groups
        num_duplicate_terms = {term:0 for term in self.observable.primitive.table.to_labels()}
        for opgroup in observable_groups:
            for term in opgroup.primitive.table.to_labels():
                num_duplicate_terms[term]+=1

        # dividing each coefficient by the number of duplications of the 
        # corresponding Pauli term will average the energy contributions
        corrected_observable_groups = []
        for opgroup in observable_groups:
            pauli_list = opgroup.primitive.table.to_labels()
            corrected_coeff_vec = []
            for coeff, term in zip(opgroup.coeffs, pauli_list):
                corrected_coeff_vec.append(coeff/num_duplicate_terms[term])
            corrected_group_observable = PauliSumOp.from_list(
                pauli_list=zip(pauli_list, corrected_coeff_vec)
            )
            corrected_observable_groups.append(corrected_group_observable)
        
        return corrected_observable_groups

    def prepare_qubitwise_commuting_measurement_groups(self):
        """ Given the grouping scheme, each measurement group is assigned a circuit
        composed from the specified ansatz and the relevant change of basis gates.
        For efficient expectation value computation post circuit execution, we return
        a binary array representing the Z_block of the transormed group observable.

        Returns:
            - circuits (List[PauliSumOp]):
                The paramtrized circuits in each measurement basis
            - group_data (List[Tuple[np.array, np.array]]):
                The minimal observable information necessary for expectation
                value estimation per measurement group.
        """
        circuits = []
        group_data = []
        for opgroup in self.observable_groups:
            if isinstance(opgroup, PauliSumOp):
                X_block, Z_block = np.hsplit(opgroup.primitive.table.array.astype(int), 2)
                coeff_vec = opgroup.coeffs
            elif isinstance(opgroup, PauliOp):
                X_block, Z_block = np.hsplit(PauliTable(opgroup.primitive).array.astype(int), 2)
                coeff_vec = np.array([opgroup.coeff])
            else:
                raise ValueError('Unrecognized observable group type, must be PauliSumOp or PauliOp')

            new_Z_block = (X_block | Z_block)[:,::-1]  
            group_data.append([new_Z_block, coeff_vec])
            X_pos = np.einsum('ij->j', X_block)!=0
            Z_pos = np.einsum('ij->j', Z_block)!=0
            cob_group = PauliTable(np.hstack([X_pos, Z_pos])).to_labels()[0]
            cob_gates, target = PauliBasisChange().get_cob_circuit(Pauli(cob_group))
            qc = self.ansatz.compose(cob_gates.to_circuit())
            qc.measure_all()
            circuits.append(qc)

        circuits = transpile(circuits, self.backend, optimization_level=3)

        return circuits, group_data
   
    def get_counts(self, param_list: List[np.array]):
        """ Given a list of parametrizations, bind the circuits and submit to the backend
        
        Returns:
            - result (List[Dict[str:int]]):
                A list of dictionaries in which keys are binary strings and 
                their values the frequency of said measurement outcome 

        """
        bound_circuits = []
        for params in param_list:
            bound_circuits+=[qc.bind_parameters(params) for qc in self.circuits]
        # implement measurement error mitigation!
        job = self.backend.run(
            circuits = bound_circuits,
            shots=self.n_shots
        )
        result = job.result()
        return result.get_counts()

    def _estimate(self, countset):
        """ Given the measurment outcomes retrieved from the the backend, calculate
        the corresponding expectation values accross the measurement groups and sum

        Returns:
            expval.real (np.float):
                The expectation value of the observable
        """
        assert(len(countset)==len(self.group_data)), 'Incompatible number of counts and observables'
        expval = 0
        for (group_Z_block, coeff_vec), measurements in zip(self.group_data, countset):
            for binstr, freq in measurements.items():
                weight = freq/self.n_shots
                binarr = np.array([int(i) for i in binstr])
                signed = (-1)**np.einsum('ij->i', np.bitwise_and(binarr, group_Z_block))
                expval += weight*np.sum(signed*coeff_vec)
        return expval.real

    def estimate(self, x):
        """ Wraps the _estimate method, calling get_counts to submit
        the circuit jobs parametrized by the input array x
        """
        samples = []
        for sample in range(self.n_realize):
            countset = self.get_counts([x])
            energy = self._estimate(countset)
            samples.append(energy)
        return sum(samples)/self.n_realize
        
    def gradient(self, x):
        """ Implementation of the parameter shif rule of https://arxiv.org/abs/1906.08728 (eq.15).
        Requires 2*n_params estimation routines, hence expensive for highly parametrized circuits.
        All circuits for the gradient estimation are submitted as a batch job. For example, given
        parameters [a,b,c], the circuits submitted are of the following form:
        [
            [a+pi/4,b,c],[a,b+pi/4,c],[a,b,c+pi/4],[a-pi/4,b,c],[a,ba-pi/4,c],[a,b,ca-pi/4]
        ],
        so the returned measurement data must be split accordingly, using split_list function below.
        
        Returns:
            estimated_gradient (np.array):
                An array representing the gradient w.r.t. each parameter at the point x
        """
        def split_list(alist, wanted_parts=1):
            length = len(alist)
            return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
                    for i in range(wanted_parts) ]
        
        upper_shift_params = [x+np.pi/4*np.eye(1,self.n_params,i)[0] for i in range(self.n_params)]
        lower_shift_params = [x-np.pi/4*np.eye(1,self.n_params,i)[0] for i in range(self.n_params)]
        
        # submit all circuits for gradient estimation as a batch job to the QPU
        all_counts = self.get_counts(upper_shift_params+lower_shift_params)
        # need to split the measurement results into the relevant sets 
        upper_shift_counts, lower_shift_counts = split_list(all_counts, 2)

        upper_shift_energy = np.array([self._estimate(countset) for countset in split_list(upper_shift_counts, self.n_params)])
        lower_shift_energy = np.array([self._estimate(countset) for countset in split_list(lower_shift_counts, self.n_params)])
        
        estimated_gradient = upper_shift_energy - lower_shift_energy
        
        return estimated_gradient
    
    def run(self):
        """ Runs a scipy.optimize.minimize routine that minimizes the output of
        self.estimate, informed by self.gradient (optimizer-dependent)

        Recommended optimizers:
            - SLSQP  (gradient-descent, does not evaluate Jacobian at each iterate like BFGS or CG so is faster)
            - COBYLA (gradient-free)
        
        Returns:
            - vqe_result (Dict[str,Union[int, float, bool, array]]):
                The optimizer output
            - interim_values (Dict[str,List[Union[float, array]]]):
                The interim energy, parameter and gradient values
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
        vqe_result.success = bool(vqe_result.success) # to ensure serializable
        self.user_messenger.publish('VQE complete')
        
        return vqe_result, interim_values

def main(backend, user_messenger, **kwargs):
    """ The main runtime program entry-point.

    All the heavy-lifting is handled by the VQE_Runtime class

    Returns:
        - vqe_result (Dict[str,Union[int, float, bool, array]]):
            The optimizer output
        - interim_values (Dict[str,List[Union[float, array]]]):
            The interim energy, parameter and gradient values
    """
    ansatz            = kwargs.pop("ansatz", None)
    observable        = kwargs.pop("observable", None)
    observable_groups = kwargs.pop("observable_groups", None)
    
    vqe = VQE_Runtime(
        backend=backend,
        user_messenger = user_messenger,
        ansatz=ansatz,
        observable=observable,
        observable_groups=observable_groups
    )
    vqe.n_shots     = kwargs.pop("n_shots", 2**12)
    vqe.n_realize   = kwargs.pop("n_realize", 1)
    vqe.maxiter     = kwargs.pop("maxiter", 10)
    vqe.optimizer   = kwargs.pop("optimizer", 'SLSQP')
    vqe.init_params = kwargs.pop("init_params", None)
    vqe.n_groups    = kwargs.pop("n_groups", 5)
    vqe.mitigate_errors = kwargs.pop("mitigate_errors", False)
    
    return vqe.run()