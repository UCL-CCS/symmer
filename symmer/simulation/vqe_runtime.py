import numpy as np
from typing import List, Union, Tuple, Dict
from functools import reduce, cached_property
from scipy.optimize import minimize
from qiskit.quantum_info import Pauli, PauliTable
from qiskit.opflow import PauliSumOp, PauliOp, PauliBasisChange
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, transpile
import mthree

def exponential_circuit(P: PauliSumOp, theta) -> QuantumCircuit:
    """
    """
    assert(len(P)==1), "Provided multiple Pauli operators to exponentiate."
    
    # identify gate positions
    X_vec, Z_vec = np.hsplit(P.primitive.table.array, 2)
    X_vec, Z_vec = X_vec[0], Z_vec[0]
    H_indices = np.where(X_vec)[0]
    S_indices = np.where(X_vec & Z_vec)[0]
    CNOT_indices = np.where(X_vec | Z_vec)[0]
    CNOT_pairs = list(zip(CNOT_indices[:-1], CNOT_indices[1:]))
    
    # build the exponential circuit
    excitation_block = QuantumCircuit(P.num_qubits)
    # to Pauli X basis
    for S_index in S_indices:
        excitation_block.sdg(S_index)
    # to Pauli Z basis
    for H_index in H_indices:
        excitation_block.h(H_index)
    # compute parity
    for control_index, target_index in CNOT_pairs:
        excitation_block.cx(control_index, target_index)
    # rotate
    excitation_block.rz(-2*theta, CNOT_indices[-1])
    # reverse parity calculation and basis change
    for control_index, target_index in CNOT_pairs[::-1]:
        excitation_block.cx(control_index, target_index)
    for H_index in H_indices:
        excitation_block.h(H_index)
    for S_index in S_indices:
        excitation_block.s(S_index)
        
    return excitation_block

def flatten(xss):
    """
    """
    return [x for xs in xss for x in xs]

def split_list(alist, wanted_parts=1):
    """
    """
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
            for i in range(wanted_parts) ]

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

    Also included is an implementation of ADAPT-VQE.
    """
    n_shots     = 2**12
    n_realize   = 1
    n_groups    = 5
    optimizer   = 'SLSQP'
    init_params = None
    opt_setting = {'maxiter':10}
    mitigate_errors = True
    
    def __init__(self,
        backend,
        user_messenger,
        ansatz_pool: PauliSumOp = None,
        observable: PauliSumOp = None,
        observable_groups: List[Union[PauliSumOp, PauliOp]] = None,
        reference_state: np.array = None
    ) -> None:
        """
        Args:
            - backend (): 
                The target QPU for circuit jobs
            - user_messenger (): 
                Allows readout during computation
            - ansatz_pool (PauliSumOp): 
                An operator whose terms shall be taken as the pool of 
                excitations from which to select in ADAPT-VQE
            - observable (PauliSumOp): 
                The observable for expecation value estimation
            - observable_groups (List[PauliSumOp]): 
                Some Pauli operators may be estimated simultaneously. The grouping 
                specifies this - if None, will determine a random grouping.
            - reference_state (array):
                A single basis state in which to initiate the ansatz circuit
        """
        self.backend = backend
        self.user_messenger = user_messenger
        self.reference_state = reference_state
        self.observable = observable
        
        # if no observable grouping is specified, will group by classical shadows (i.e. random measurement bases)
        if observable_groups is None:
            assert(observable is not None), 'Must provide an observable or observable grouping'
            self.observable_groups = self.prepare_classical_shadows_observable_groups()
        else:
            self.observable_groups = observable_groups
        self.n_qubits = self.observable_groups[0].num_qubits
        
        # ansatz_pool may be specified for the purposes of ADAPT-VQE
        if ansatz_pool is not None:
            params = ParameterVector('P', len(ansatz_pool))
            # build the exponentiated QuantumCircuit for each pool operator
            self.excitation_block_pool = {
                index:{
                    'pauli_label':P.primitive.table.to_labels()[0],
                    'exp_circuit':exponential_circuit(P, params[index])
                } 
                for index, P in enumerate(ansatz_pool)
            }
        
        # identify change-of-basis circuit blocks and data required to evaluate expectation values
        self.cob_blocks,self.group_data = self.prepare_qubitwise_commuting_measurement_groups()

        if self.mitigate_errors:
            #self.maps = mthree.utils.final_measurement_mapping(self.circuits)
            self.m3   = mthree.M3Mitigation(self.backend)
            self.m3.cals_from_system(range(self.n_qubits))
            
    @cached_property
    def reference_circuit(self):
        """ A circuit preparing the supplied reference state (such as Hartree-Fock)

        Returns:
            - ref_qc (QuantumCircuit):
                The reference state in which to initiate the ansatz circuit
        """
        ref_qc = QuantumCircuit(self.n_qubits)
        if self.reference_state is not None:
            for i in np.where(self.reference_state[::-1])[0]:
                ref_qc.x(i)
        return ref_qc
            
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
            - cob_blocks (List[QuantumCircuit]):
                The change of basis gates for each QWC measurement group
            - group_data (List[Tuple[np.array, np.array]]):
                The minimal observable information necessary for expectation
                value estimation per measurement group.
        """
        cob_blocks = []
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
            cob_blocks.append(cob_gates.to_circuit())

        return cob_blocks, group_data
    
    def prepare_circuits(self, ansatz: QuantumCircuit) -> List[QuantumCircuit]:
        """ Given an ansatz, compose with the change of basis block per QWC measurement group
        
        Returns:
            - circuits (List[PauliSumOp]):
                The paramtrized circuits in each measurement basis
        """
        circuits = []
        for cob_gates in self.cob_blocks:
            qc = ansatz.compose(cob_gates)
            qc.measure_all()
            circuits.append(qc)
        circuits = transpile(circuits, self.backend, optimization_level=3)
        return circuits
   
    def get_counts(self, param_list: List[np.array]) -> List[Dict[str, float]]:
        """ Given a list of parametrizations, bind the circuits and submit to the backend
        
        Returns:
            - result (List[Dict[str:int]]):
                A list of dictionaries in which keys are binary strings and 
                their values the frequency of said measurement outcome 

        """
        bound_circuits = []
        for params in param_list:
            bound_circuits+=[qc.bind_parameters(params) for qc in self.circuits]
        
        job = self.backend.run(
            circuits = bound_circuits,
            shots=self.n_shots
        )
        result = job.result()
        raw_counts = result.get_counts()
        
        if self.mitigate_errors:
            quasis = self.m3.apply_correction(raw_counts, range(self.n_qubits))
            return quasis.nearest_probability_distribution()
        else:
            return [{binstr:freq/self.n_shots for binstr,freq in counts.items()} 
                    for counts in raw_counts] 

    def _estimate(self, countset: List[Dict[str, float]]) -> float:
        """ Given the measurment outcomes retrieved from the the backend, calculate
        the corresponding expectation values accross the measurement groups and sum

        Returns:
            expval.real (np.float):
                The expectation value of the observable
        """
        assert(len(countset)==len(self.group_data)), 'Incompatible number of counts and observables'
        expval = 0
        for (group_Z_block, coeff_vec), measurements in zip(self.group_data, countset):
            for binstr, weight in measurements.items():
                binarr = np.array([int(i) for i in binstr])
                signed = (-1)**np.einsum('ij->i', np.bitwise_and(binarr, group_Z_block))
                expval += weight*np.sum(signed*coeff_vec)
        return expval.real

    def estimate(self, x: np.array) -> Tuple[float, float]:
        """ Wraps the _estimate method, calling get_counts to submit
        the circuit jobs parametrized by the input array x

        Returns:
            - energy (float):
                The mean energy over each expectation value realization
            - stddev (float):
                The standard deviation of expectation value realizations
        """
        samples = []

        for sample in range(self.n_realize):
            countset = self.get_counts([x])
            energy = self._estimate(countset)
            samples.append(energy)
        
        energy = np.mean(samples)
        stddev = np.std(samples)

        return energy, stddev 
    
    def gradient_batch(self, param_list: List[np.array], n_batches: int) -> np.array:
        """ Prepares for batch submission of QuantumCircuits in calculating gradients.
        Can be used either to submit multiple parametrizations (as in self.gradient(...))
        and/or to submit multiple circuits (as in self.ADAPT_VQE(...)). The output counts
        are received in a flat list, so one needs to specify the number of batches to
        decompose the results into like-experiments.

        Returns:
            - estimated_gradient (array):
                Partial gradients with respect to the input parametrizations
        """
        # submit all circuits for gradient estimation as a batch job to the QPU
        all_counts = self.get_counts(param_list)
        # need to split the measurement results into the relevant sets 
        upper_shift_counts, lower_shift_counts = split_list(all_counts, 2)
        upper_shift_energy = np.array([self._estimate(countset) for countset in split_list(upper_shift_counts, n_batches)])
        lower_shift_energy = np.array([self._estimate(countset) for countset in split_list(lower_shift_counts, n_batches)])
        
        estimated_gradient = upper_shift_energy - lower_shift_energy
        
        return estimated_gradient
        
    def gradient(self, x: np.array) -> np.array:
        """ Implementation of the parameter shif rule of https://arxiv.org/abs/1906.08728 (eq.15).
        Requires 2*n_params estimation routines, hence expensive for highly parametrized circuits.
        All circuits for the gradient estimation are submitted as a batch job. For example, given
        parameters [a,b,c], the circuits submitted are of the following form:
        [
            [a+pi/4,b,c],[a,b+pi/4,c],[a,b,c+pi/4],[a-pi/4,b,c],[a,ba-pi/4,c],[a,b,ca-pi/4]
        ],
        so the returned measurement data must be split accordingly, using split_list function above.
        
        Returns:
            estimated_gradient (np.array):
                An array representing the gradient w.r.t. each parameter at the point x
        """        
        upper_shift_params = [x+np.pi/4*np.eye(1,self.n_params,i)[0] for i in range(self.n_params)]
        lower_shift_params = [x-np.pi/4*np.eye(1,self.n_params,i)[0] for i in range(self.n_params)]
        estimated_gradient = self.gradient_batch(upper_shift_params+lower_shift_params, self.n_params)

        return estimated_gradient
    
    def run(self, ansatz: QuantumCircuit) -> Tuple[Dict, Dict]:
        """ Runs a scipy.optimize.minimize routine that minimizes the output of
        self.estimate, informed by self.gradient (optimizer-dependent)

        Recommended optimizers:
            - SLSQP  (gradient-descent, does not evaluate Jacobian at each iterate like BFGS or CG so is faster)
            - COBYLA (gradient-free)

        Inputs:
            - ansatz (QuantumCircuit): 
                The parametrized QuantumCircuit used in state preparation
        
        Returns:
            - vqe_result (Dict[str,Union[int, float, bool, array]]):
                The optimizer output
            - interim_values (Dict[str,List[Union[float, array]]]):
                The interim energy, parameter and gradient values
        """
        self.n_params = ansatz.num_parameters
        self.circuits = self.prepare_circuits(ansatz)
        
        interim_values = {'values':[], 'stddev':[], 'params':[], 'gradients':[], 'count':0}

        if self.init_params is None:
            self.init_params = np.zeros(self.n_params)
            
        def fun(x: np.array) -> float:
            interim_values['count']+=1
            countnum = interim_values['count']
            energy, stddev = self.estimate(x)
            self.user_messenger.publish(f'Optimization step #{countnum}: energy = {energy}')
            interim_values['params'].append((interim_values['count'], x))
            interim_values['values'].append((interim_values['count'], energy))
            interim_values['stddev'].append((interim_values['count'], stddev))
            return energy

        def jac(x: np.array) -> np.array:
            countnum = interim_values['count']
            Delta = self.gradient(x)
            grad_norm = np.sqrt(np.sum(np.square(Delta)))
            self.user_messenger.publish(f'Optimization step #{countnum}: gradient norm = {grad_norm}')
            interim_values['gradients'].append((interim_values['count'], Delta))
            return Delta
            
        self.user_messenger.publish('Optimization commencing')
        vqe_result = minimize(
            fun=fun, 
            jac=jac,
            x0=self.init_params,
            method=self.optimizer,
            #tol=opt_tol
            options=self.opt_setting
        )
        vqe_result.success = bool(vqe_result.success) # to ensure serializable
        self.user_messenger.publish('VQE complete')
        
        return vqe_result, interim_values
    
    def get_ansatz_from_pool(self, pool_indices: List[int] = None) -> QuantumCircuit:
        """ Given a list of pool indices, build the corresponding 
        ansatz from their exponentiated circuit blocks.

        Returns:
            - ansatz (QuantumCircuit):
                The ansatz circuit corresponding with the pool indices
        """
        if pool_indices is None:
            pool_indices = range(len(self.ansatz_pool))
        
        ansatz = reduce(
            lambda x,y:x.compose(y),
            [self.reference_circuit] + [self.excitation_block_pool[index]['exp_circuit'] for index in pool_indices]
        )
        return ansatz
    
    def VQE(self, pool_indices: List[int] = None) -> Tuple[Dict, Dict]:
        """ Perform VQE over the full ansatz pool

        Returns:
            - vqe_result (Dict[str,Union[int, float, bool, array]]):
                The optimizer output
            - interim_values (Dict[str,List[Union[float, array]]]):
                The interim energy, parameter and gradient values
        """
        ansatz = self.get_ansatz_from_pool(pool_indices)
        return self.run(ansatz)
    
    def ADAPT_VQE(self, 
            max_cycles: int = 5, 
            termination_threshold: float = 0.1, 
            reference_energy: float = None
        ) -> Dict:
        """ Implementation of qubit-ADAPT-VQE from https://doi.org/10.1103/PRXQuantum.2.020310
        
        Identifies a subset of terms from the input ansatz_pool that achieves the termination
        threshold (e.g. reaches chemical accuracy or gradient vector sufficiently close to zero) 

        Returns:
            - adapt_vqe_result (Dict):
                The optimial energy, parametrization, pool indices and the final ansatz circuit
            - adapt_vqe_interim_data (Dict):
                The interim data for each ADAPT cycle, each containing self.VQE() output data
        """
        adapt_vqe_interim_data = {}
        remaining_pool_indices = range(len(self.excitation_block_pool))
        optimized_pool_indices = []
        cycle = 0
        value = 1
        self.init_params = np.zeros(1)
        
        # iterate until termination threshold is achieved or max number of cycles exceeded
        while abs(value)>termination_threshold and cycle<max_cycles:
            cycle+=1
            self.user_messenger.publish(f'*** ADAPT cycle {cycle} ***')
            
            # calculate partial derivative w.r.t. each remaining pool element at zero
            self.circuits = flatten(
                [
                    self.prepare_circuits(
                        self.get_ansatz_from_pool(optimized_pool_indices + [index])
                    )
                    for index in remaining_pool_indices
                ]
            )
            upper_shift_params = self.init_params.copy(); upper_shift_params[-1] = np.pi/4
            lower_shift_params = self.init_params.copy(); lower_shift_params[-1] = -np.pi/4
            gradients = self.gradient_batch([upper_shift_params,lower_shift_params], len(remaining_pool_indices))
            gradients = list(zip(remaining_pool_indices, gradients))
            for index, Delta in gradients:
                self.user_messenger.publish(f'delta_{index} = {Delta}')
                
            # choose the pool element with largest gradient to append to ansatz
            new_index, new_Delta = sorted(gradients, key=lambda x:-abs(x[1]))[0]
            optimized_pool_indices.append(new_index)
            self.user_messenger.publish(f'Optimal pool indices: {optimized_pool_indices}')
            
            # perform a VQE simulation over the expanded ansatz
            vqe_result, interim_results = self.VQE(pool_indices=optimized_pool_indices)
            vqe_energy, vqe_params = vqe_result['fun'], vqe_result['x']
            self.user_messenger.publish(f'VQE optimal energy = {vqe_energy}')
            self.user_messenger.publish(f'VQE optimal params = {vqe_params}')
            
            if reference_energy is not None:
                value = vqe_result['fun'] - reference_energy
            else:
                value = new_Delta
            
            adapt_vqe_interim_data[cycle] = {
                'gradients':            gradients,
                'optimal_pool_indices': optimized_pool_indices,
                'vqe_result':           vqe_result,
                'interim_results':      interim_results,
                'termination_value':    value,
                'termination_status':   bool(abs(value)<=termination_threshold)
            }
            
            self.init_params = np.append(vqe_params, 0)
            self.user_messenger.publish(f'ADAPT cycle {cycle} value = {value}')
            remaining_pool_indices = list(set(remaining_pool_indices)-set(optimized_pool_indices))
        
        self.user_messenger.publish(f'ADAPT-VQE terminated.')
        adapt_vqe_result = {
            'optimal_energy': vqe_energy,
            'optimal_params': vqe_params,
            'optimal_pool_indices': optimized_pool_indices,
            'optimal_pool_labels': [self.excitation_block_pool[i]['pauli_label'] for i in optimized_pool_indices]
            #'optimal_ansatz': self.get_ansatz_from_pool(optimized_pool_indices)
        }
        return adapt_vqe_result, adapt_vqe_interim_data
            

def main(backend, user_messenger, **kwargs):
    """ The main runtime program entry-point.

    All the heavy-lifting is handled by the VQE_Runtime class

    Returns:
        - vqe_result (Dict[str,Union[int, float, bool, array]]):
            The optimizer output
        - interim_values (Dict[str,List[Union[float, array]]]):
            The interim energy, parameter and gradient values

        OR

        - adapt_vqe_result (Dict):
            The optimial energy, parametrization, pool indices and the final ansatz circuit
        - adapt_vqe_interim_data (Dict):
            The interim data for each ADAPT cycle, each containing self.VQE() output data
        
    """
    observable        = kwargs.pop("observable", None)
    observable_groups = kwargs.pop("observable_groups", None)
    reference_state   = kwargs.pop("reference_state", None)
    ansatz_circuit    = kwargs.pop("ansatz_circuit", None)
    ansatz_pool       = kwargs.pop("ansatz_pool", None)
    if ansatz_pool is not None:
        ansatz_pool = sum(ansatz_pool) # Required due to how qiskit serializes PauliSumOps
    
    vqe = VQE_Runtime(
        backend=backend,
        user_messenger = user_messenger,
        ansatz_pool=ansatz_pool,
        observable=observable,
        observable_groups=observable_groups,
        reference_state=reference_state
    )
    vqe.n_groups    = kwargs.pop("n_groups", 5)
    vqe.n_shots     = kwargs.pop("n_shots", 2**12)
    vqe.n_realize   = kwargs.pop("n_realize", 1)
    vqe.optimizer   = kwargs.pop("optimizer", 'SLSQP')
    vqe.opt_setting = kwargs.pop("opt_setting", {'maxiter':10})
    vqe.init_params = kwargs.pop("init_params", None)
    vqe.mitigate_errors = kwargs.pop("mitigate_errors", True)
    
    if ansatz_circuit is not None:
        return vqe.run(ansatz_circuit)
    else:
        assert(ansatz_pool is not None), 'No ansatz or operator pool has been specified'
        adapt_flag = kwargs.pop("adapt_flag", True)
        if adapt_flag:
            return vqe.ADAPT_VQE(
                reference_energy=kwargs.pop("reference_energy", None), 
                termination_threshold=kwargs.pop("adapt_termination_threshold", 0.1),
                max_cycles=kwargs.pop("adapt_max_cycles", 5)
            )
        else:
            return vqe.VQE()