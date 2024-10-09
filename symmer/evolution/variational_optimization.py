from cached_property import cached_property
from qiskit.quantum_info import Statevector
# from qiskit.opflow import CircuitStateFn # old qiskit function replaced by Statevector
from qiskit import QuantumCircuit
from symmer import process, QuantumState, PauliwordOp
from symmer.operators.utils import (
    symplectic_to_string, safe_PauliwordOp_to_dict, safe_QuantumState_to_dict
)
from symmer.evolution import PauliwordOp_to_QuantumCircuit, get_CNOT_connectivity_graph, topology_match_score
from networkx.algorithms.cycles import cycle_basis
from scipy.optimize import minimize
from scipy.sparse import csc_array
from copy import deepcopy
import numpy as np
from typing import *

class VQE_Driver:
    """
    expectation value method one of the following choices:
        - symbolic_direct: uses symmer to compute <psi|H|psi> directly using the QuantumState class
        - symbolic_projector: computes expval by projecting onto +-1 eigenspaces of observable terms
        - observable_rotation: implements the circuit as rotations applied to the observable
        - sparse_array: direct calcaultion by converting observable/state to sparse array
        - dense_array: direct calcaultion by converting observable/state to dense array
    
    Attributes:
        expectation_eval (str): expectation value method. Its default value is 'symbolic_direct'.
        verbose (bool): If True, prints out useful information during computation. By default it is set to 'True'.
    """
    expectation_eval = 'sparse_array'
    # prints out useful information during computation:
    verbose = True
       
    def __init__(self,
        observable: PauliwordOp,
        ansatz_circuit: QuantumCircuit = None,
        excitation_ops: PauliwordOp = None,
        ref_state: QuantumState = None
        ) -> None:
        """
        Args:
            observable (PauliwordOp): Observables
            ansatz_circuit (QuantumCircuit): Ansatz Circuit. By default, it's set to 'None'.
            excitation_ops (PauliwordOp): Excitation Operators. By default, it's set to 'None'.
            ref_state (QuantumState): Reference State. By default, it's set to 'None'.
        """
        self.observable = observable
        self.ref_state = ref_state
        # observables must have real coefficients over the Pauli group:
        assert np.all(self.observable.coeff_vec.imag == 0), 'Observable not Hermitian'
        
        if excitation_ops is not None:
            self.prepare_for_evolution(excitation_ops)
        else:
            self.circuit = ansatz_circuit

    def prepare_for_evolution(self, excitation_ops: PauliwordOp) -> None:
        """ 
        Save the excitation generators and construct corresponding ansatz circuit.

        Args:
            excitation_ops (PauliwordOp): Excitation Operators.
        """
        self.excitation_generators = PauliwordOp(
            excitation_ops.symp_matrix, np.ones(excitation_ops.n_terms)
        )
        self.circuit = PauliwordOp_to_QuantumCircuit(
                PwordOp=self.excitation_generators, ref_state=self.ref_state, bind_params=False
        )

    def get_state(self, 
            evolution_obj: Union[QuantumCircuit, PauliwordOp], 
            x: np.array
        ) -> Union[np.array, QuantumState, List[Tuple[PauliwordOp, float]]]:
        """ 
        Args:
            evolution_obj (Union[QuantumCircuit, PauliwordOp]): Evoluation object is either a Quantum Circuit (QuantumCircuit) or a Excitation Generating Set(PauliwordOp).

        Returns:
            Given a quantum circuit or excitation generating set, return the relevant state-type object:
            - Array (np.array) of the QuantumCircuit (for sparse/dense array methods).
            - Quantum State (QuantumState) representation of the QuantumCircuit (for symbolic methods).
            - Rotations of the form [(generator, angle)] for the observable_rotation expectation_eval method.
        """
        if self.expectation_eval == 'observable_rotation':
            return list(zip(evolution_obj, -2*x))
        else:
            state = Statevector(evolution_obj.assign_parameters(x)).data.reshape([-1,1])
            if self.expectation_eval == 'dense_array':
                return state
            elif self.expectation_eval == 'sparse_array':
                return csc_array(state)
            elif self.expectation_eval.find('symbolic') != -1:
                return QuantumState.from_array(state)
        
    def _f(self, 
           observable: PauliwordOp, 
           state: Union[np.array, QuantumState, List[Tuple[PauliwordOp, float]]]
        ) -> float:
        """ 
        Given an observable and state in the relevant form for the
        expectation value method, calculate the expectation value and return.

        Args:
            observable (PauliwordOp): Observable
            state: State-type object. It can be:
                - Array (np.array) of the QuantumCircuit (for sparse/dense array methods).
                - Quantum State (QuantumState) representation of the QuantumCircuit (for symbolic methods).
                - Rotations of the form [(generator, angle)] for the observable_rotation expectation_eval method.
        
        Returns:
            Expectation Value (float)
        """
        if self.expectation_eval == 'dense_array':
            return (state.conjugate().T @ observable.to_sparse_matrix.toarray() @ state)[0,0].real
        elif self.expectation_eval == 'sparse_array':
            return (state.conjugate().T @ observable.to_sparse_matrix @ state)[0,0].real
        elif self.expectation_eval == 'symbolic_projector':
            return observable.expval(state).real
        elif self.expectation_eval == 'symbolic_direct':
            return (state.dagger * observable * state).real   
        elif self.expectation_eval == 'observable_rotation':
            return (self.ref_state.dagger * observable.perform_rotations(state) * self.ref_state).real
        
    def f(self, x: np.array) -> float:
        """ 
        Given a parameter vector, bind to the circuit and retrieve expectation value.

        Args:
            x (np.array): Parameter vector

        Returns:
            Expectation Value (float)
        """
        if self.expectation_eval == 'observable_rotation':
            state = self.get_state(self.excitation_generators, x)
        else:
            state = self.get_state(self.circuit, x)
        return self._f(self.observable, state)
        
    def partial_derivative(self, x: np.array, param_index: int) -> float:
        """ 
        Get the partial derivative with respect to an ansatz parameter
        by the parameter shift rule.

        Args:
            x (np.array): Parameter vector
            param_index (int): Prarameter index
        
        Returns:
            Partial derivative(float) with respect to an ansatz parameter.
        """
        x_upper = x.copy(); x_upper[param_index]+=np.pi/4
        x_lower = x.copy(); x_lower[param_index]-=np.pi/4
        return self.f(x_upper) - self.f(x_lower)
    
    def gradient(self, x: np.array) -> np.array:
        """ 
        Get the ansatz parameter gradient, i.e. the vector of partial derivatives.

        Args:
            x (np.array): Parameter vector
        
        Returns:
            Ansatz parameter gradient (np.array)
        """
        if self.expectation_eval.find('projector') == -1:
            @process.parallelize
            def f(index, param):
                return self.partial_derivative(param,index)
            grad_vec = f(range(self.circuit.num_parameters), x)

        else:
            grad_vec = [self.partial_derivative(x, i) for i in range(self.circuit.num_parameters)]
        
        return np.asarray(grad_vec)
    
    def run(self, x0:np.array=None, **kwargs):
        """ 
        Run the VQE routine.
        
        Args:
            x0 (np.array): Parameter vector
        """
        if x0 is None:
            x0 = np.random.random(self.circuit.num_parameters)
        
        vqe_history = {'params':{}, 'energy':{}, 'gradient':{}}

        # set up a counter to keep track of optimization steps this is important 
        # as some optimizers do not compute gradients at each optimization step and 
        # therefore must be labeled to match with the correct iteration later on
        global counter
        counter = -1
        def get_counter(increment=True):
            global counter
            if increment:
                counter += 1
            return counter

        # wrap VQE_Driver.f() for the optimizer and store the interim values
        def fun(x):    
            counter = get_counter(increment=True)
            energy  = self.f(x)
            vqe_history['params'][counter] = tuple(x)
            vqe_history['energy'][counter] = energy
            if self.verbose:
                print(f'Optimization step {counter: <2}:\n\t Energy = {energy}')
            return energy

        # wrap VQE_Driver.gradient() for the optimizer and store the interim values
        def jac(x):
            counter = get_counter(increment=False)
            grad    = self.gradient(x)
            vqe_history['gradient'][counter] = tuple(grad)
            if self.verbose:
                print(f'\t    |∆| = {np.linalg.norm(grad)}')
            return grad
        
        if self.verbose:
            print('VQE simulation commencing...\n')
        opt_out = minimize(
            fun=fun, jac=jac, x0=x0, **kwargs
        )
        return serialize_opt_data(opt_out), vqe_history

class ADAPT_VQE(VQE_Driver):
    """ 
    Performs qubit-ADAPT-VQE (https://doi.org/10.1103/PRXQuantum.2.020310), a 
    variant of ADAPT-VQE (https://doi.org/10.1038/s41467-019-10988-2) that takes 
    its excitation pool as Pauli operators (mapped via some transformation such 
    as Jordan-Wigner) instead of the originating fermionic operators.

    Attributes:
        derivative_eval (str): Method which is to be used to calculate the operator pool derivatives.
        TETRIS (bool): If True, TETRIS-ADAPT-VQE is performed. By default it is set to False.
        topology_aware (bool): If True, Hardware-Aware ADAPT-VQE is performed. By default it is set to True.
        topology_bias (float): Bias value used in Hardware-Aware ADAPT-VQE. It's default value is 1.
    """
    # method by which to calculate the operator pool derivatives, either
    # commutators: compute the commutator of the observable with each pool element
    # param_shift: use the parameter shift rule, requiring two expectation values per derivative
    derivative_eval = 'param_shift'
    # we have alost implemented TETRIS-ADAPT-VQE as per https://doi.org/10.48550/arXiv.2209.10562
    # that aims to reduce circuit-depth in the ADAPT routine by adding multiple excitation terms
    # per cycle that are supported on distinct qubit positions.
    TETRIS = False
    topology_aware = False
    topology_bias = 1
    topology = None
    subgraph_match_depth = 3
    
    def __init__(self,
        observable: PauliwordOp,
        excitation_pool: PauliwordOp = None,
        ref_state: QuantumState = None
        ) -> None:
        """
        Args:
            observable (PauliwordOp): Observable
            excitation_pool (PauliwordOp): Excitation pool as Pauli operators. By default, it's set to 'None'.
            ref_state (QuantumState): Reference State. By default, it's set to 'None'.
        """
        super().__init__(
            observable     = observable,
            excitation_ops = PauliwordOp.empty(observable.n_qubits),
            ref_state      = ref_state
        )
        self.excitation_pool = PauliwordOp(
            excitation_pool.symp_matrix, np.ones(excitation_pool.n_terms)
        )
        self.adapt_operator = PauliwordOp.empty(observable.n_qubits)
        self.opt_parameters = []
        self.current_state  = None
      
    @cached_property
    def commutators(self) -> List[PauliwordOp]:
        """ 
        List of commutators [H, P] where P is some operator pool element.

        Returns:
            List of commutators [H, P]
        """
        @process.parallelize
        def f(P, obs):
            return obs.commutator(P)*1j
        commutators = f(self.excitation_pool, self.observable)
        return commutators
        
    def _derivative_from_commutators(self, index: int) -> float:
        """ 
        Calculate derivative using the commutator method.

        Args:
            index (int): Index

        Returns:
            Derivative (float) using the commutator method.
        """
        assert self.current_state is not None
        return self._f(observable=self.commutators[index], state=self.current_state) 
    
    def _derivative_from_param_shift(self, index):
        """ 
        Calculate the derivative using the parameter shift rule.

         Args:
            index (int): Index

        Returns:
            Derivative (float) using the parameter shift rule.
        """
        adapt_op_temp = self.adapt_operator.append(self.excitation_pool[index])
        circuit_temp = PauliwordOp_to_QuantumCircuit(
            PwordOp=adapt_op_temp, ref_state=self.ref_state, bind_params=False)
        upper_state = self.get_state(circuit_temp, np.append(self.opt_parameters, +np.pi/4))
        lower_state = self.get_state(circuit_temp, np.append(self.opt_parameters, -np.pi/4))
        return self._f(self.observable,upper_state) - self._f(self.observable,lower_state)

    def pool_gradient(self):
        """ 
        Get the operator pool gradient by calculating the derivative with respect to
        each element of the pool. This is parallelized for all but the symbolic_projector
        expectation value calculation method as that is already multiprocessed and therefore
        would result in nested daemonic processes.

        Returns:
            Operator pool gradient (np.array)
        """
        if self.derivative_eval == 'commutators':
            self.commutators # to ensure this has been cached, else nested daemonic process occurs            
            if self.expectation_eval == 'observable_rotation':
                self.current_state = self.get_state(self.adapt_operator, self.opt_parameters)
            else:
                circuit_temp = PauliwordOp_to_QuantumCircuit(
                    PwordOp=self.adapt_operator, ref_state=self.ref_state, bind_params=False)
                self.current_state = self.get_state(circuit_temp, self.opt_parameters)
            if self.expectation_eval in ['sparse_array', 'symbolic_direct', 'observable_rotation']:
                # the commutator method may be parallelized since the state is constant
                @process.parallelize
                def f(index, obs):
                    return obs._derivative_from_commutators(index)
                gradient = f(range(self.excitation_pool.n_terms), self)
            else:
                # ... unless using symbolic_projector since this is multiprocessed
                gradient = list(map(self._derivative_from_commutators, range(self.excitation_pool.n_terms)))
        
        elif self.derivative_eval == 'param_shift':
            # not parallelizable due the Statevector already using multiprocessing! 
            gradient = list(map(self._derivative_from_param_shift, range(self.excitation_pool.n_terms)))
        
        else:
            raise ValueError('Unrecognised derivative_eval method')
        
        return np.asarray(gradient)
    
    def pool_score(self):
        """ 
        Score the operator pool with respect to gradients and topology likeness.
        """
        scores = abs(self.pool_gradient())

        if self.topology_aware:
            assert self.topology is not None, 'No hardware topology specified'
            # Hardware-Aware ADAPT-VQE favours circuits that match the target topology closely
            topology_scores = []
            for index in range(self.excitation_pool.n_terms):
                adapt_op_temp = self.adapt_operator.append(self.excitation_pool[index])
                topology_scores.append(
                    topology_match_score(adapt_op_temp, self.topology, max_depth=self.subgraph_match_depth)
                )
            scores *= np.power(np.array(topology_scores), self.topology_bias)
        
        return scores
        
    def append_to_adapt_operator(self, excitations_to_append: List[PauliwordOp]):
        """ 
        Append the input term(s) to the expanding adapt_operator.
        """
        for excitation in excitations_to_append:
            if ~np.any(self.adapt_operator.symp_matrix):
                self.adapt_operator += excitation
            else:
                self.adapt_operator = self.adapt_operator.append(excitation)
        
    def optimize(self, 
            max_cycles:int=10, gtol:float=1e-3, atol:float=1e-10, 
            target:float=0, target_error:float=1e-3
        ):
        """ 
        Perform the ADAPT-VQE optimization

        Args:
            gtol: gradient throeshold below which optimization will terminate
            atol: if the difference between successive expectation values is below this threshold, terminate
            max_cycles: maximum number of ADAPT cycles to perform
            target: if a target energy is known, this may be specified here
            target_error: the absoluate error threshold with respect to the target energy 
        """
        interim_data = {'history':[]}
        adapt_cycle=1
        gmax=1
        anew=1
        aold=0
        
        while (
                gmax>gtol and adapt_cycle<=max_cycles and 
                abs(anew-aold)>atol and abs(anew-target)>target_error
            ):
            # save the previous gmax to compare for the gdiff check
            aold = deepcopy(anew)
            # calculate gradient across the pool and select term with the largest derivative
            scores = self.pool_score()
            grad_rank = list(map(int, np.argsort(scores)[::-1]))
            gmax = scores[grad_rank[0]]

            # TETRIS-ADAPT-VQE
            if self.TETRIS:
                new_excitation_list = []
                support_mask = np.zeros(self.observable.n_qubits, dtype=bool)
                for i in grad_rank:
                    new_excitation = self.excitation_pool[i]
                    support_exists = (new_excitation.X_block | new_excitation.Z_block) & support_mask
                    if ~np.any(support_exists):
                        new_excitation_list.append(new_excitation)
                        support_mask = support_mask | (new_excitation.X_block | new_excitation.Z_block)
                    if np.all(support_mask) or scores[i] < gtol:
                        break
            else:
                new_excitation_list = [self.excitation_pool[grad_rank[0]]]
                
            # append new term(s) to the adapt_operator that stores our ansatz as it expands
            n_new_terms = len(new_excitation_list)
            self.append_to_adapt_operator(new_excitation_list)
                        
            if self.verbose:
                print('-'*39)
                print(f'ADAPT cycle {adapt_cycle}\n')
                print(f'Largest pool derivative ∂P∂θ = {gmax: .5f}\n')
                print('Selected excitation generator(s):\n')
                for op in new_excitation_list:
                    print(f'\t{symplectic_to_string(op.symp_matrix[0])}')
                print('\n', '-'*39)
            
            # having selected a new term to append to the ansatz, reoptimize with VQE
            self.prepare_for_evolution(self.adapt_operator)
            opt_out, vqe_hist = self.run(
                x0=np.append(self.opt_parameters, [0]*n_new_terms), method='BFGS'
            )
            interim_data[adapt_cycle] = {
                'output':opt_out, 'history':vqe_hist, 'gmax':gmax, 
                'excitation': [symplectic_to_string(t.symp_matrix[0]) for t in new_excitation_list]
            }
            anew = opt_out['fun']
            interim_data['history'].append(anew)
            if self.verbose:
                print(F'\nEnergy at ADAPT cycle {adapt_cycle}: {anew: .5f}\n')
            self.opt_parameters = opt_out['x']
            adapt_cycle+=1

        return {
            'result': opt_out, 
            'interim_data': interim_data,
            'ref_state': safe_QuantumState_to_dict(self.ref_state),
            'adapt_operator': [symplectic_to_string(t) for t in self.adapt_operator.symp_matrix]
        }
    
def serialize_opt_data(opt_data):
    return {
        'message':opt_data.message, 'success':opt_data.success, 'status':opt_data.status,
        'fun':opt_data.fun, 'x':tuple(opt_data.x),'jac':tuple(opt_data.jac),
        'nit':opt_data.nit, 'nfev':opt_data.nfev,'njev':opt_data.njev,
    }