from symred.S3_projection import CS_VQE
from symred.symplectic_form import PauliwordOp
from quantumtools.variational import VariationalAlgorithm
import numpy as np
from typing import List, Dict, Tuple, Union
from itertools import combinations
from scipy.linalg import expm
from scipy.sparse import csr_matrix

class ADAPT_CS_VQE(CS_VQE):
    """
    """
    def __init__(self, 
        operator: PauliwordOp,
        ansatz_pool: PauliwordOp,
        ref_state: np.array = None,
        ref_energy: float = None
        ) -> None:
        """
        """
        self.ansatz_pool = ansatz_pool
        self.ref_energy = ref_energy
        # build the CS-VQE model
        super().__init__(operator = operator,
                        ref_state = ref_state,
                        target_sqp='Z')

    def project_problem(self, 
            stabilizer_indices: List[int]
        ) -> Tuple[PauliwordOp, PauliwordOp]:
        """
        """
        proj_operator = self.contextual_subspace_projection(
            stabilizer_indices=stabilizer_indices
        )
        proj_anz_pool = self.contextual_subspace_projection(
            stabilizer_indices = stabilizer_indices,
            aux_operator = self.ansatz_pool
        )
        return proj_operator, proj_anz_pool

    def exact_expectation_value(self, 
        observable: PauliwordOp, 
        parameters: np.array, 
        ansatz_op: PauliwordOp, 
        ref_state: np.array
        ) -> float:
        """ The method of evaluating expectation values, for example it might be:
        - exact, meaning the Hamiltonian is diagonalized (expensive!)
        - computed via Hamiltonian averaging from many circuit shots
        - or plugged into other quantum computing packages, Qiskit, Cirq, myQLM etc.
        """
        # Hartree-Fock state:
        ref_index = int(''.join([str(i) for i in ref_state]), 2)
        ref_vector = np.zeros(2**observable.n_qubits)
        ref_vector[ref_index]=1
        ref_vector = csr_matrix(ref_vector).T

        # Exponentiated ansatz operator
        A_op = ansatz_op.copy()
        A_op.coeff_vec = 1j*np.array(parameters)
        A_mat = A_op.to_sparse_matrix.toarray()
        expA = expm(A_mat)
        expA = csr_matrix(expA)

        # Hamiltonian
        H = observable.to_sparse_matrix

        # Compute expectation value
        energy = (ref_vector.T @ expA.H @ H @ expA @ ref_vector)[0,0].real

        return energy

    def parameter_shift(self,
        param_index: int,
        observable: PauliwordOp, 
        parameters: np.array, 
        ansatz_op: PauliwordOp, 
        ref_state: np.array
        ):
        """ return the first-order derivative at x w.r.t. to the observable operator
        """
        assert(param_index<=len(parameters)), 'Index outside provided parameters'
        diff_vec = np.zeros(len(parameters))
        diff_vec[param_index] = np.pi/4

        shift_upper = self.exact_expectation_value(
            observable = observable, 
            parameters = parameters+diff_vec,
            ansatz_op  = ansatz_op,
            ref_state  = ref_state
        )
        shift_lower = self.exact_expectation_value(
            observable = observable, 
            parameters = parameters-diff_vec,
            ansatz_op  = ansatz_op,
            ref_state  = ref_state
        )
        grad = shift_upper - shift_lower
        return grad

    def _ADAPT_VQE(self, 
            stabilizer_indices: List[int],
            threshold: float  = 0.01,
            maxiter: int = 10,
            param_shift: bool = True
        ) -> Tuple[float, PauliwordOp, np.array]:
        """ Implementation of qubit-ADAPT-VQE from https://doi.org/10.1103/PRXQuantum.2.020310
        
        Identifies a subset of terms from the input ansatz_operator that achieves the termination
        criterion (e.g. reaches chemical accuracy or gradient vector sufficiently close to zero) 

        Returns:
        The simplified ansatz operator and optimal parameter configuration

        """
        build_ansatz = []
        opt_params = []
        opt_energy=0
        ansatz_operator=None
        observable, ansatz_pool = self.project_problem(stabilizer_indices)
        # break up ansatz pool terms into list so ordering isn't scrambled
        ansatz_pool = [ansatz_pool[i] for i in range(ansatz_pool.n_terms)]
        if ansatz_pool == []:
            print('<!> Ansatz pool vanished under the noncontextual projection <!>\n')
        ref_state = self.ref_state[self.free_qubit_indices]
        termination_criterion = 1
        
        while (termination_criterion>threshold and ansatz_pool != []):
            ansatz_pool_trials = []
            trial_params = opt_params + [0]
            
            for index, new_ansatz_term in enumerate(ansatz_pool):

                # append ansatz term on the right with corresponding parameter zeroed
                pauli_string = list(new_ansatz_term.to_dictionary.keys())[0]
                trial_ansatz = build_ansatz + [pauli_string]
                trial_ansatz = PauliwordOp(trial_ansatz, trial_params)
                #print('>>> Testing ansatz:', trial_ansatz)
                # estimate gradient w.r.t. new paramter at zero by...
                if not param_shift:
                    # measuring commutator:
                    new_ansatz_term.coeff_vec = np.ones(1)
                    obs_commutator = observable.commutator(new_ansatz_term)
                    grad = self.exact_expectation_value(
                        observable = obs_commutator, 
                        parameters = trial_params, 
                        ansatz_op  = trial_ansatz, 
                        ref_state  = ref_state
                    )     
                else:
                    # or parameter shift rule:    
                    grad = self.parameter_shift(
                        param_index=-1,
                        observable = observable, 
                        parameters = trial_params, 
                        ansatz_op  = trial_ansatz, 
                        ref_state  = ref_state
                    )
                ansatz_pool_trials.append([index, pauli_string, grad])

            # choose ansatz term with the largest gradient at zero
            best_index, best_term, best_grad = sorted(ansatz_pool_trials, key=lambda x:-abs(x[2]))[0]
            ansatz_pool.pop(best_index)
            build_ansatz.append(best_term)

            # re-optimize the full ansatz
            ansatz_operator = PauliwordOp(build_ansatz, trial_params)
            vqe = VariationalAlgorithm(
                observable, 
                ansatz_operator, 
                ref_state,
            )
            opt_out, interim = vqe.VQE(optimizer='SLSQP', exact=True, maxiter=maxiter)
            opt_params = list(opt_out['x'])
            opt_energy = opt_out['fun']

            # update the gradient norm that inform the termination criterion
            if self.ref_energy is not None:
                # if reference energy given (such as FCI energy) then absolute error is selected
                termination_criterion = abs(opt_out['fun']-self.ref_energy)         
            else:
                # otherwise use best gradient value
                termination_criterion = abs(best_grad)
            
            print(f'>>> {ansatz_operator.n_terms}-term ansatz termination criterion: {termination_criterion} < {threshold}? {termination_criterion<threshold}')
            if ansatz_pool == []:
                print('<!> Ansatz pool has been exhausted <!>\n')
        return opt_energy, ansatz_operator, opt_params


    def _greedy_search(self, 
            n_sim_qubits: int, 
            pool: set, 
            depth: int, 
            threshold:float,
            maxiter:int,
            print_errors: bool):
        """ for depth d, greedily select stabilizers to relax d-many at a time, choosing
        that which minimizes the CS-VQE error. This heuristic scales as O(N^{d+1}).
        In https://doi.org/10.22331/q-2021-05-14-456 d=2 was taken.
        """
        if n_sim_qubits<depth:
            depth = n_sim_qubits
        if n_sim_qubits == 0:
            # once the number of simulation qubits is exhausted, return the stabilizer pool
            # these are the stabilizers the heuristic has chosen to enforce
            return pool
        else:
            num_qubits = self.operator.n_qubits - len(pool) + depth

            message = f'Searching for optimal {num_qubits}-qubit contextual subspace'
            dashes = '-'*len(message)
            print(dashes);print(message);print(dashes);print()

            subspace_energies = []
            # search over combinations from the stabilizer index pool of length d (the depth)
            for relax_indices in combinations(pool, r=depth):
                relax_indices = list(relax_indices)
                stab_indices = list(pool.difference(relax_indices))
                print(f'Testing stabilizer indices {stab_indices}')
                # perform the stabilizer subsapce projection and compute energy via ADAPT-VQE
                energy, ansatz, parameters = self._ADAPT_VQE(
                    stabilizer_indices=stab_indices,
                    threshold=threshold,
                    maxiter=maxiter
                )
                subspace_energies.append([relax_indices, energy, ansatz, parameters])
            # choose the best error and remove the corresponding stabilizer indices from the pool
            best_relax_indices, best_energy, best_ansatz, best_parameters = sorted(
                subspace_energies, 
                key=lambda x:x[1]
            )[0]
            new_pool = pool.difference(best_relax_indices)

            print()
            print(f'{self.operator.n_qubits-len(pool)+depth}-qubit CS-VQE energy is {best_energy: .8f} for stabilizer indices {new_pool}')
            print()
            # perform an N-d qubit search over the reduced pool 
            return self._greedy_search(n_sim_qubits = n_sim_qubits-depth,
                            pool=new_pool, 
                            depth=depth,
                            threshold=threshold,
                            maxiter=maxiter,
                            print_errors=print_errors)

    def adaptive_greedy_search(self, 
            n_sim_qubits, 
            depth=1, 
            threshold=0.01,
            maxiter=10,
            print_errors=False
        ):
        """ wraps the _greedy_search recursive method for stabilizer relaxation ordering
        """
        if self.ref_energy is not None:
            message = 'Using error w.r.t. reference energy as termination criterion'
        else:
            message = 'Using best gradient as termination criterion'

        print('-'*len(message))
        print(message)
        print('-'*len(message))
        print()
        
        # take the full stabilizer pool
        all_indices = set(range(self.operator.n_qubits))
        return list(
            self._greedy_search(
                n_sim_qubits=n_sim_qubits, 
                pool=all_indices, 
                depth=depth,
                threshold=threshold,
                maxiter=maxiter,
                print_errors=print_errors
            )
        )

