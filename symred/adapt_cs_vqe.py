from symred.S3_projection import CS_VQE
from symred.symplectic_form import PauliwordOp, ObservableOp, AnsatzOp
import numpy as np
from typing import List, Dict, Tuple, Union
from itertools import combinations

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

    def ADAPT_VQE(self, 
            stabilizer_indices: List[int],
            threshold:   float = 0.01,
            maxiter:     int   = 10,
            maxterms:    int   = 10,
            param_shift: bool  = True,
            print_info:  bool  = False
        ) -> Tuple[float, PauliwordOp, np.array]:
        """ Implementation of qubit-ADAPT-VQE from https://doi.org/10.1103/PRXQuantum.2.020310
        
        Identifies a subset of terms from the input ansatz_operator that achieves the termination
        criterion (e.g. reaches chemical accuracy or gradient vector sufficiently close to zero) 

        Returns:
        The simplified ansatz operator and optimal parameter configuration

        """
        # termination messages
        vanishing_pool = '<!> The ansatz pool vanished under the noncontextual projection <!>'
        exhausted_pool = '<!> The ansatz pool has been exhausted <!>'
        small_gradient = '<!> The new ansatz term achieved the gradient threshold <!>'
        too_many_terms = '<!> ADAPT-VQE reached the maximum number of ansatz terms <!>'
        
        # start with empty ansatz to which ADAPT-VQE will append terms
        build_ansatz = []
        opt_params = []
        opt_energy=0
        ansatz_operator=None

        # perform noncontextual projection over the ansatz pool
        observable, ansatz_pool = self.project_problem(stabilizer_indices)
        observable = ObservableOp(observable.symp_matrix, observable.coeff_vec)
        # break up ansatz pool terms into list so ordering isn't scrambled
        ansatz_pool = [ansatz_pool[i] for i in range(ansatz_pool.n_terms)]
        ref_state = self.ref_state[self.free_qubit_indices]
        
        termination_criterion = 1
        # ADAPT-VQE will terminate if any of these criteria fail
        large_gradient = True # gradient is above the threshold
        below_maxterms = True # ansatz_operator is smaller than maxterms
        non_empty_pool = ansatz_pool != [] # the ansatz pool has not yet been exhausted

        if not non_empty_pool and print_info:
            print(vanishing_pool)

        while (large_gradient and non_empty_pool and below_maxterms):            
            ansatz_pool_trials = []
            trial_params = opt_params + [0]
            for index, new_ansatz_term in enumerate(ansatz_pool):
                # append ansatz term on the right with corresponding parameter zeroed
                pauli_string = list(new_ansatz_term.to_dictionary.keys())[0]
                trial_ansatz = build_ansatz + [pauli_string]
                trial_ansatz = AnsatzOp(trial_ansatz, trial_params)
                # estimate gradient w.r.t. new paramter at zero by...
                if not param_shift:
                    # measuring commutator:
                    new_ansatz_term.coeff_vec = np.ones(1)
                    obs_commutator = observable.commutator(new_ansatz_term)
                    obs_commutator = ObservableOp(obs_commutator.symp_matrix, obs_commutator.coeff_vec.imag)
                    grad = obs_commutator.ansatz_expectation(trial_ansatz, ref_state)     
                else:
                    # or parameter shift rule:    
                    grad = observable.parameter_shift_at_index(
                        param_index=-1,
                        ansatz_op  = trial_ansatz, 
                        ref_state  = ref_state
                    )
                ansatz_pool_trials.append([index, pauli_string, grad])
            # choose ansatz term with the largest gradient at zero
            best_index, best_term, best_grad = sorted(ansatz_pool_trials, key=lambda x:-abs(x[2]))[0]
            ansatz_pool.pop(best_index)
            build_ansatz.append(best_term)
            # re-optimize the full ansatz
            ansatz_operator = AnsatzOp(build_ansatz, trial_params)
            opt_out, interim = observable.VQE(
                ansatz_op=ansatz_operator,
                ref_state=ref_state,
                maxiter  = maxiter
            )
            ansatz_operator.coeff_vec = opt_out['x']
            opt_params = list(opt_out['x'])
            opt_energy = opt_out['fun']
            
            # update the gradient norm that inform the termination criterion
            if self.ref_energy is not None:
                # if reference energy given (such as FCI energy) then absolute error is selected
                termination_criterion = abs(opt_out['fun']-self.ref_energy)         
            else:
                # otherwise use best gradient value
                termination_criterion = abs(best_grad)
            
            large_gradient = termination_criterion>threshold
            non_empty_pool = ansatz_pool != []
            below_maxterms = len(build_ansatz)<maxterms
                
            if print_info:
                print(f'>>> Ansatz term #{ansatz_operator.n_terms} gradient = {termination_criterion}.')
                if not large_gradient:
                    print(small_gradient)
                if not below_maxterms:
                    print(too_many_terms)
                if not non_empty_pool:
                    print(exhausted_pool)
    
        return opt_energy, ansatz_operator

    def _greedy_search(self, 
            n_sim_qubits: int, 
            pool: set, 
            depth: int, 
            threshold:float,
            maxiter:int,
            maxterms:int,
            print_info: bool  = False,
            best_ansatz:tuple = ()):
        """ for depth d, greedily select stabilizers to relax d-many at a time, choosing
        that which minimizes the CS-VQE error. This heuristic scales as O(N^{d+1}).
        In https://doi.org/10.22331/q-2021-05-14-456 d=2 was taken.
        """
        if n_sim_qubits<depth:
            depth = n_sim_qubits
        if n_sim_qubits == 0:
            # once the number of simulation qubits is exhausted, return the stabilizer pool
            # these are the stabilizers the heuristic has chosen to enforce
            return list(pool), *best_ansatz
        else:
            num_qubits = self.operator.n_qubits - len(pool) + depth

            if print_info:
                message = f'Searching for optimal {num_qubits}-qubit contextual subspace'
                dashes = '-'*len(message)
                print(dashes);print(message);print(dashes);print()

            subspace_energies = []
            # search over combinations from the stabilizer index pool of length d (the depth)
            for relax_indices in combinations(pool, r=depth):
                relax_indices = list(relax_indices)
                stab_indices = list(pool.difference(relax_indices))
                if print_info:
                    print(f'Testing stabilizer indices {stab_indices}')
                # perform the stabilizer subsapce projection and compute energy via ADAPT-VQE
                energy, ansatz = self.ADAPT_VQE(
                    stabilizer_indices=stab_indices,
                    threshold=threshold,
                    maxiter=maxiter,
                    maxterms=maxterms,
                    print_info=print_info
                )
                subspace_energies.append([relax_indices, energy, ansatz])
                if print_info:
                    print()
            # choose the best error and remove the corresponding stabilizer indices from the pool
            best_relax_indices, best_energy, best_ansatz = sorted(
                subspace_energies, 
                key=lambda x:x[1]
            )[0]
            new_pool = pool.difference(best_relax_indices)

            if print_info:
                message = f'{self.operator.n_qubits-len(pool)+depth}-qubit CS-VQE energy is {best_energy: .8f} for stabilizer indices {new_pool}'
                print(message);print()

            # perform an N-d qubit search over the reduced pool 
            return self._greedy_search(n_sim_qubits = n_sim_qubits-depth,
                            pool=new_pool, 
                            depth=depth,
                            threshold=threshold,
                            maxiter=maxiter,
                            maxterms=maxterms,
                            print_info=print_info,
                            best_ansatz = (best_energy, best_ansatz))

    def adaptive_greedy_search(self, 
            n_sim_qubits, 
            depth=1, 
            threshold=0.01,
            maxiter=10,
            maxterms=10,
            print_info=False
        ):
        """ wraps the _greedy_search recursive method for stabilizer relaxation ordering
        """
        if print_info:
            if self.ref_energy is not None:
                message = 'Using error w.r.t. reference energy as termination criterion'
            else:
                message = 'Using best gradient as termination criterion'
            dashes = '-'*len(message)
            print(dashes);print(message);print(dashes);print()
        
        # take the full stabilizer pool
        all_indices = set(range(self.operator.n_qubits))
        return self._greedy_search(
            n_sim_qubits=n_sim_qubits, 
            pool=all_indices, 
            depth=depth,
            threshold=threshold,
            maxiter=maxiter,
            maxterms=maxterms,
            print_info=print_info
        )
