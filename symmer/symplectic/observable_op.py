import numpy as np
from typing import Dict, List, Union
from functools import reduce
from cached_property import cached_property
from scipy.optimize import minimize
from scipy.sparse.linalg import expm
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from symmer.symplectic import PauliwordOp, QuantumState, AnsatzOp, symplectic_to_string
from qiskit.opflow import CircuitStateFn
from qiskit import BasicAer, execute

class ObservableGraph(PauliwordOp):
    # TODO
    def __init__(self,
            operator:   Union[List[str], Dict[str, float], np.array],
            coeff_list: Union[List[complex], np.array] = None):
        super().__init__(operator, coeff_list)

    def build_graph(self, edge_relation='C', weighted=None):

        if edge_relation =='AC':
            # commuting edges
            adjacency_mat = np.bitwise_not(self.adjacency_matrix)

            # removes self adjacency for graph
            np.fill_diagonal(adjacency_mat, 0)

        elif edge_relation =='C':
            # anticommuting edges
            adjacency_mat = self.adjacency_matrix
            np.fill_diagonal(adjacency_mat, 0)

        elif edge_relation =='QWC':
            adjacency_mat = np.zeros((self.n_terms, self.n_terms))
            for i in range(self.n_terms):
                for j in range(i+1, self.n_terms):
                    Pword_i = self.symp_matrix[i]
                    Pword_j = self.symp_matrix[j]

                    self_I = np.bitwise_or(Pword_i[:self.n_qubits], Pword_i[self.n_qubits:]).astype(bool)
                    Pword_I = np.bitwise_or(Pword_j[:self.n_qubits], Pword_j[self.n_qubits:]).astype(bool)

                    # Get the positions where neither self nor Pword have I acting on them
                    unique_non_I_locations = np.bitwise_and(self_I, Pword_I)

                    # check non I operators are the same!
                    same_Xs = np.bitwise_not(
                        np.bitwise_xor(Pword_i[:self.n_qubits][unique_non_I_locations],
                                       Pword_j[:self.n_qubits][unique_non_I_locations]).astype(
                            bool))
                    same_Zs = np.bitwise_not(
                        np.bitwise_xor(Pword_i[self.n_qubits:][unique_non_I_locations],
                                       Pword_j[self.n_qubits:][unique_non_I_locations]).astype(
                            bool))

                    if np.all(same_Xs) and np.all(same_Zs):
                        adjacency_mat[i,j] = adjacency_mat[j,i] = 1
                    else:
                        continue
        else:
            raise ValueError(f'unknown edge relation: {edge_relation}')

        graph = nx.from_numpy_matrix(adjacency_mat)

        return graph

    def clique_cover(self, clique_relation, colouring_strategy, colour_interchange=False,
                     plot_graph=False, with_node_label=False, node_sizes=True):

        if clique_relation == 'AC':
            graph = self.build_graph(edge_relation='C')
        elif clique_relation == 'C':
            graph = self.build_graph(edge_relation='AC')
        elif clique_relation == 'QWC':
            graph = self.build_graph(edge_relation='QWC')
            graph = nx.complement(graph)
        else:
            raise ValueError(f'unknown clique relation: {clique_relation}')

        # keys give symplectic row index and value gives colour of clique
        greedy_colouring_output_dic = nx.greedy_color(graph,
                                                      strategy=colouring_strategy,
                                                      interchange=colour_interchange)

        unique_colours = set(greedy_colouring_output_dic.values())

        clique_dict = {}
        for Clique_ind in unique_colours:
            clique_Pword_symp = []
            clique_coeff_symp = []
            for sym_row_ind, clique_id in greedy_colouring_output_dic.items():
                if clique_id == Clique_ind:
                    clique_Pword_symp.append(self.symp_matrix[sym_row_ind,:])
                    clique_coeff_symp.append(self.coeff_vec[sym_row_ind])

            clique = PauliwordOp(np.array(clique_Pword_symp, dtype=int),
                                 clique_coeff_symp)

            clique_dict[Clique_ind] = clique

        if plot_graph:
            possilbe_colours = cm.rainbow(np.linspace(0, 1, len(unique_colours)))
            colour_list = [possilbe_colours[greedy_colouring_output_dic[node_id]] for node_id in graph.nodes()]
            # print(colour_list)
            # print([symplectic_to_string(self.symp_matrix[row_ind]) for row_ind in graph.nodes])
            self.draw_graph(graph, with_node_label=with_node_label, node_sizes=node_sizes, node_colours=colour_list)

        return clique_dict

    def draw_graph(self, graph_input, with_node_label=False, node_sizes=True, node_colours=None):

        if node_sizes:
            node_sizes = 200 * np.abs(np.round(self.coeff_vec)) + 1
            options = {
                'node_size': node_sizes,
                'node_color': 'r' if node_colours is None else node_colours
                     }
        else:
            options = {
                'node_color': 'r' if node_colours is None else node_colours
                     }

        plt.figure()
        pos = nx.circular_layout(graph_input)
        # # pos = nx.spring_layout(graph_input)
        # pos = nx.nx_pydot.graphviz_layout(graph_input)

        nx.draw_networkx_nodes(graph_input,
                               pos,
                               nodelist=list(graph_input.nodes),
                               **options)
        nx.draw_networkx_edges(graph_input, pos,
                               width=1.0,
                               # alpha=0.5,
                               nodelist=list(graph_input.nodes),
                               )

        if with_node_label:
            labels = {row_ind: symplectic_to_string(self.symp_matrix[row_ind]) for row_ind in graph_input.nodes}
            nx.draw_networkx_labels(graph_input,
                                    pos,
                                    labels,
                                    font_size=18)

        # plt.savefig('G_raw', dpi=300, transparent=True, )  # edgecolor='black', facecolor='white')
        plt.show()
        return None

class ObservableOp(PauliwordOp):
    """ Based on PauliwordOp and introduces functionality for evaluating expectation values
    """
    # here we define the expectation evaluation parameters and can be updated by the user
    evaluation_method = 'statevector' # one of statevector, trotter_rotations or sampled
    trotter_number = 1 # the number of repetition in the QuantumCircuit
    n_shots = 2**10 # number of samples taken from each QWC group circuit
    n_realizations = 1 # number of expectation evaluations in average for sampled method
    backend = BasicAer.get_backend('qasm_simulator')

    def __init__(self,
        operator:   Union[List[str], Dict[str, float], np.array],
        coeff_vec: Union[List[complex], np.array] = None
        ) -> None:
        """
        """
        super().__init__(operator, coeff_vec)
        assert(np.all(self.coeff_vec.imag==0)), 'Coefficients must be real, ensuring the operator is Hermitian'
        self.coeff_vec = self.coeff_vec.real

    def Z_basis_expectation(self, ref_state: np.array) -> float:
        """ Provided a single Pauli-Z basis state, computes the expectation value of the operator
        """
        assert(set(ref_state).issubset({0,1})), f'Basis state must consist of binary elements, not {set(ref_state)}'
        assert(len(ref_state)==self.n_qubits), f'Number of qubits {len(ref_state)} in the basis state incompatible with {self.n_qubits}'
        mask_diagonal = np.where(np.all(self.X_block==0, axis=1))
        measurement_signs = (-1)**np.sum(self.Z_block[mask_diagonal] & ref_state, axis=0)
        return np.sum(measurement_signs * self.coeff_vec[mask_diagonal]).real

    def _ansatz_expectation_trotter_rotations(self, 
            ansatz_op: AnsatzOp, 
            ref_state: np.array
        ):
        """ Exact expectation value - expensive! Trotterizes the ansatz operator and applies the terms as
        Pauli rotations to the observable operator, resulting in an exponential increase in the number of terms
        """
        pauli_rotations = [PauliwordOp(row, [1]) for row in ansatz_op.symp_matrix]*self.trotter_number
        angles = -2*np.tile(ansatz_op.coeff_vec, self.trotter_number)/self.trotter_number

        trotterized_observable = self.perform_rotations(zip(pauli_rotations[::-1], angles[::-1]))
        trotterized_observable = ObservableOp(trotterized_observable.symp_matrix, trotterized_observable.coeff_vec)

        return trotterized_observable.Z_basis_expectation(ref_state)

    def _ansatz_expectation_statevector(self, 
            ansatz_op: AnsatzOp, 
            ref_state: np.array,
            sparse = False
        ) -> float:
        """ Exact expectation value - expensive! Converts the ansatz operator to a sparse vector | psi >
        and return the quantity < psi | Observable | psi >
        """
        if sparse:
            # sparse multiplication does not scale nicely with number of qubits
            ansatz_qc = ansatz_op.to_QuantumCircuit(ref_state=ref_state, trotter_number=self.trotter_number)
            psi = CircuitStateFn(ansatz_qc).to_spmatrix()
            return (psi @ self.to_sparse_matrix @ psi.T)[0,0].real
        else:
            psi = ansatz_op.exponentiate() * QuantumState([ref_state])
            return (psi.dagger * self * psi).real

    @cached_property
    def QWC_decomposition(self):
        """
        """
        HamGraph = ObservableGraph(self.symp_matrix, self.coeff_vec)
        QWC_operators = HamGraph.clique_cover(clique_relation='QWC', colouring_strategy='largest_first')
        # check the QWC groups sum to the original observable operator
        reconstructed = reduce(lambda x,y:x+y, QWC_operators.values())
        assert(
            np.all((self-reconstructed).coeff_vec == 0)
        ), 'Summing QWC group operators does not yield the original Hamiltonian'
        
        return QWC_operators

    def _ansatz_expectation_sampled(self, 
            ansatz_op: AnsatzOp, 
            ref_state: np.array
        ) -> float:
        """ Evaluates epectation values by quantum circuit sampling. Decomposes the ObservableOp into
        qubitwise commuting (QWC) components that may be measured simultaneously by transforming onto 
        the Pauli Z basis. This allows one to reconstruct the ansatz operator in a tomography-esque
        fashion and determine the expectation value w.r.t. the observable operator.

        Returns:
            expectation (float): the expectation value of the ObseravleOp w.r.t. the given ansatz and reference state
        """
        #start = time.time()
        QWC_group_data = {}
        for group_index, group_operator in self.QWC_decomposition.items():
            # find the qubit positions containing X's or Y's in the QWC group
            X_indices = np.where(np.sum(group_operator.X_block, axis=0)!=0)[0]
            Y_indices = np.where(np.sum(group_operator.X_block & group_operator.Z_block, axis=0)!=0)[0]
            basis_change_indices={'X_indices':X_indices, 'Y_indices':Y_indices}
            # we make a change of basis from Pauli X/Y --> Z, allowing us to take measurements in the Z basis
            group_qc = ansatz_op.to_QuantumCircuit(
                ref_state = ref_state,
                trotter_number=self.trotter_number,
                basis_change_indices=basis_change_indices, 
                ZX_reduction=False
            )
            group_qc.measure_all()
            QWC_group_data[group_index] = {
                "operator":PauliwordOp(group_operator.symp_matrix, group_operator.coeff_vec),
                "circuit":group_qc,
                "measured_state":None
            }
        #stop = time.time()
        #print('QWC group data', stop - start)
        #start = time.time()
        # send all the QWC group circuits off to be executed via the backend
        QWC_group_circuits = [group['circuit'] for group in QWC_group_data.values()]
        job = execute(QWC_group_circuits, self.backend, shots=self.n_shots)
        #stop = time.time()
        #print('Obtain measurements', stop - start)
        #start = time.time()
        expectation = 0

        for i in range(len(QWC_group_circuits)):
            # reconstruct ansatz state from measurement outcomes (tomography-esque)
            states_hex, frequency = zip(*job.result().data(i)['counts'].items())
            state_matrix = [
                [int(i) for i in np.binary_repr(int(state, 16), self.n_qubits)]
                for state in states_hex
            ]
            state = QuantumState(state_matrix, np.array(frequency)/self.n_shots)
            QWC_group_data[i]['measured_state'] = state
            observable = QWC_group_data[i]['operator']
            # modify the Z block in line with the change of basis
            Z_symp_matrix = observable.X_block | observable.Z_block
            for b_state, b_state_coeff in zip(state.state_matrix, state.coeff_vector):
                expectation += np.sum((-1)**np.count_nonzero(Z_symp_matrix & b_state, axis=1)*observable.coeff_vec*b_state_coeff)
        #stop = time.time()
        #print('Evaluate expectation value', stop - start)

        return expectation.real

    def ansatz_expectation(self, 
            ansatz_op: AnsatzOp, 
            ref_state: np.array, 
        ) -> float:
        """ 
        """
        if self.evaluation_method == 'statevector':
            return self._ansatz_expectation_statevector(ansatz_op, ref_state)
        elif self.evaluation_method == 'trotter_rotations':
            return self._ansatz_expectation_trotter_rotations(ansatz_op, ref_state)
        elif self.evaluation_method == 'sampled':
            return np.mean([self._ansatz_expectation_sampled(ansatz_op, ref_state) for i in range(self.n_realizations)])
        else:
            raise ValueError('Invalid evaluation method, must be one of statevector, trotter_rotations or sampled')

    def parameter_shift_at_index(self,
        param_index: int,
        ansatz_op: AnsatzOp, 
        ref_state: np.array
        ):
        """ return the first-order derivative at x w.r.t. to the observable operator
        """
        assert(param_index<ansatz_op.n_terms), 'Indexing outside ansatz parameters'
        
        anz_upper_param = ansatz_op.copy()
        anz_lower_param = ansatz_op.copy()
        anz_upper_param.coeff_vec[param_index] += np.pi/4
        anz_lower_param.coeff_vec[param_index] -= np.pi/4 
        
        shift_upper = self.ansatz_expectation(anz_upper_param, ref_state=ref_state)
        shift_lower = self.ansatz_expectation(anz_lower_param, ref_state=ref_state)

        return shift_upper-shift_lower

    def gradient(self,
        ansatz_op: AnsatzOp,
        ref_state: np.array
        ) -> np.array:
        """ multiprocessing allows partial gradients to be computed simultaneously

        TODO does not work with sampled expectation values due to nested (daemonic) multiprocesses!
        Possbile workaround using ThreadPool, but is slower than standard Pool
        """
        pool = mp.Pool(mp.cpu_count())
        return pool.starmap(
            self.parameter_shift_at_index, 
            [(i, ansatz_op, ref_state) for i in range(ansatz_op.n_terms)]
        )

    def VQE(self,
        ansatz_op:  AnsatzOp,
        ref_state:  np.array,
        optimizer:  str   = 'SLSQP',
        maxiter:    int   = 10, 
        opt_tol:    float = None
        ):
        """ 
        Rcommended optimizers:
            - SLSQP  (gradient-descent, does not evaluate Jacobian at each iterate like BFGS or CG so is faster)
            - COBYLA (gradient-free)
        """
        interim_values = {'values':[], 'params':[], 'gradients':[], 'count':0}

        init_params = ansatz_op.coeff_vec

        def fun(x):
            interim_values['count']+=1
            ansatz_op.coeff_vec = x
            energy = self.ansatz_expectation(ansatz_op, ref_state)
            interim_values['params'].append((interim_values['count'], x))
            interim_values['values'].append((interim_values['count'], energy))
            return energy

        def jac(x):
            ansatz_op.coeff_vec = x
            grad = self.gradient(ansatz_op, ref_state)
            interim_values['gradients'].append((interim_values['count'], grad))
            return grad

        vqe_result = minimize(
            fun=fun, 
            jac=jac,
            x0=init_params,
            method=optimizer,
            tol=opt_tol,
            options={'maxiter':maxiter}
        )

        return vqe_result, interim_values