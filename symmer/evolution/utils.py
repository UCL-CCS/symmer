import numpy as np
from itertools import combinations
from networkx import Graph, draw_spring
from networkx.algorithms.isomorphism.isomorphvf2 import GraphMatcher
from symmer import PauliwordOp
from collections import Counter
from typing import Union
from qiskit import QuantumCircuit

def get_CNOT_connectivity_graph(evolution_obj:Union[PauliwordOp,QuantumCircuit], print_graph=False):
    """ 
    Get the graph whoss edges denote nonlocal interaction between two qubits.
    This is useful for device-aware ansatz construction to ensure the circuit connectiviy
    may be accomodated by the topology of the target quantum processor. 

    Args:
        evolution_obj (Union[PauliwordOp, QuantumCircuit]): Evolution Object
        print_graph (bool): If True, the graph is drawn. By default, it's set to False.
    """
    if isinstance(evolution_obj, QuantumCircuit):
        edges = [[q.index for q in step[1]] for step in evolution_obj.data if step[0].name!='barrier' and len(step[1])>1]
        weighted_edges = [(u,v,w) for (u,v),w in Counter(edges).items()]
    else:
        rows, cols = np.where(evolution_obj.X_block | evolution_obj.Z_block)
        support_indices = [evolution_obj.n_qubits - 1 - cols[rows==i] for i in np.unique(rows)]
        qubit_coupling = [list(zip(x[:-1], x[1:])) for x in support_indices]
        edges = [a for b in qubit_coupling for a in b]
        weighted_edges = [(u,v,w*2) for (u,v),w in Counter(edges).items()]
        
    G = Graph()
    G.add_weighted_edges_from(weighted_edges)
    if print_graph:
        draw_spring(G)
    return G

def _subgraph_isomorphism_distance(G, target, depth=0):

    if depth == 0:
        if GraphMatcher(target, G).subgraph_is_isomorphic():
            return 0
        else:
            return None
    else:
        ordered_nodes = sorted(
            combinations(G.nodes, r=depth), 
            key=lambda nodes:-np.sum([len(G.edges(n)) for n in nodes])
        )
        for nodes in ordered_nodes:
            G_temp = G.copy()
            for n in nodes:
                G_temp.remove_node(n)
            if GraphMatcher(target, G_temp).subgraph_is_isomorphic():
                dropped_edge_weights = [G.edges[e]['weight'] for n in nodes for e in G.edges(n)]
                return sum(dropped_edge_weights)

        return None

def subgraph_isomorphism_distance(G, target, max_depth=3):
    depth = 0
    for depth in range(max_depth):
        dist = _subgraph_isomorphism_distance(G, target, depth)
        if dist is not None:
            return depth * dist
        else:
            depth += 1
    return None

def topology_match_score(ansatz_operator, topology, max_depth=3):
    entangling_graph = get_CNOT_connectivity_graph(ansatz_operator)
    subgraph_cost = subgraph_isomorphism_distance(entangling_graph, topology, max_depth=max_depth)
    n_entangling_gates = np.count_nonzero(ansatz_operator.X_block | ansatz_operator.Z_block)
    return 1-subgraph_cost/n_entangling_gates