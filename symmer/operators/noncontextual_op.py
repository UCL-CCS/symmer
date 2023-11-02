import warnings
import os
import itertools
import numpy as np
import networkx as nx
import multiprocessing as mp
import qubovert as qv
from cached_property import cached_property
from time import time
from functools import reduce
from typing import Optional, Union, Tuple, List
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, shgo
from symmer.operators import PauliwordOp, IndependentOp, AntiCommutingOp, QuantumState
from symmer.operators.utils import binomial_coefficient, perform_noncontextual_sweep
from ray import put, get, remote
from symmer.utils import random_anitcomm_2n_1_PauliwordOp

class NoncontextualOp(PauliwordOp):
    """ 
    Class for representing noncontextual Hamiltonians

    Noncontextual Hamiltonians are precisely those whose terms may be reconstructed 
    under the Jordan product (AB = {A, B}/2) from a generating set of the form 
    G ∪ {C_1, ..., C_M} where {C_i, C_j}=0 for i != j and G commutes universally.
    Refer to https://arxiv.org/abs/1904.02260 for further details. 
    
    Attributes:
        up_method (str): 
    """
    up_method = 'seq_rot'

    def __init__(self,
            symp_matrix,
            coeff_vec
        ):
        """
        Args:
            symp_matrix (np.array): Symplectic matrix.
            coeff_vec (np.array): Coefficient Vector.
        """
        super().__init__(symp_matrix, coeff_vec)
        assert(self.is_noncontextual), 'Specified operator is contextual.'
        # extract the symmetry generating set G and clique operator C(r)
        self.noncontextual_generators()
        # Reconstruct the noncontextual Hamiltonian into its G and C(r) components
        self.noncontextual_reconstruction()
        
    @classmethod
    def from_PauliwordOp(cls, H) -> "NoncontextualOp":
        """ 
        For convenience, initialize from an existing PauliwordOp.       

        Args:
            H: A PauliwordOp object representing the operator.

        Returns:
            NoncontextualOp: A NoncontextualOp instance initialized from the given PauliwordOp.
        """
        noncontextual_operator = cls(
            H.symp_matrix,
            H.coeff_vec
        )
        return noncontextual_operator

    @classmethod
    def from_hamiltonian(cls, 
            H: PauliwordOp, 
            strategy: str = 'diag', 
            generators:  PauliwordOp = None,
            stabilizers: IndependentOp = None, 
            DFS_runtime: int = 10,
            use_jordan_product = False,
            override_noncontextuality_check: bool = True
        ) -> "NoncontextualOp":
        """ 
        Given a PauliwordOp, extract from it a noncontextual sub-Hamiltonian by the specified strategy.

        Args:
            H (PauliwordOp): PauliwordOp representing the operator.
            strategy (str, optional): The strategy for constructing the noncontextual operator.
            generators (PauliwordOp, optional): PauliwordOp representing the generators for the 'generators' strategy.
            stabilizers (IndependentOp, optional): IndependentOp representing the stabilizers for the 'stabilizers' strategy.
            DFS_runtime (int, optional): The maximum runtime in seconds for the DFS-based strategies. Default is 10.
            use_jordan_product (bool, optional): Specifies whether to use the Jordan product for the 'generators' strategy. Default is False.
            override_noncontextuality_check (bool, optional): Specifies whether to override the noncontextuality check for the input Hamiltonian. Default is True.

        Returns:
            NoncontextualOp: A NoncontextualOp instance constructed from the given operator using the specified strategy.
        """
        if not override_noncontextuality_check:
            if H.is_noncontextual:
                warnings.warn('input H is already noncontextual ignoring strategy')
                return cls.from_PauliwordOp(H)
        
        if strategy == 'diag':
            return cls._diag_noncontextual_op(H)
        elif strategy == 'generators':
            return cls._from_generators_noncontextual_op(H, generators, use_jordan_product=use_jordan_product)
        elif strategy == 'stabilizers':
            return cls._from_stabilizers_noncontextual_op(H, stabilizers, use_jordan_product=use_jordan_product)
        elif strategy.find('DFS') != -1:
            _, strategy = strategy.split('_')
            return cls._dfs_noncontextual_op(H, strategy=strategy, runtime=DFS_runtime)
        elif strategy.find('SingleSweep') != -1:
            _, strategy = strategy.split('_')
            return cls._single_sweep_noncontextual_operator(H, strategy=strategy)
        else:
            raise ValueError(f'Unrecognised noncontextual operator strategy {strategy}')

    @classmethod
    def _diag_noncontextual_op(cls, H: PauliwordOp) -> "NoncontextualOp":
        """ 
        Return the diagonal terms of the PauliwordOp - this is the simplest noncontextual operator.

        Args:
            H (PauliwordOp): PauliwordOp representing the operator.

        Returns:
            NoncontextualOp: A NoncontextualOp instance constructed from the diagonal terms of the given operator.
        """
        mask_diag = np.where(~np.any(H.X_block, axis=1))
        noncontextual_operator = cls(
            H.symp_matrix[mask_diag],
            H.coeff_vec[mask_diag]
        )
        return noncontextual_operator

    @classmethod
    def _dfs_noncontextual_op(cls, H: PauliwordOp, runtime=10, strategy='magnitude') -> "NoncontextualOp":
        """ 
        function orders operator by coeff mag
        then going from first term adds ops to a pauliword op ensuring it is noncontextual
        adds to a tracking list and then changes the original ordering so first term is now at the end
        repeats from the start (aka generating a list of possible noncon Hamiltonians)
        from this list one can then choose the noncon op with the most terms OR largest sum of abs coeff weights
        cutoff time ensures if the number of possibilities is large the function will STOP and not take too long

        Args:
            H (PauliwordOp): PauliwordOp representing the operator.
            runtime (float, optional): The maximum runtime of the method in seconds. Default is 10.
            strategy (str, optional): The strategy for selecting the noncontextual operator.
                - 'magnitude': Chooses the operator with the largest sum of absolute coefficient weights.
                - 'largest': Chooses the operator with the most terms.
                Default is 'magnitude'.

        Returns:
            NoncontextualOp: A NoncontextualOp instance constructed from the given operator.
        """
        operator = H.sort(by='magnitude')
        noncontextual_ops = []

        n=0
        start_time = time()
        while n < H.n_terms and time()-start_time < runtime:
            order = np.roll(np.arange(H.n_terms), -n)
            ordered_operator = PauliwordOp(
                symp_matrix=operator.symp_matrix[order],
                coeff_vec=operator.coeff_vec[order]
            )
            noncontextual_operator = perform_noncontextual_sweep(ordered_operator)
            noncontextual_ops.append(noncontextual_operator)
            n+=1

        if strategy == 'magnitude':
            noncontextual_operator = sorted(noncontextual_ops, key=lambda x:-np.sum(abs(x.coeff_vec)))[0]
        elif strategy == 'largest':
            noncontextual_operator = sorted(noncontextual_ops, key=lambda x:-x.n_terms)[0]
        else:
            raise ValueError('Unrecognised noncontextual operator strategy.')

        return cls.from_PauliwordOp(noncontextual_operator)

    @classmethod
    def _diag_first_noncontextual_op(cls, H: PauliwordOp) -> "NoncontextualOp":
        """ 
        Start from the diagonal noncontextual form and append additional off-diagonal
        contributions with respect to their coefficient magnitude.
        
        Args:
            H (PauliwordOp): PauliwordOp representing the operator.

        Returns:
            NoncontextualOp: A NoncontextualOp instance constructed from the given operator.
        """
        noncontextual_operator = cls._diag_noncontextual_op(H)
        # order the remaining terms by coefficient magnitude
        off_diag_terms = (H - noncontextual_operator).sort(by='magnitude')
        # append terms that do not make the noncontextual_operator contextual!
        for term in off_diag_terms:
            if (noncontextual_operator+term).is_noncontextual:
                noncontextual_operator+=term
        
        return cls.from_PauliwordOp(noncontextual_operator)

    @classmethod
    def _single_sweep_noncontextual_operator(cls, H, strategy='magnitude') -> "NoncontextualOp":
        """ 
        Order the operator by some sorting key (magnitude, random or CurrentOrder)
        and then sweep accross the terms, appending to a growing noncontextual operator
        whenever possible.

        Args:
            H (PauliwordOp): PauliwordOp representing the operator.
            strategy (str, optional): Sorting strategy for ordering the operator terms.
                - 'magnitude': Orders the terms by magnitude.
                - 'random': Randomly shuffles the terms.
                - 'CurrentOrder': Uses the current order of the terms.
                Default is 'magnitude'.

        Returns:
            NoncontextualOp: A NoncontextualOp instance constructed from the given operator using the specified strategy.
        """
        if strategy=='magnitude':
            operator = H.sort(by='magnitude')
        elif strategy=='random':
            order = np.arange(H.n_terms)
            np.random.shuffle(order)
            operator = PauliwordOp(
                H.symp_matrix[order],
                H.coeff_vec[order]
            )
        elif strategy =='CurrentOrder':
            operator = H
        else:
            raise ValueError('Unrecognised strategy, must be one of magnitude, random or CurrentOrder')            

        nc_operator = perform_noncontextual_sweep(operator)
        return cls.from_PauliwordOp(nc_operator)

    @classmethod
    def _from_generators_noncontextual_op(cls, 
            H: PauliwordOp, generators: PauliwordOp, use_jordan_product:bool=False
        ) -> "NoncontextualOp":
        """ 
        Construct a noncontextual operator given a noncontextual generating set, via the Jordan product ( regular matrix product if the operators commute, and equal to zero if the operators anticommute.)

        Args:
            H (PauliwordOp): PauliwordOp representing the Hamiltonian.
            generators (PauliwordOp): PauliwordOp representing the noncontextual generating set.
            use_jordan_product (bool, optional): Determines whether to use the Jordan product for construction. 
                If True, the Jordan product is used. If False, an alternative strategy is used. Default is False.

        Returns:
            NoncontextualOp: A NoncontextualOp instance constructed from the given Hamiltonian and generators.
        """
        assert generators is not None, 'Must specify a noncontextual generating set.'
        assert generators.is_noncontextual, 'Generating set is contextual.'
        if use_jordan_product:
            _, noncontextual_terms_mask = H.jordan_generator_reconstruction(generators)
        else:
            _, noncontextual_terms_mask = H.generator_reconstruction(generators, override_independence_check=True)

        return cls.from_PauliwordOp(H[noncontextual_terms_mask])

    @classmethod
    def random(cls,
            n_qubits: int,
            n_cliques:Optional[int]=3,
            complex_coeffs:Optional[bool]=False,
            n_commuting_terms:Optional[int]=None,
        ) -> "NoncontextualOp":
        """
        Generate a random Noncontextual operator with normally distributed coefficients.
        Note to maximise size choose number of n_cliques to be 3 (and for 2<= n_cliques <= 5 the operator
        will be larger than only using commuting generators).
        WARNING: this function can generates an exponentially large Hamiltonian unless n_terms set.
        size when NOT set is: n_cliques * [ 2**(n_qubits -  int(np.ceil((n_cliques - 1) / 2))) ]

        Note: The number of terms in output will be: n_cliques*n_commuting_terms

        Args:
            n_qubits (int): Number of qubits noncontextual operator defined on
            n_cliques (int): Number of cliques representives in operator
            complex_coeffs (bool): Whether to generate complex coefficients (default: True).
            n_commuting_terms (int): Optional int for number of commuting terms. if not set then it will be: 2**(n_qubits -  int(np.ceil((n_cliques - 1) / 2))) (i.e. exponentially large)

        Returns:
            NoncontextualOp: A random NoncontextualOp object.
        """
        assert n_cliques > 1, 'number of cliques must be set to 2 or more (cannot have one anticommuting term)'
        n_clique_qubits = int(np.ceil((n_cliques - 1) / 2))
        assert n_clique_qubits <= n_qubits, 'cannot have {n_cliques} anticommuting cliques on {n_qubits} qubits'

        remaining_qubits = n_qubits - n_clique_qubits

        if n_commuting_terms:
            assert n_commuting_terms<= 2**remaining_qubits, f'cannot have {n_commuting_terms} commuting operators on {remaining_qubits} qubits'

        if remaining_qubits>=1:
            if n_commuting_terms==None:
                n_commuting_terms = 2 ** (remaining_qubits)
                XZ_block = (((np.arange(n_commuting_terms)[:, None] & (1 << np.arange(2 * remaining_qubits))[
                                                                            ::-1])) > 0).astype(bool)
            else:
                # randomly chooise Z bitstrings in symp matrix:
                indices = np.unique(np.random.random_integers(0,
                                                              high=2**remaining_qubits-1,
                                                              size=10*n_commuting_terms))
                while len(indices) < n_commuting_terms:
                    indices = np.unique(np.append(indices,
                                                  np.unique(np.random.random_integers(0,
                                                                                      high=2 ** remaining_qubits - 1,
                                                                                      size=10*n_commuting_terms)))
                                        )

                indices = indices[:n_commuting_terms]
                XZ_block = (((indices[:, None] & (1 << np.arange(2 * remaining_qubits))[
                                                                        ::-1])) > 0).astype(bool)

        if n_cliques == 0:
            H_nc = PauliwordOp(XZ_block, np.ones(XZ_block.shape[0]))

        else:
            AC = random_anitcomm_2n_1_PauliwordOp(n_clique_qubits,
                                                  apply_clifford=True)[:n_cliques]
            AC.coeff_vec = np.ones_like(AC.coeff_vec)
            if remaining_qubits >= 1:
                diag_H = PauliwordOp(XZ_block, np.ones(XZ_block.shape[0]))
            else:
                diag_H = PauliwordOp.from_list(['I' * remaining_qubits])

            AC_full = PauliwordOp.from_list(['I' * remaining_qubits]).tensor(AC)
            H_sym = diag_H.tensor(PauliwordOp.from_list(['I' * n_clique_qubits]))
            H_nc = AC_full * H_sym
            assert AC.n_terms * n_commuting_terms == H_nc.n_terms, 'operator not largest it can be'

        coeff_vec = np.random.randn(H_nc.n_terms).astype(complex)
        if complex_coeffs:
            coeff_vec += 1j * np.random.randn(H_nc.n_terms)

        # apply clifford rotations to get rid of some of generation structure
        U_cliff_rotations = []
        for _ in range(n_qubits * 5):
            P_rand = PauliwordOp.random(H_nc.n_qubits, n_terms=1)
            P_rand.coeff_vec = [1]
            U_cliff_rotations.append((P_rand, None))

        H_nc = H_nc.perform_rotations(U_cliff_rotations)

        return cls(H_nc.symp_matrix, coeff_vec)

    @classmethod
    def _from_stabilizers_noncontextual_op(cls, 
            H:PauliwordOp, stabilizers: IndependentOp, use_jordan_product=False
        ) -> "NoncontextualOp":
        """
        Args:
            H (PauliwordOp): The PauliwordOp representing the Hamiltonian.
            stabilizers (IndependentOp): The IndependentOp representing the stabilizers.
            use_jordan_product (bool, optional): Determines whether to use the Jordan product for constructing generators. 
                If True, the Jordan product is used. If False, an alternative strategy is used. Default is False.

        Returns:
            NoncontextualOp: A NoncontextualOp instance constructed from the given PauliwordOp and stabilizers.
        """
        symmetries = IndependentOp.symmetry_generators(stabilizers, commuting_override=True)
        noncon = NoncontextualOp.from_hamiltonian(symmetries, strategy='DFS_magnitude')
        generators = noncon.symmetry_generators
        if noncon.clique_operator.n_terms>0:
            generators+=noncon.clique_operator
            use_jordan_product=True

        return cls._from_generators_noncontextual_op(H=H, generators=generators, use_jordan_product=use_jordan_product)
        
    def draw_graph_structure(self, 
            clique_lw=1,
            symmetry_lw=.25,
            node_colour='black',
            node_size=20,
            seed=None,
            axis=None,
            include_symmetries=True
        ):
        """ 
        Draw the noncontextual graph structure.

        Args:
            clique_lw (int, optional): Line width for non-symmetry edges. Default is 1. 
            symmetry_lw (float, optional): Line width for symmetry edges. Default is 0.25. 
            node_colour (str, optional): Color of the nodes. Default is 'black'. 
            node_size (int, optional): Size of the nodes. Default is 20.
            seed (int or None, optional): Random seed for layout. Default is None.
            axis (matplotlib.axes.Axes or None, optional): Matplotlib axis to draw the graph on. Default is None.
            include_symmetries (bool, optional): Determines whether to include symmetry edges in the visualization. Default is True.
        """
        adjmat = self.adjacency_matrix.copy()
        index_symmetries = np.where(np.all(adjmat, axis=1))[0]
        np.fill_diagonal(adjmat, False)
        
        G = nx.Graph()
        for i,j in list(zip(*np.where(adjmat))):
            if i in index_symmetries or j in index_symmetries:
                if include_symmetries:
                    G.add_edge(i,j,color='grey',weight=symmetry_lw)
            else:
                G.add_edge(i,j,color='black',weight=clique_lw)

        pos = nx.spring_layout(G, seed=seed)
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]
        nx.draw(G, pos, edge_color=colors, width=weights, 
                node_color=node_colour, node_size=node_size, ax=axis)

    def noncontextual_generators(self) -> None:
        """ 
        Find an independent generating set for the noncontextual operator.
        """
        Z2_symmerties = IndependentOp.symmetry_generators(self, commuting_override=True)

        if not np.all(Z2_symmerties.commutes_termwise(Z2_symmerties)):
            # need to account for Z2_symmerties not commuting with themselves
            sym_gens = self.generators
            # z2_mask = np.sum(sym_gens.adjacency_matrix, axis=1) == sym_gens.n_terms
            z2_mask = np.sum(sym_gens.commutes_termwise(sym_gens), axis=1) == sym_gens.n_terms

            Z2_incomplete = sym_gens[z2_mask]
            _, missing_mask = sym_gens.generator_reconstruction(Z2_incomplete)
            Z2_missing = sym_gens[~missing_mask]

            cover = Z2_missing.clique_cover('C')
            clique_rep_list = [C.sort()[0] for C in cover.values()]

            sym_from_cliques = sum((cover[n] - C_rep) * C_rep for n, C_rep in enumerate(clique_rep_list) if
                                   cover[n].n_terms > 1)

            Z2_symmerties = (sym_from_cliques + Z2_incomplete).generators
            _, z2_mask = self.generator_reconstruction(Z2_symmerties)
        else:
            _, z2_mask = self.generator_reconstruction(Z2_symmerties)

        remaining = self[~z2_mask]

        if remaining.n_terms>0:
            ## rather than doing graph coloring (line below)
            #self.decomposed = remaining.clique_cover('C')

            ## use noncon structure of disjoint cliques
            # remaining must be disjoint union of commuting cliques...
            # So find unique rows of adj matrix and check there is NO overlap between them (disjoint!)
            adj_matrix_view = np.ascontiguousarray(remaining.adjacency_matrix).view(
                np.dtype((np.void, remaining.adjacency_matrix.dtype.itemsize * remaining.adjacency_matrix.shape[1]))
            )
            re_order_indices = np.argsort(adj_matrix_view.ravel())
            # sort the adj matrix and vector of coefficients accordingly
            sorted_terms = remaining.adjacency_matrix[re_order_indices]
            # unique terms are those with non-zero entries in the adjacent row difference array
            diff_adjacent = np.diff(sorted_terms, axis=0)
            mask_unique_terms = np.append(True, np.any(diff_adjacent, axis=1))
            clique_mask = sorted_terms[mask_unique_terms]
            self.decomposed = {ind: remaining[c_mask] for ind, c_mask in enumerate(clique_mask)}

            self.n_cliques = len(self.decomposed)
            if self.n_cliques > 0:
                # choose clique representatives with the greatest coefficient
                # see equation 3 of https://arxiv.org/pdf/2002.05693.pdf
                clique_rep_list = [C.sort()[0] for C in self.decomposed.values()]
                self.clique_operator = AntiCommutingOp.from_PauliwordOp(
                    sum(clique_rep_list)
                )
                self.clique_operator.coeff_vec = np.ones_like(self.clique_operator.coeff_vec)

                ## cliques can form new Z2 syms
                sym_from_cliques = sum((self.decomposed[n] - C_rep) * C_rep for n, C_rep in enumerate(clique_rep_list) if
                                       self.decomposed[n].n_terms > 1)
                if sym_from_cliques:
                    Z2_symmerties = (sym_from_cliques + Z2_symmerties).generators
        else:
            self.clique_operator = PauliwordOp.empty(self.n_qubits).cleanup()
            self.decomposed = dict()
            self.n_cliques=0

        self.symmetry_generators = IndependentOp.from_PauliwordOp(Z2_symmerties)
        _, Z2_mask = self.generator_reconstruction(Z2_symmerties)
        self.decomposed['symmetry'] = self[Z2_mask]

    def noncontextual_reconstruction(self) -> None:
        """ 
        Reconstruct the noncontextual operator in each independent basis GuCi - one for every clique.
        This mitigates against dependency between the symmetry generators G and the clique representatives Ci.
        """
        noncon_generators = PauliwordOp(
            np.vstack([self.symmetry_generators.symp_matrix, self.clique_operator.symp_matrix]),
            np.ones(self.symmetry_generators.n_terms + self.n_cliques)
        )

        # Cannot simultaneously know eigenvalues of cliques so we peform a generator reconstruction
        # that respects the jordan product A*B = {A, B}/2, i.e. anticommuting elements are zeroed out
        jordan_recon_matrix, successful = self.jordan_generator_reconstruction(noncon_generators)#, override_independence_check=True)
        assert(np.all(successful)), 'The generating set is not sufficient to reconstruct the noncontextual Hamiltonian'
        self.G_indices = jordan_recon_matrix[:, :self.symmetry_generators.n_terms]
        self.C_indices = jordan_recon_matrix[:, self.symmetry_generators.n_terms:]
        self.mask_S0 = ~np.any(self.C_indices, axis=1)
        self.mask_Ci = self.C_indices.astype(bool).T
        # individual elements of r_part commute with all of G_part - taking products over G_part with
        # a single element of r_part will therefore never produce a complex phase, but might result in
        # a sign flip that must be accounted for in the generator reconstruction:
        multiply_indices = lambda inds:reduce(
            lambda x,y:x*y, # pairwise multiplication of Pauli factors
            noncon_generators[inds], # index the relevant noncontextual generating elements
            PauliwordOp.from_list(['I'*self.n_qubits]) # initialise product with identity
        ).coeff_vec[0].real

        self.pauli_mult_signs = np.array(
            list(map(multiply_indices,jordan_recon_matrix.astype(bool)))
        ).astype(int)
        
    def symmetrized_operator(self, expansion_order=1):
        """ 
        Get the symmetrized noncontextual operator S_0 - sqrt(S_1^2 + .. S_M^2).
        In the infinite limit of expansion_order the ground state of this operator
        will coincide exactly with the true noncontextual operator. This is used
        for xUSO solver since this reformulation of the Hamiltonian is polynomial.

        Args:
            expansion_order (int): Expansion order. By default, it is set to 1.

        Returns:
            Symmetrized noncontextual operator.
        """
        Si_list = [self.decomposed['symmetry']]
        for i in range(self.n_cliques):
            Ci = self.decomposed[i][0]; Ci.coeff_vec[0]=1
            Si = Ci*self.decomposed[i]
            Si_list.append(Si)

        S = sum([Si**2 for Si in Si_list[1:]])
        norm = np.linalg.norm(S.coeff_vec, ord=1)
        S *= (1/norm)
        I = PauliwordOp.from_list(['I'*self.n_qubits])
        terms = [
            (I-S)**n * (-1)**n * binomial_coefficient(.5, n) 
            for n in range(expansion_order+1)
        ] # power series expansion of the oeprator root
        S_root = sum(terms) * np.sqrt(norm)
        
        return Si_list[0] - S_root

    def get_symmetry_contributions(self, nu: np.array) -> float:
        """
        """
        nu = np.asarray(nu)
        coeff_mod =  (
            # coefficient vector whose signs we are modifying:
            self.coeff_vec *
            # sign flips from generator reconstruction:
            self.pauli_mult_signs *
            # sign flips from nu assignment:
            (-1)**np.count_nonzero(np.logical_and(self.G_indices==1, nu == -1), axis=1)
        )
        s0 = np.sum(coeff_mod[self.mask_S0]).real
        si = np.array([np.sum(coeff_mod[mask]).real for mask in self.mask_Ci])
        return s0, si

    def get_energy(self, nu: np.array) -> float:
        """
        The classical objective function that encodes the noncontextual energies.
        """
        s0, si = self.get_symmetry_contributions(nu)
        return s0 - np.linalg.norm(si)
    
    def update_clique_representative_operator(self, clique_index:int = None) -> List[Tuple[PauliwordOp, float]]:
        _, si = self.get_symmetry_contributions(self.symmetry_generators.coeff_vec)
        self.clique_operator.coeff_vec = si
        if clique_index is None:
            clique_index = 0
        (
            self.mapped_clique_rep, 
            self.unitary_partitioning_rotations, 
            self.clique_normalization,
            self.clique_operator
        ) = self.clique_operator.unitary_partitioning(up_method=self.up_method, s_index=clique_index)
        
    def solve(self, 
            strategy: str = 'brute_force', 
            ref_state: np.array = None, 
            num_anneals:int = 1_000,
            expansion_order:int = 1
        ) -> None:
        """ 
        Minimize the classical objective function, yielding the noncontextual 
        ground state. This updates the coefficients of the clique representative 
        operator C(r) and symmetry generators G with the optimal configuration.

        Note: Most QUSO functions/methods work faster than their PUSO counterparts.

        Args:
            strategy (str): Optimization strategy. By default it is set to 'brute_force'. It can be 'brute_force', 'binary_relaxation', 'brute_force_PUSO', 'brute_force_QUSO', 'annealing_PUSO', or 'annealing_QUSO'.
            ref_state (np.array): Reference State.
            num_anneals (int): Number of simulated anneals to do.
            expansion_order (int): Expansion order. By default, it is set to 1.
        """
        if ref_state is not None:
            # update the symmetry generator G coefficients w.r.t. the reference state
            self.symmetry_generators.update_sector(ref_state)
            ev_assignment = self.symmetry_generators.coeff_vec
            fixed_ev_mask = ev_assignment!=0
            fixed_eigvals = (ev_assignment[fixed_ev_mask]).astype(int)
            NC_solver = NoncontextualSolver(self, fixed_ev_mask, fixed_eigvals)
            # any remaining unfixed symmetry generators are solved via other means:
        else:
            NC_solver = NoncontextualSolver(self)

        NC_solver.num_anneals = num_anneals
        NC_solver.expansion_order = expansion_order

        if strategy=='brute_force':
            self.energy, nu = NC_solver.energy_via_brute_force()

        elif strategy=='binary_relaxation':
            self.energy, nu = NC_solver.energy_via_relaxation()
        
        else:
            #### qubovert strategies below this point ####
            # PUSO = Polynomial unconstrained spin Optimization
            # QUSO: Quadratic Unconstrained Spin Optimization
            if strategy == 'brute_force_PUSO':
                NC_solver.method = 'brute_force'
                NC_solver.x = 'P'   
            elif strategy == 'brute_force_QUSO':  
                NC_solver.method = 'brute_force'
                NC_solver.x = 'Q'
            elif strategy == 'annealing_PUSO':
                NC_solver.method = 'annealing'
                NC_solver.x = 'P'
            elif strategy == 'annealing_QUSO':
                NC_solver.method = 'annealing'
                NC_solver.x = 'Q'
            else:
                raise ValueError(f'Unknown optimization strategy: {strategy}')
        
            self.energy, nu = NC_solver.energy_xUSO()

        # optimize the clique operator coefficients
        self.symmetry_generators.coeff_vec = nu.astype(int)
        if self.n_cliques > 0:
            self.update_clique_representative_operator()

    def noncon_state(self, UP_method:Optional[str]= 'LCU') -> Tuple[QuantumState, np.array]:
        """
        Method to generate noncontextual state for current symmetry generators assignments. Note by default
        UP_method is set to LCU as this avoids generating exponentially large states (which seq_rot can do!)

        Args:
            UP_method: string of unitary partitioning approach.

        Returns:
            state (QuantumState): noncontextual ground state
            nu_assignment (np.array): vector (nu) of expectation value assignments for noncontexutal symmetry generators

        """
        nu_assignment = self.symmetry_generators.coeff_vec.copy()

        ## update clique coeffs from nu assignment!
        _, si = self.get_symmetry_contributions(nu_assignment)
        self.clique_operator.coeff_vec = si

        assert UP_method in ['LCU', 'seq_rot']

        if UP_method == 'LCU':
            Ps, rotations_LCU, gamma_l, AC_normed = self.clique_operator.unitary_partitioning(s_index=0,
                                                                                                  up_method='LCU')
        else:
            Ps, rotations_SEQ, gamma_l, AC_normed = self.clique_operator.unitary_partitioning(s_index=0,
                                                                                   up_method='seq_rot')

        # choose negative value for clique operator (to minimize energy)
        Ps.coeff_vec[0] = -1

        ### to find ground state, need to map noncontextual stabilizers to single qubit Pauli Zs
        independent_stabilizers = self.symmetry_generators + Ps

        # rotate onto computational basis
        independent_stabilizers.target_sqp = 'Z'

        rotated_stabs = independent_stabilizers.rotate_onto_single_qubit_paulis()
        clifford_rots = independent_stabilizers.stabilizer_rotations

        ## get stabilizer state for the rotated stabilizers
        Z_indices = np.sum(rotated_stabs.Z_block, axis=0)
        Z_vals = np.sum(rotated_stabs.Z_block[:, Z_indices.astype(bool)] * rotated_stabs.coeff_vec, axis=1)
        Z_indices[Z_indices.astype(bool)] = ((Z_vals - 1) * -0.5).astype(int)

        state = QuantumState(Z_indices.reshape(1, -1))

        ## undo clifford rotations
        from symmer.evolution.exponentiation import exponentiate_single_Pop
        for op, _ in clifford_rots:
            rot = exponentiate_single_Pop(op.multiply_by_constant(1j * np.pi / 4))
            state = rot.dagger * state

        ## undo unitary partitioning step
        if UP_method == 'LCU':
            state = rotations_LCU.dagger * state
        else:
            for op, angle in rotations_SEQ[::-1]:
                state = exponentiate_single_Pop(op.multiply_by_constant(1j * angle / 2)).dagger * state

        # TODO: could return clifford and UP rotations here too!
        return state, nu_assignment    
###############################################################################
################### NONCONTEXTUAL SOLVERS BELOW ###############################
###############################################################################

class NoncontextualSolver:

    # xUSO settings
    method:str = 'brute_force'
    x:str = 'P'
    num_anneals:int = 1_000,
    _nu = None,
    expansion_order=1

    def __init__(
        self,
        NC_op: NoncontextualOp,
        fixed_ev_mask: np.array = None,
        fixed_eigvals: np.array = None
        ) -> None:
        self.NC_op = NC_op
        
        if fixed_ev_mask is not None:
            assert fixed_eigvals is not None, 'Must specify the fixed eigenvalues'
            assert np.sum(fixed_ev_mask) == len(fixed_eigvals), 'Number of non-zero elements in mask does not match the number of fixed eigenvalues'
            self.fixed_ev_mask = fixed_ev_mask
            self.fixed_eigvals = fixed_eigvals
        else:
            self.fixed_ev_mask = np.zeros(NC_op.symmetry_generators.n_terms, dtype=bool)
            self.fixed_eigvals = np.array([], dtype=int)
    
    #################################################################
    ########################## BRUTE FORCE ##########################
    #################################################################

    def energy_via_brute_force(self) -> Tuple[float, np.array, np.array]:
        """ 
        Does what is says on the tin! Try every single eigenvalue assignment in parallel
        and return the minimizing noncontextual configuration. This scales exponentially in 
        the number of unassigned symmetry elements.
        """
        if np.all(self.fixed_ev_mask):
            nu_list = self.fixed_eigvals.reshape([1,-1])
        else:
            search_size = 2**np.sum(~self.fixed_ev_mask)
            nu_list = np.ones([search_size, self.NC_op.symmetry_generators.n_terms], dtype=int)
            nu_list[:,self.fixed_ev_mask] = np.tile(self.fixed_eigvals, [search_size,1])
            nu_list[:,~self.fixed_ev_mask] = np.array(list(itertools.product([-1,1],repeat=np.sum(~self.fixed_ev_mask))))
        
        # # optimize over all discrete value assignments of nu in parallel
        noncon_H = put(self.NC_op)
        tracker = np.array(get(
            [get_noncon_energy.remote(noncon_H, nu_vec) for nu_vec in nu_list]))

        # with mp.Pool(mp.cpu_count()) as pool:
        #     tracker = pool.map(self.NC_op.get_energy, nu_list)
        
        # find the lowest energy eigenvalue assignment from the full list
        full_search_results = zip(tracker, nu_list)
        energy, fixed_nu = min(full_search_results, key=lambda x:x[0])

        return energy, fixed_nu

    #################################################################
    ###################### BINARY RELAXATION ########################
    #################################################################

    def energy_via_relaxation(self) -> Tuple[float, np.array, np.array]:
        """ 
        Relax the binary value assignment of symmetry generators to continuous variables.
        """
        # optimize discrete value assignments nu by relaxation to continuous variables
        nu_bounds = [(0, np.pi)]*(self.NC_op.symmetry_generators.n_terms-np.sum(self.fixed_ev_mask))

        def get_nu(angles):
            """ 
            Build nu vector given fixed values.
            """
            nu = np.ones(self.NC_op.symmetry_generators.n_terms)
            nu[self.fixed_ev_mask] = self.fixed_eigvals
            nu[~self.fixed_ev_mask] = np.cos(angles)
            return nu

        optimizer_output = shgo(func=lambda angles:self.NC_op.get_energy(get_nu(angles)), bounds=nu_bounds)
        # if optimization was successful the optimal angles should consist of 0 and pi
        fix_nu = np.sign(np.array(get_nu(np.cos(optimizer_output['x'])))).astype(int)
        self.NC_op.symmetry_generators.coeff_vec = fix_nu 
        return optimizer_output['fun'], fix_nu
    
    #################################################################
    ################ UNCONSTRAINED SPIN OPTIMIZATION ################
    #################################################################    

    def get_cost_func(self):
        """ 
        Define the unconstrained spin cost function.
        """
        symmetrized_operator = self.NC_op.symmetrized_operator(expansion_order=self.expansion_order)
        G_indices, _ = symmetrized_operator.generator_reconstruction(self.NC_op.symmetry_generators)
        # setup spin variables
        fixed_indices = np.where(self.fixed_ev_mask)[0] # bool to indices
        fixed_assignments = dict(zip(fixed_indices, self.fixed_eigvals))
        q_vec_SPIN={}
        for ind in range(self.NC_op.symmetry_generators.n_terms):
            if ind in fixed_assignments.keys():
                q_vec_SPIN[ind] = fixed_assignments[ind]
            else:
                q_vec_SPIN[ind] = qv.spin_var('x%d' % ind)

        COST = 0
        for P_index, term in enumerate(G_indices):
            non_zero_inds = term.nonzero()[0]
            # collect all the spin terms
            G_term = 1
            for i in non_zero_inds:
                G_term *= q_vec_SPIN[i]

            # cost function
            COST += (
                G_term * 
                symmetrized_operator.coeff_vec[P_index].real
                #self.NC_op.pauli_mult_signs[P_index]# * 
                #r_part[P_index].real
            )

        return COST

    def energy_xUSO(self) -> Tuple[float, np.array, np.array]:
        """
        Get energy via either: Polynomial unconstrained spin Optimization (x=P)
                                    or
                                Quadratic Unconstrained Spin Optimization  (x=Q)

        via a brute force search over q_vector or via simulated annealing

        Note in this method the r_vector is fixed upon input! (aka just does binary optimization)

        Args:
            NC_op (NoncontextualOp): Non-contextual operator
            fixed_ev_mask (np.array): bool list of where eigenvalues in nu vector are fixed
            fixed_eigvals (np.array): list of nu eigenvalues that are fixed
            method (str): brute force or annealing optimization
            x (str): Whether method is Polynomial or Quadratic optimization
            num_anneals (optional): number of simulated anneals to do

        Returns:
            energy (float): noncontextual energy
        """
        assert self.x in ['P', 'Q']
        assert self.method in ['brute_force', 'annealing']
        
        COST = self.get_cost_func()
        
        if np.all(self.fixed_ev_mask):
            # if no degrees of freedom over nu vector, COST is a number
            nu_vec = self.fixed_eigvals
        else:
            if self.x =='P':
                spin_problem = COST.to_puso()
            else:
                spin_problem = COST.to_quso()

            if self.method=='brute_force':
                sol = spin_problem.solve_bruteforce()
            elif self.method == 'annealing':
                if self.x == 'P':
                    puso_res = qv.sim.anneal_puso(spin_problem, num_anneals=self.num_anneals)
                elif self.x == 'Q':
                    puso_res= qv.sim.anneal_quso(spin_problem, num_anneals=self.num_anneals)
                    assert COST.is_solution_valid(puso_res.best.state) is True
                sol = puso_res.best.state

            solution = COST.convert_solution(sol)
            nu_vec = np.ones(self.NC_op.symmetry_generators.n_terms, dtype=int)
            nu_vec[self.fixed_ev_mask] = self.fixed_eigvals
            # must ensure the binary variables are correctly ordered in the solution:
            nu_vec[~self.fixed_ev_mask] = np.array([solution[x_i] for x_i in sorted(COST.variables)])
        
        return self.NC_op.get_energy(nu_vec), nu_vec


@remote(num_cpus=os.cpu_count(),
            runtime_env={
                "env_vars": {
                    "NUMBA_NUM_THREADS": os.getenv("NUMBA_NUM_THREADS"),
                    # "OMP_NUM_THREADS": str(os.cpu_count()),
                    "OMP_NUM_THREADS": os.getenv("NUMBA_NUM_THREADS"),
                    "NUMEXPR_MAX_THREADS": str(os.cpu_count())
                }
            }
            )
def get_noncon_energy(noncon_H:NoncontextualOp, nu: np.array) -> float:
    """
    The classical objective function that encodes the noncontextual energies.
    """
    s0, si = noncon_H.get_symmetry_contributions(nu)
    return s0 - np.linalg.norm(si)