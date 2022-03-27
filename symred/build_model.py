import numpy as np
from copy import deepcopy
from typing import Dict, List
import openfermion as of
from openfermion import get_fermion_operator, jordan_wigner, FermionOperator
from openfermion.circuits import ( uccsd_singlet_get_packed_amplitudes,
                                   uccsd_singlet_generator, uccsd_generator,
                                   uccsd_convert_amplitude_format)
from openfermionpyscf import PyscfMolecularData
from itertools import combinations
from symred.S3_projection import QubitTapering, CS_VQE
from symred.symplectic_form import PauliwordOp, StabilizerOp
from symred.utils import greedy_dfs, exact_gs_energy

# comment out due to incompatible versions of Cirq and OpenFermion in Orquestra
def QubitOperator_to_dict(op, num_qubits):
    assert(type(op) == of.QubitOperator)
    op_dict = {}
    term_dict = op.terms
    terms = list(term_dict.keys())

    for t in terms:    
        letters = ['I' for i in range(num_qubits)]
        for i in t:
            letters[i[0]] = i[1]
        p_string = ''.join(letters)        
        op_dict[p_string] = term_dict[t]
         
    return op_dict

class build_molecule_for_projection(CS_VQE):
    """ Class for assessing various generator removal ordering heuristics
    """
    def __init__(self,
            calculated_molecule: PyscfMolecularData,
            basis_weighting = 'ham_coeff'
        )-> None:
        """
        """
        dashes = "------------------------------------------------"
        # Orbital nums and HF state
        self.n_electrons  = calculated_molecule.n_electrons
        self.n_qubits     = 2*calculated_molecule.n_orbitals
        self.hf_state = np.concatenate([
            np.ones(self.n_electrons, dtype=int),
            np.zeros(self.n_qubits-self.n_electrons, dtype=int)
            ]
        )
        self.HL_index = list(self.hf_state).index(0) #index HOMO-LUMO gap
        hf_string   = ''.join([str(i) for i in list(self.hf_state)])
        print(dashes)
        print('Information concerning the full system:')
        print(dashes)
        print(f'Number of qubits in full problem: {self.n_qubits}')
        print(f'The Hartree-Fock state is |{hf_string}>')
        
        # reference energies
        self.hf_energy  = calculated_molecule.hf_energy
        self.mp_energy  = calculated_molecule.mp2_energy
        self.cisd_energy= calculated_molecule.cisd_energy
        self.ccsd_energy= calculated_molecule.ccsd_energy
        self.fci_energy = calculated_molecule.fci_energy
        print(f'HF   energy = {self.hf_energy: .8f}') #Hartree-Fock
        print(f'MP2  energy = {self.mp_energy: .8f}') #Møller–Plesset
        print(f'CISD energy = {self.cisd_energy: .8f}')
        print(f'CCSD energy = {self.ccsd_energy: .8f}')
        if self.fci_energy is not None:
            print(f'FCI energy  = {self.fci_energy:.8f}')
        print(dashes)
        
        # Hamiltonian
        ham_ferm_data = calculated_molecule.get_molecular_hamiltonian()
        self.ham_fermionic = get_fermion_operator(ham_ferm_data)
        ham_jw = jordan_wigner(self.ham_fermionic)
        self.ham_dict = QubitOperator_to_dict(ham_jw, self.n_qubits)
        self.ham = PauliwordOp(self.ham_dict)
        self.ham_sor = PauliwordOp({op:coeff for op, coeff in self.second_order_response().items() if op in self.ham_dict})
        self.ham_sor.coeff_vec/=np.max(self.ham_sor.coeff_vec)

        # UCCSD Ansatz (for singlets)
        ccsd_single_amps = calculated_molecule.ccsd_single_amps
        ccsd_double_amps = calculated_molecule.ccsd_double_amps
        packed_amps = uccsd_singlet_get_packed_amplitudes(ccsd_single_amps,  ccsd_double_amps, self.n_qubits, self.n_electrons)
        ucc_singlet = uccsd_singlet_generator(packed_amps, self.n_qubits, self.n_electrons)
        ucc_jw = jordan_wigner(ucc_singlet)
        self.ucc_dict = QubitOperator_to_dict(ucc_jw, self.n_qubits)
        self.ucc = PauliwordOp(self.ucc_dict)

        # taper Hamiltonians + ansatz
        taper_hamiltonian = QubitTapering(self.ham)
        self.ham_tap = taper_hamiltonian.taper_it(ref_state=self.hf_state)
        self.sor_tap = taper_hamiltonian.taper_it(aux_operator=self.ham_sor, ref_state=self.hf_state)
        self.ucc_tap = taper_hamiltonian.taper_it(aux_operator=self.ucc, ref_state=self.hf_state)
        self.n_taper = taper_hamiltonian.n_taper
        self.tapered_qubits   = taper_hamiltonian.stab_qubit_indices
        self.untapered_qubits = taper_hamiltonian.free_qubit_indices
        self.hf_tapered = taper_hamiltonian.tapered_ref_state
        hf_tap_str = ''.join([str(i) for i in self.hf_tapered])
        
        print("Tapering information:")
        print(dashes)
        print(f'We are able to taper {self.n_taper} qubits from the Hamiltonian')
        print('The symmetry basis/sector is:') 
        print(taper_hamiltonian.symmetry_generators)
        print(f'The tapered Hartree-Fock state is |{hf_tap_str}>')
        print(dashes)

        # build CS-VQE model
        if basis_weighting == 'ham_coeff':
            weighting_operator = None
        elif basis_weighting == 'SOR':
            weighting_operator = self.sor_tap
        elif basis_weighting == 'num_commuting':
            weighting_operator = self.ham_tap.copy()
            weighting_operator.coeff_vec = np.ones(weighting_operator.n_terms)
        elif basis_weighting == 'UCCSD':
            weighting_operator = self.ucc_tap.copy()
            weighting_operator.coeff_vec = np.sin(weighting_operator.coeff_vec.imag)
        else:
            raise ValueError(f'Invalid basis_weighting {basis_weighting}:\n'+
                                'Must be one of ham_coeff, SOR or num_commuting.')

        super().__init__(operator=self.ham_tap,
                        ref_state=self.hf_tapered,
                        target_sqp='Z',
                        basis_weighting_operator=weighting_operator)

        print("CS-VQE information:")
        print(dashes)
        print("Noncontextual GS energy:", self.noncontextual_energy)#, ' // matches original?', match_original)
        print("Symmetry generators:    ") 
        print(self.symmetry_generators)
        print("Clique representatives: ")
        print(self.clique_operator)
        print(dashes)

    def update_basis(self, basis):
        """ for testing purposes
        """
        basis_order = np.lexsort(basis.adjacency_matrix)
        basis = StabilizerOp(basis.symp_matrix[basis_order],np.ones(basis.n_terms))
        self.noncontextual_basis = basis
        super().__init__(operator=self.ham_tap, ref_state=self.hf_tapered)
    
    def sor_data(self):
        """ Calculate the w(i) function 
        as in https://arxiv.org/pdf/1406.4920.pdf
        """
        w = {i:0 for i in range(self.n_qubits)}
        for f_op,coeff in self.ham_fermionic.terms.items():
            if len(f_op)==2:
                (p,p_ex),(q,q_ex) = f_op
                # self-interaction terms p==q
                if p==q:
                    w[p] += coeff
            if len(f_op)==4:
                (p,p_ex),(q,q_ex),(r,r_ex),(s,s_ex) = f_op
                #want p==r and q==s for hopping
                if p==r:
                    if q==s and self.hf_state[q]==1:
                        w[p]+=coeff
        return w


    def second_order_response(self):
        """ Calculate the I_a Hamiltonian term importance metric 
        as in https://arxiv.org/pdf/1406.4920.pdf
        """
        w = self.sor_data()
        f_out = FermionOperator()
        for H_a,coeff in self.ham_fermionic.terms.items():
            if len(H_a)==4:
                (p,p_ex),(q,q_ex),(r,r_ex),(s,s_ex) = H_a
                Delta_pqrs = abs(w[p]+w[q]-w[r]-w[s])
                if Delta_pqrs == 0:
                    I_a = 1e15
                else:
                    I_a = (abs(coeff)**2)/Delta_pqrs
                
                f_out += FermionOperator(H_a, I_a)
        f_out_jw = jordan_wigner(f_out)
        f_out_q = QubitOperator_to_dict(f_out_jw, self.n_qubits)
        return f_out_q
            
    ###############################################
    ######### for running VQE simulations #########
    ###############################################

    def _greedy_search(self, n_sim_qubits: int, pool: set, depth: int, print_errors: bool):
        """ for depth d, greedily select stabilizers to relax d-many at a time, choosing
        that which minimizes the CS-VQE error. This heuristic scales as O(N^d).
        In https://doi.org/10.22331/q-2021-05-14-456 d=2 was taken.
        """
        if n_sim_qubits<depth:
            depth = n_sim_qubits
        if n_sim_qubits == 0:
            # once the number of simulation qubits is exhausted, return the stabilizer pool
            # these are the stabilizers the heuristic has chosen to enforce
            return pool
        else:
            cs_vqe_errors = []
            # search over combinations from the stabilizer index pool of length d (the depth)
            for relax_indices in combinations(pool, r=depth):
                relax_indices = list(relax_indices)
                stab_indices = list(pool.difference(relax_indices))
                # perform the stabilizer subsapce projection and compute error
                # to be replaced with actual VQE simulation instead of direct diagonalization
                projected = self.contextual_subspace_projection(stab_indices)
                error = abs(exact_gs_energy(projected.to_sparse_matrix)[0]-self.fci_energy)
                cs_vqe_errors.append([relax_indices, error])
            # choose the best error and remove the corresponding stabilizer indices from the pool
            best_relax_indices, best_error = sorted(cs_vqe_errors, key=lambda x:x[1])[0]
            if print_errors:
                print(f'{projected.n_qubits}-qubit CS-VQE error: {best_error: .6f}')
            new_pool = pool.difference(best_relax_indices)
            # perform an N-d qubit search over the reduced pool 
            return self._greedy_search(n_sim_qubits = n_sim_qubits-depth,
                            pool=new_pool, 
                            depth=depth, 
                            print_errors=print_errors)

    def greedy_search(self, n_sim_qubits, depth=1, print_errors=False):
        """ wraps the _greedy_search recursive method for stabilizer relaxation ordering
        """
        # take the full stabilizer pool
        all_indices = set(range(self.ham_tap.n_qubits))
        return list(
            self._greedy_search(
                n_sim_qubits=n_sim_qubits, 
                pool=all_indices, 
                depth=depth, 
                print_errors=print_errors
            )
        )
