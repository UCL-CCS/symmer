from cached_property import cached_property
from symmer.symplectic import PauliwordOp
from symmer.symplectic.utils import QubitOperator_to_dict, safe_PauliwordOp_to_dict
from symmer.chemistry import FermionicHamiltonian, FermioniCC, PySCFDriver
from symmer.chemistry.utils import (
    build_bk_matrix, fermion_to_qubit_operator, get_fermionic_number_operator, get_fermionic_spin_operators, get_parity_operators_JW, get_parity_operators_BK
    )
from pyscf.scf.addons import get_ghf_orbspin
from openfermion import get_fermion_operator, jordan_wigner, FermionOperator
from typing import Tuple, List

def list_to_xyz(geometry: List[Tuple[str, Tuple[float, float, float]]]) -> str:
    """ Convert the geometry stored as a list to xyz string
    """
    xyz_file = str(len(geometry))+'\n '

    for atom, coords in geometry:
        xyz_file += '\n'+atom+'\t'
        xyz_file += '\t'.join(list(map(str, coords)))
    
    return xyz_file

class MoleculeBuilder:
    def __init__(self, 
        geometry, 
        charge=0, 
        basis='STO-3G', 
        spin=0,
        run_mp2  = True,
        run_cisd = True,
        run_ccsd = True,
        run_fci  = True,
        qubit_mapping_str = 'jordan_wigner',
        hf_method = 'RHF',
        symmetry = False,
        print_info = True) -> None:
        """
        """
        if isinstance(geometry, list):
            geometry = list_to_xyz(geometry)
        if print_info:
            print('Molecule geometry:')
            print(geometry[4:])
            print()
        self.geometry = geometry
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.qubit_mapping_str = qubit_mapping_str
        self.symmetry = symmetry
        self.print_info = print_info
        self.calculate(
            run_mp2=run_mp2,run_cisd=run_cisd,run_ccsd=run_ccsd,run_fci=run_fci, 
            hf_method=hf_method)
        self.n_particles = self.pyscf_obj.pyscf_hf.mol.nelectron
        if hf_method.find('RHF') != -1:
            n_electron = self.pyscf_obj.pyscf_hf.mol.nelectron
            self.n_alpha = self.n_beta = n_electron//2
        else:
            self.n_alpha, self.n_beta = self.pyscf_obj.pyscf_hf.nelec
        orbspin = get_ghf_orbspin(
            self.pyscf_obj.pyscf_hf.mo_energy,
            self.pyscf_obj.pyscf_hf.mo_occ, 
            is_rhf=True
        )

        # build the fermionic hamiltonian/CC operator
        self.H_fermion = FermionicHamiltonian(self.pyscf_obj.pyscf_hf)
        self.T_fermion = FermioniCC(self.pyscf_obj.pyscf_ccsd)
        self.H_fermion.build_fermionic_hamiltonian_operator()
        self.T_fermion.build_operator(orbspin)

        self.n_qubits = self.H_fermion.n_qubits
        self.hf_array = self.H_fermion.hf_fermionic_basis_state
        if self.qubit_mapping_str == 'bravyi_kitaev':
            self.hf_array = (build_bk_matrix(self.n_qubits) @ self.hf_array.reshape(-1,1)).reshape(-1) % 2

        if self.print_info:
            print()
            print('Number of qubits:', self.n_qubits)

        self.H = get_fermion_operator(self.H_fermion.fermionic_molecular_hamiltonian)
        self.T = self.T_fermion.fermionic_cc_operator

        # map to QubitOperator via fermion -> qubit mapping and convert to PauliwordOp
        self.H_q = fermion_to_qubit_operator(self.H, self.qubit_mapping_str, N_qubits=self.n_qubits)
        if len(self.T.terms)==0:
            self.T_q = PauliwordOp.empty(self.n_qubits)
        else:
            self.T_q = fermion_to_qubit_operator(self.T, self.qubit_mapping_str, N_qubits=self.n_qubits)

        self.UCC_q = self.T_q - self.T_q.dagger
        self.UCC_q.coeff_vec = self.UCC_q.coeff_vec.imag
        self.SOR_q = self.second_order_response()

    def calculate(self,
        run_mp2  = True, 
        run_cisd = True, 
        run_ccsd = True, 
        run_fci  = True,
        hf_method= 'RHF',
        convergence = 1e-6, 
        max_hf_cycles=100_000,
        ram = 8_000) -> None:

        self.pyscf_obj = PySCFDriver(self.geometry,
                                self.basis,
                                charge=self.charge,
                                spin=self.spin,
                                run_mp2=run_mp2,
                                run_cisd=run_cisd,
                                run_ccsd=run_ccsd,
                                run_fci=run_fci,
                                hf_method=hf_method,
                                symmetry=self.symmetry,
                                convergence=convergence,
                                max_ram_memory=ram,
                                max_hf_cycles=max_hf_cycles,                   
        )
        self.pyscf_obj.run_pyscf()

        if run_mp2:
            self.mp2_energy = self.pyscf_obj.pyscf_mp2.e_tot
        else:
            self.mp2_energy = None
        if run_cisd:
            if self.print_info:
                print('CISD converged? ', self.pyscf_obj.pyscf_cisd.converged)
            self.cisd_energy = self.pyscf_obj.pyscf_cisd.e_tot
        else:
            self.cisd_energy = None
        if run_ccsd:
            if self.print_info:
                print('FCI converged? ', self.pyscf_obj.pyscf_ccsd.converged)
            self.ccsd_energy = self.pyscf_obj.pyscf_ccsd.e_tot
        else:
            self.ccsd_energy = None
        if run_fci:
            if self.print_info:
                print('FCI converged? ', self.pyscf_obj.pyscf_fci.converged)
            self.fci_energy = self.pyscf_obj.pyscf_fci.e_tot
        else:
            self.fci_energy = None
            
        self.hf_energy = self.pyscf_obj.pyscf_hf.e_tot
        
        if self.print_info:
            print()
            print(f'HF energy:   {self.hf_energy}')
            print(f'MP2 energy:  {self.mp2_energy}')
            print(f'CCSD energy: {self.ccsd_energy}')
            print(f'CISD energy: {self.cisd_energy}')
            print(f'FCI energy:  {self.fci_energy}')
            print()

    def sor_data(self):
        """ Calculate the w(i) function 
        as in https://arxiv.org/pdf/1406.4920.pdf
        """
        w = {i:0 for i in range(self.n_qubits)}
        for f_op,coeff in self.H.terms.items():
            if len(f_op)==2:
                (p,p_ex),(q,q_ex) = f_op
                # self-interaction terms p==q
                if p==q:
                    w[p] += coeff
            if len(f_op)==4:
                (p,p_ex),(q,q_ex),(r,r_ex),(s,s_ex) = f_op
                #want p==r and q==s for hopping
                if p==r:
                    if q==s and self.hf_array[q]==1:
                        w[p]+=coeff
        return w

    def second_order_response(self):
        """ Calculate the I_a Hamiltonian term importance metric 
        as in https://arxiv.org/pdf/1406.4920.pdf
        """
        w = self.sor_data()
        f_out = FermionOperator()
        for H_a,coeff in self.H.terms.items():
            if len(H_a)==4:
                (p,p_ex),(q,q_ex),(r,r_ex),(s,s_ex) = H_a
                Delta_pqrs = abs(w[p]+w[q]-w[r]-w[s])
                if Delta_pqrs == 0:
                    I_a = 1e6
                else:
                    I_a = (abs(coeff)**2)/Delta_pqrs
                
                f_out += FermionOperator(H_a, I_a)
        f_out_jw = jordan_wigner(f_out)
        f_out_q = QubitOperator_to_dict(f_out_jw, self.n_qubits)
        return PauliwordOp.from_dictionary(f_out_q)
    
    @cached_property
    def number_operator(self):
        """
        """
        fermionic_number_op = get_fermionic_number_operator(self.n_qubits)
        return fermion_to_qubit_operator(
            fermionic_number_op, self.qubit_mapping_str, N_qubits=self.n_qubits
        )

    @cached_property
    def spin_operators(self):
        """
        """
        S2, Sz = get_fermionic_spin_operators(self.n_qubits)
        S2_q = fermion_to_qubit_operator(S2, self.qubit_mapping_str, N_qubits=self.n_qubits)
        Sz_q = fermion_to_qubit_operator(Sz, self.qubit_mapping_str, N_qubits=self.n_qubits)

        return S2_q, Sz_q

    @cached_property
    def up_down_parity_operators(self):
        """ Assumes alternating up/down spin orbitals
        """
        if self.qubit_mapping_str == 'jordan_wigner':
            parity_up_pword, parity_down_pword = get_parity_operators_JW(self.n_qubits)
        elif self.qubit_mapping_str == 'bravyi_kitaev':
            parity_up_pword, parity_down_pword = get_parity_operators_BK(self.n_qubits)
        else:
            raise ValueError('Unrecognised qubit mapping, must be one of jordan_wigner or bravyi_kitaev')
        return parity_up_pword, parity_down_pword

    def data_dictionary(self):
        mol_data = {
            'qubit_encoding': self.qubit_mapping_str,
            'unit':self.pyscf_obj.unit,
            'geometry': self.geometry,
            'basis': self.basis,
            'charge': int(self.charge),
            'spin': int(self.spin),
            'hf_array': self.hf_array.tolist(),
            'hf_method': f'{self.pyscf_obj.pyscf_hf.__module__}.{self.pyscf_obj.pyscf_hf.__class__.__name__}',
            'n_particles': {
                'total':int(self.n_particles), 
                'alpha':int(self.n_alpha), 'beta':int(self.n_beta)
            },
            'n_qubits': int(self.n_qubits),
            'convergence_threshold':self.pyscf_obj.convergence,
            'point_group':{
                'groupname':self.pyscf_obj.pyscf_hf.mol.groupname,
                'topgroup':self.pyscf_obj.pyscf_hf.mol.topgroup
            },
            'calculated_properties':{
                'HF':{'energy':self.hf_energy, 'converged':bool(self.pyscf_obj.pyscf_hf.converged)}
            },
            'auxiliary_operators':{
                #'HF_operator': safe_PauliwordOp_to_dict(self.HF_fermionic_op_q),
                'number_operator': safe_PauliwordOp_to_dict(self.number_operator),
                'S^2_operator': safe_PauliwordOp_to_dict(self.spin_operators[0]),
                'Sz_operator': safe_PauliwordOp_to_dict(self.spin_operators[1]),
                'alpha_parity_operator': safe_PauliwordOp_to_dict(self.up_down_parity_operators[0]),
                'beta_parity_operator': safe_PauliwordOp_to_dict(self.up_down_parity_operators[1])
            }    
        }

        if self.pyscf_obj.run_mp2:
            mol_data['calculated_properties']['MP2'] = {
                'energy':self.mp2_energy, 'converged':bool(self.pyscf_obj.pyscf_hf.converged)}
        if self.pyscf_obj.run_ccsd:
            mol_data['calculated_properties']['CCSD'] = {
                'energy':self.ccsd_energy, 'converged':bool(self.pyscf_obj.pyscf_ccsd.converged)}
            mol_data['auxiliary_operators']['UCCSD_operator'] = safe_PauliwordOp_to_dict(self.UCC_q)
        if self.pyscf_obj.run_fci:
            mol_data['calculated_properties']['FCI'] = {
                'energy':self.fci_energy, 'converged':bool(self.pyscf_obj.pyscf_fci.converged)}        
        
        return mol_data


