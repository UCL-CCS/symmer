import numpy as np
from pyscf.scf.addons import get_ghf_orbspin
from symmer import QuantumState
from symmer.symplectic.utils import safe_PauliwordOp_to_dict
from symmer.chemistry.utils import list_to_xyz
from symmer.chemistry import (
    PySCFDriver,
    get_hamiltonian, 
    get_hf_state, 
    get_coupled_cluster_operator, 
    get_perturbation_operator,
    get_molecular_symmetries
)

class MoleculeBuilder:
    def __init__(self, 
        geometry, 
        charge=0, 
        basis='STO-3G', 
        spin=0,
        run_mp2  = True,
        run_cisd = False,
        run_ccsd = True,
        run_fci  = True,
        qubit_transformation = 'jordan_wigner',
        hf_method = 'RHF',
        CI_ansatz = 'CISD',
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
        self.qubit_transformation = qubit_transformation
        self.symmetry = symmetry
        self.print_info = print_info
        self.CI_ansatz = CI_ansatz
        self.calculate(
            run_mp2=run_mp2,run_cisd=run_cisd,run_ccsd=run_ccsd,run_fci=run_fci, 
            hf_method=hf_method)
        self.n_particles = self.pyscf_obj.pyscf_hf.mol.nelectron
        self.n_qubits = self.pyscf_obj.pyscf_hf.mol.nao*2
        if self.print_info:
            print(f'System of {self.n_particles} particles in {self.n_qubits} spin-orbitals.')

        if self.pyscf_obj.pyscf_hf.__class__.__name__.find('RHF') != -1:
            n_electron = self.pyscf_obj.pyscf_hf.mol.nelectron
            self.n_alpha = self.n_beta = n_electron//2
        else:
            self.n_alpha, self.n_beta = self.pyscf_obj.pyscf_hf.nelec
        orbspin = get_ghf_orbspin(
            self.pyscf_obj.pyscf_hf.mo_energy,
            self.pyscf_obj.pyscf_hf.mo_occ, 
            is_rhf=True
        )
        # Buid molecular operator
        self.H_q  = get_hamiltonian(self.pyscf_obj.pyscf_hf, qubit_transformation=qubit_transformation)
        self.CC_q = get_coupled_cluster_operator(self.pyscf_obj.pyscf_ccsd, qubit_transformation=qubit_transformation, orbspin=orbspin)
        self.MP_q = get_perturbation_operator(self.pyscf_obj.pyscf_mp2, qubit_transformation=qubit_transformation)
        self.hf_array = get_hf_state(self.pyscf_obj.pyscf_hf, state_type='array', qubit_transformation=qubit_transformation)
        self.hf_state = QuantumState(self.hf_array) 
        # build various molecular symmetries
        self.symmetries = get_molecular_symmetries(self.n_qubits, qubit_transformation=qubit_transformation)

        # self.CI, self.total_CI_energy, self.psi_CI = self.H_fermion.get_fermionic_CI_ansatz(
        #     S=spin/2, method=CI_ansatz
        # )
        if run_cisd and self.CI_ansatz == 'CISD':
            raise NotImplementedError    
            assert(np.isclose(self.total_CI_energy,self.cisd_energy)), 'Manual CISD calculation does not match PySCF'
        
    def calculate(self,
        run_mp2  = True, 
        run_cisd = True, 
        run_ccsd = True, 
        run_fci  = True,
        hf_method= 'RHF',
        convergence = 1e-6, 
        max_hf_cycles=100_000,
        ram = 8_000) -> None:

        self.pyscf_obj = PySCFDriver(
            self.geometry,
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
            #print(f'CISD energy: {self.cisd_energy}')
            print(f'FCI energy:  {self.fci_energy}')
            print()

    def data_dictionary(self):
        mol_data = {
            'qubit_encoding': self.qubit_transformation,
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
                'number_operator': safe_PauliwordOp_to_dict(self.symmetries['number']),
                'S^2_operator': safe_PauliwordOp_to_dict(self.symmetries['S^2']),
                'Sz_operator': safe_PauliwordOp_to_dict(self.symmetries['S_z']),
                'alpha_parity_operator': safe_PauliwordOp_to_dict(self.symmetries['up_parity']),
                'beta_parity_operator': safe_PauliwordOp_to_dict(self.symmetries['down_parity'])
            }
        }

        if self.pyscf_obj.run_mp2:
            mol_data['calculated_properties']['MP2'] = {
                'energy':self.mp2_energy, 'converged':bool(self.pyscf_obj.pyscf_hf.converged)}
            mol_data['auxiliary_operators']['MP2_operator'] = safe_PauliwordOp_to_dict(self.MP_q)
        if self.pyscf_obj.run_ccsd:
            mol_data['calculated_properties']['CCSD'] = {
                'energy':self.ccsd_energy, 'converged':bool(self.pyscf_obj.pyscf_ccsd.converged)}
            mol_data['auxiliary_operators']['CCSD_operator'] = safe_PauliwordOp_to_dict(self.CC_q)
        if self.pyscf_obj.run_cisd:
            mol_data['calculated_properties'][self.CI_ansatz] = {
                'energy':self.total_CI_energy, 'converged':bool(self.pyscf_obj.pyscf_cisd.converged)}
            mol_data['auxiliary_operators'][f'{self.CI_ansatz}_operator'] = safe_PauliwordOp_to_dict(self.CI_q)
        if self.pyscf_obj.run_fci:
            mol_data['calculated_properties']['FCI'] = {
                'energy':self.fci_energy, 'converged':bool(self.pyscf_obj.pyscf_fci.converged)}        
        
        return mol_data


