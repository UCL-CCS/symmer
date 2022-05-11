from symred.symplectic import PauliwordOp
from symred.utils import QubitOperator_to_dict
from symred.chem import FermionicHamilt, FermioniCC, PySCFDriver
from openfermion import get_fermion_operator, jordan_wigner
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
        charge=1, 
        basis='STO-3G', 
        spin=1,
        run_fci = True,
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
        self.print_info = print_info
        self.calculate(run_fci=run_fci)

        # build the fermionic hamiltonian/CC operator
        self.H_fermion = FermionicHamilt(self.pyscf_obj.pyscf_hf)
        self.T_fermion = FermioniCC(self.pyscf_obj.pyscf_ccsd)
        self.H_fermion.build_operator()
        self.T_fermion.build_operator()

        self.n_qubits = self.H_fermion.n_qubits
        if self.print_info:
            print()
            print('Number of qubits:', self.n_qubits)

        self.H = get_fermion_operator(self.H_fermion.fermionic_molecular_hamiltonian)
        self.T = self.T_fermion.fermionic_cc_operator
        
        # map to QubitOperator via Jordan-Wigner
        self.H_jw = jordan_wigner(self.H)
        self.T_jw = jordan_wigner(self.T)

        # convert to PauliwordOp
        self.H_q = PauliwordOp(QubitOperator_to_dict(self.H_jw, self.n_qubits))
        self.T_q = PauliwordOp(QubitOperator_to_dict(self.T_jw, self.n_qubits))
        self.T_q.coeff_vec = self.T_q.coeff_vec.imag

    def calculate(self,
        run_mp2  = True, 
        run_cisd = True, 
        run_ccsd = True, 
        run_fci  = True,
        convergence = 1e-6, 
        max_hf_cycles=100_000,
        ram = 8_000) -> None:

        self.pyscf_obj = PySCFDriver(self.geometry,
                                self.basis,
                                charge=self.charge,
                                run_mp2=run_mp2,
                                run_cisd=run_cisd,
                                run_ccsd=run_ccsd,
                                run_fci=run_fci,
                                convergence=convergence,
                                max_ram_memory=ram,
                                max_hf_cycles=max_hf_cycles,                   
        )
        self.pyscf_obj.run_pyscf()

        if self.print_info:
            print('HF converged?  ', self.pyscf_obj.pyscf_hf.converged)
            print('CCSD converged?', self.pyscf_obj.pyscf_ccsd.converged)
        if run_fci:
            if self.print_info:
                print('FCI converged? ', self.pyscf_obj.pyscf_fci.converged)
            self.fci_energy = self.pyscf_obj.pyscf_fci.e_tot
        else:
            self.fci_energy = None
            
        self.hf_energy = self.pyscf_obj.pyscf_hf.e_tot
        self.mp2_energy = self.pyscf_obj.pyscf_mp2.e_tot
        self.ccsd_energy = self.pyscf_obj.pyscf_ccsd.e_tot

        if self.print_info:
            print()
            print(f'HF energy:   {self.hf_energy}')
            print(f'MP2 energy:  {self.mp2_energy}')
            print(f'CCSD energy: {self.ccsd_energy}')
            print(f'FCI energy:  {self.fci_energy}')
            print()
 
