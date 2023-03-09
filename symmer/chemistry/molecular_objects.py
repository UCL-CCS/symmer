import os
import warnings
import numpy as np
from math import comb
from typing import Optional, Union
from pathlib import Path
from cached_property import cached_property
from pyscf import ao2mo, gto, scf, mp, ci, cc, fci, mcscf
from pyscf.lib import StreamObject
from pyscf.cc.addons import spatial2spin
from symmer.symplectic import QuantumState, PauliwordOp
from openfermion import InteractionOperator, FermionOperator
from openfermion.transforms.opconversions.jordan_wigner import _jordan_wigner_interaction_op
from openfermion.transforms.opconversions.bravyi_kitaev import _bravyi_kitaev_interaction_operator
from symmer.chemistry.utils import (
    get_fermionic_number_operator, 
    get_fermionic_spin_operators,
    get_fermionic_up_down_parity_operators,
    get_parity_operators_BK, get_parity_operators_JW,
    array_to_dict_nonzero_indices, 
    fermion_to_qubit_operator,
    build_bk_matrix
)

class PySCFDriver:
    """Function run PySCF chemistry calc.

    Args:
        geometry (str): Path to .xyz file containing molecular geometry or raw xyz string.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        unit (str): molecular geometry unit 'Angstrom' or 'Bohr'
        max_hf_cycles (int): max number of Hartree-Fock iterations allowed
        spin (int): 2S, twice the total spin operator
        symmetry (str, bool): Point-group symmetry of molecular system (see pyscf for details)
        hf_method (str): Type of Hartree-Fock calulcation, one of the following:
                        restricted (RHF), restricted open-shell (ROHF), 
                        unrestriced (UHF) or generalised (GHF) Hartree-Fock.

    Attributes:

    """

    def __init__(
        self,
        geometry: str,
        basis: str,
        convergence: Optional[float] = 1e-6,
        charge: Optional[int] = 0,
        max_ram_memory: Optional[int] = 4000,
        pyscf_print_level: int = 1,
        savefile: Optional[Path] = None,
        unit: Optional[str] = "angstrom",
        max_hf_cycles: int = 50,
        spin: Optional[int] = 0,
        symmetry: Optional[Union[str, bool]] = False,

        run_mp2: Optional[bool] = False,
        run_cisd: Optional[bool] = False,
        run_ccsd: Optional[bool] = False,
        run_fci: Optional[bool] = False,
        hf_method: Optional[str] = 'RHF'
    ):
        if convergence>1e-2:
            warnings.warn('note scf convergence threshold not very low')

        self.geometry = geometry
        self.basis = basis.lower()
        self.convergence = convergence
        self.charge = charge
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.savefile = savefile
        self.unit = unit
        self.max_hf_cycles = max_hf_cycles
        self.symmetry = symmetry

        self.run_mp2  = run_mp2
        self.run_cisd = run_cisd
        self.run_ccsd = run_ccsd
        self.run_fci = run_fci
        self.hf_method = hf_method
        self.spin = spin

    def _build_mol(self) -> gto.mole:
        """Function to build PySCF molecule.

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        if os.path.exists(self.geometry):
            # geometry is an xyz file
            full_mol = gto.Mole(
                atom=self.geometry,
                basis=self.basis,
                charge=self.charge,
                unit=self.unit,
                spin=self.spin,
                symmetry=self.symmetry
            ).build()
        else:
            # geometry is raw xyz string
            full_mol = gto.Mole(
                atom=self.geometry[3:],
                basis=self.basis,
                charge=self.charge,
                unit=self.unit,
                spin=self.spin,
                symmetry=self.symmetry
            ).build()
        return full_mol


    @cached_property
    def pyscf_hf(self) -> StreamObject:
        """Run Hartree-Fock calculation."""
        mol_full = self._build_mol()
        # run Hartree-Fock

        if self.hf_method.upper() == 'RHF':
            global_hf = scf.RHF(mol_full)
        elif self.hf_method.upper() == 'ROHF':
            global_hf = scf.ROHF(mol_full)
        elif self.hf_method.upper() == 'UHF':
            raise NotImplementedError('Unrestricted HF currently not implemented')
        elif self.hf_method.upper() == 'GHF':
            raise NotImplementedError('Generalised HF currently not implemented')
        else:
            raise ValueError('Unknown Hartree-Fock method, must be one of RHF, ROHF, UHF or GHF.')

        global_hf.conv_tol = self.convergence
        global_hf.max_memory = self.max_ram_memory
        global_hf.verbose = self.pyscf_print_level
        global_hf.max_cycle = self.max_hf_cycles
        global_hf.kernel()
        if global_hf.converged is False:
            warnings.warn("Hartree-Fock calc not converged")
        
        return global_hf

    def run_pyscf(self):

        if self.run_mp2:
            self.pyscf_mp2 = mp.MP2(self.pyscf_hf)
            self.pyscf_mp2.verbose = self.pyscf_print_level
            self.pyscf_mp2.run()

        if self.run_cisd:
            self.pyscf_cisd = ci.CISD(self.pyscf_hf)
            self.pyscf_cisd.verbose = self.pyscf_print_level
            self.pyscf_cisd.run()
            if self.pyscf_cisd.converged is False:
                warnings.warn("CISD calc not converged")


        if self.run_ccsd:
            self.pyscf_ccsd = cc.CCSD(self.pyscf_hf)
            self.pyscf_ccsd.verbose = self.pyscf_print_level
            # self.pyscf_ccsd.diis = False
            self.pyscf_ccsd.max_cycle = self.max_hf_cycles

            self.pyscf_ccsd.run()
            if self.pyscf_ccsd.converged is False:
                warnings.warn("CCSD calc not converged")

        # Run FCI.
        if self.run_fci:
            # check how large calc will be and raise error if too big.
            n_deterimants = comb(2*self.pyscf_hf.mol.nao,
                                      self.pyscf_hf.mol.nelectron)
            if n_deterimants > 2**25:
                raise NotImplementedError(f'FCI calc too expensive. Number of determinants = {n_deterimants} ')

            self.pyscf_fci = fci.FCI(self.pyscf_hf.mol, self.pyscf_hf.mo_coeff)
            self.pyscf_fci.verbose = 0
            self.pyscf_fci.kernel()
            if self.pyscf_fci.converged is False:
                warnings.warn("FCI calc not converged")


class MolecularIntegrals:
    """Class to build Fermionic molecular hamiltonians.

      Holds fermionic operators + integrals
      coefficients assume a particular convention which depends on how integrals are labeled:
      h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
      h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
      In this labelling convention, the molecular Hamiltonian becomes:
      H =\sum_{p,q} h[p,q] a_p^\dagger a_q
        + 0.5 * \sum_{p,q,r,s} h[p,q,r,s] a_p^\dagger a_q^\dagger a_r a_s

    """   
    def __init__(self, scf_obj, index_ordering='physicist') -> None:
        self.scf_obj = scf_obj
        self.c_matrix = scf_obj.mo_coeff
        self.n_spatial_orbs = scf_obj.mol.nao
        self.index_ordering = index_ordering
        self.n_spin_orbs = self.n_qubits = 2 * self.n_spatial_orbs
        self.n_electrons = scf_obj.mol.nelectron
        self.fermionic_molecular_hamiltonian=None
        self.qubit_molecular_hamiltonian=None
        
    @cached_property
    def _one_body_integrals(self):
        one_body_integrals = (
            self.c_matrix.T @ self.scf_obj.get_hcore() @ self.c_matrix
        )
        return one_body_integrals
    
    @cached_property
    def _two_body_integrals(self):
        two_body_integrals = ao2mo.restore(1, 
            ao2mo.kernel(self.scf_obj.mol,self.scf_obj.mo_coeff),
            self.scf_obj.mo_coeff.shape[1]
        )
        if self.index_ordering == 'physicist':
            # mapping to physicists' notations from PySCF chemists'
            # p, q, r, s -> p^ r^ s q, e.g. ⟨01|23⟩ ==> ⟨02|31⟩
            two_body_integrals = two_body_integrals.transpose(0,2,3,1)
        return two_body_integrals
    
    @cached_property
    def core_h_spin_basis(self):
        h_core_mo_basis_spin = np.zeros([self.n_spin_orbs]*2)
        h_core_mo_basis_spin[ ::2, ::2] = self._one_body_integrals
        h_core_mo_basis_spin[1::2,1::2] = self._one_body_integrals
        return h_core_mo_basis_spin
    
    @cached_property
    def eri_spin_basis(self):
        # alpha/beta electron indexing for chemists' and physicists' notation
        if self.index_ordering == 'chemist':
            a,b,c,d=(0,0,1,1)
            e,f,g,h=(1,1,0,0)
        elif self.index_ordering == 'physicist':
            a,b,c,d=(0,1,1,0)
            e,f,g,h=(1,0,0,1)
            
        eri_mo_basis_spin = np.zeros([self.n_spin_orbs]*4)
        # same spin (even *or* odd indices)
        eri_mo_basis_spin[ ::2, ::2, ::2, ::2] = self._two_body_integrals
        eri_mo_basis_spin[1::2,1::2,1::2,1::2] = self._two_body_integrals
        # different spin (even *and* odd indices)
        eri_mo_basis_spin[a::2,b::2,c::2,d::2] = self._two_body_integrals
        eri_mo_basis_spin[e::2,f::2,g::2,h::2] = self._two_body_integrals
        return eri_mo_basis_spin

def get_hamiltonian(scf_obj=None,
        constant_shift=0, hcore=None, eri=None, 
        operator_type='qubit', qubit_transformation='JW'
    ):
    """
    """
    if scf_obj is not None:
        integral_storage = MolecularIntegrals(scf_obj)
        constant_shift = scf_obj.energy_nuc()
        hcore = integral_storage.core_h_spin_basis
        eri = integral_storage.eri_spin_basis
        n_qubits = integral_storage.n_qubits
    else:
        assert (
            hcore is not None and
            eri is not None
        ), 'Must supply molecular integrals'
        n_qubits = eri.shape[0]

    if operator_type == 'qubit':
        interaction_operator = InteractionOperator(
            constant = constant_shift, 
            one_body_tensor = hcore, 
            two_body_tensor = eri*.5
        )
        if qubit_transformation in ['JW', 'jordan_wigner']:
            qubit_op = _jordan_wigner_interaction_op(interaction_operator, n_qubits=n_qubits)
        elif qubit_transformation in ['BK', 'bravyi_kitaev']:
            qubit_op = _bravyi_kitaev_interaction_operator(interaction_operator, n_qubits=n_qubits)
        else:
            raise ValueError('Unrecognised qubit transformation')
        return PauliwordOp.from_openfermion(qubit_op, n_qubits=n_qubits) 
    
    elif operator_type == 'fermion':
        one_body_coefficients = array_to_dict_nonzero_indices(hcore)
        two_body_coefficients = array_to_dict_nonzero_indices(eri)

        fermionic_molecular_hamiltonian = FermionOperator('', constant_shift)
        for (p,q), coeff in one_body_coefficients.items():
            fermionic_molecular_hamiltonian += FermionOperator(f'{p}^ {q}', coeff)
        for (p,q,r,s), coeff in two_body_coefficients.items():
            fermionic_molecular_hamiltonian += FermionOperator(f'{p}^ {q}^ {r} {s}', coeff*.5)
        return fermionic_molecular_hamiltonian

def get_hf_state(scf_obj, state_type = 'QuantumState', qubit_transformation='JW'):
    """
    """
    n_spinorbs = scf_obj.mol.nao*2

    if scf_obj.__class__.__name__.find('RHF') != -1:
        n_alpha = n_beta = scf_obj.mol.nelectron//2
    else:
        n_alpha, n_beta = scf_obj.nelec

    hf_array = np.zeros(n_spinorbs, dtype=int)
    hf_array[::2] = np.hstack([np.ones(n_alpha), np.zeros(n_spinorbs//2-n_alpha)])
    hf_array[1::2] = np.hstack([np.ones(n_beta), np.zeros(n_spinorbs//2-n_beta)])
    
    if qubit_transformation in ['bravyi_kitaev', 'BK']:
        hf_array = (build_bk_matrix(n_spinorbs) @ hf_array.reshape(-1,1)).reshape(-1) % 2
    else:
        assert qubit_transformation in ['jordan_wigner', 'JW',], 'Invaild qubit transformation'
    
    if state_type == 'array':
        return hf_array
    elif state_type == 'QuantumState':
        return QuantumState(hf_array)

def get_coupled_cluster_operator(cc_obj=None, t1=None, t2=None, 
        operator_type='qubit', qubit_transformation='JW', orbspin=None
    ):
    """
    """
    if cc_obj is not None:
        t1 = spatial2spin(cc_obj.t1, orbspin=orbspin)
        t2 = spatial2spin(cc_obj.t2, orbspin=orbspin)
    else:
        assert (
            t1 is not None and
            t2 is not None
        ), 'Must supply t1 and t2 matrices'

    no, nv = t1.shape
    nmo = no + nv
    
    # dictionary of single aplitudes of form {(i,j):t_ij}
    single_amplitudes = np.zeros((nmo, nmo))
    single_amplitudes[no:,:no] = t1.T
    single_amp_dict = array_to_dict_nonzero_indices(single_amplitudes)
    
    # dictionary of double aplitudes of form {(i,j,k,l):t_ijkl}
    double_amplitudes = np.zeros((nmo, nmo, nmo, nmo))
    double_amplitudes[no:,:no,no:,:no] = .25 * t2.transpose(2,0,3,1)
    double_amp_dict = array_to_dict_nonzero_indices(double_amplitudes)
    
    generator = FermionOperator()
    for (i, j), t_ij in single_amp_dict.items():
        generator += FermionOperator(f'{i}^ {j}', t_ij)
    for (i, j, k, l), t_ijkl in double_amp_dict.items():
        generator += FermionOperator(f'{i}^ {j} {k}^ {l}', t_ijkl)

    if operator_type == 'fermion':
        return generator
    elif operator_type == 'qubit':
        return fermion_to_qubit_operator(generator, qubit_transformation=qubit_transformation, n_qubits=nmo)
    
def get_perturbation_operator(mp_obj=None, t2=None, 
        operator_type='qubit', qubit_transformation='JW'
    ):
    """
    """
    if mp_obj is not None:
        t2 = spatial2spin(mp_obj.t2)
    else:
        assert t2 is not None, 'Must supply t2 matrix'

    no, nv = t2.shape[1:3]
    nmo = no + nv
    
    # dictionary of double aplitudes of form {(i,j,k,l):t_ijkl}
    double_amplitudes = np.zeros((nmo, nmo, nmo, nmo))
    double_amplitudes[no:,:no,no:,:no] = .25 * t2.transpose(2,0,3,1)
    double_amp_dict = array_to_dict_nonzero_indices(double_amplitudes)
    
    generator = FermionOperator()
    for (i, j, k, l), t_ijkl in double_amp_dict.items():
        generator += FermionOperator(f'{i}^ {j} {k}^ {l}', t_ijkl)
    #generator -= hermitian_conjugated(generator)

    if operator_type == 'fermion':
        return generator
    elif operator_type == 'qubit':
        return fermion_to_qubit_operator(generator, qubit_transformation=qubit_transformation, n_qubits=nmo)

def get_molecular_symmetries(n_qubits, operator_type='qubit', qubit_transformation='JW'):
    """
    """
    number_op = get_fermionic_number_operator(N_qubits=n_qubits)
    S2_op, Sz_op = get_fermionic_spin_operators(N_qubits=n_qubits)
    
    if operator_type == 'qubit':
        number_op = fermion_to_qubit_operator(number_op, qubit_transformation=qubit_transformation, n_qubits=n_qubits)
        S2_op = fermion_to_qubit_operator(S2_op, qubit_transformation=qubit_transformation, n_qubits=n_qubits)
        Sz_op = fermion_to_qubit_operator(Sz_op, qubit_transformation=qubit_transformation, n_qubits=n_qubits)
        if qubit_transformation in ['JW', 'jordan_wigner']:
            up_parity_op, down_parity_op = get_parity_operators_JW(n_qubits=n_qubits)
        elif qubit_transformation in ['BK', 'bravyi_kitaev']:
            up_parity_op, down_parity_op = get_parity_operators_BK(n_qubits=n_qubits)
    
    elif operator_type == 'fermion':
        up_parity_op, down_parity_op = get_fermionic_up_down_parity_operators(N_qubits=n_qubits)
    
    symmetry_dict = {
            'number':number_op,
            'S^2':S2_op,
            'S_z':Sz_op,
            'up_parity':up_parity_op,
            'down_parity':down_parity_op
        }
    
    return symmetry_dict