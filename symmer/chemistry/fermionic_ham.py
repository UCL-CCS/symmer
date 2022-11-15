from typing import Optional, Union
from pathlib import Path
import os
import numpy as np
from scipy.special import comb
from cached_property import cached_property
from openfermion import InteractionOperator, get_sparse_operator, FermionOperator, count_qubits
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import get_active_space_integrals
from pyscf import ao2mo, gto, scf, mp, ci, cc, fci
from pyscf.lib import StreamObject
from pyscf.cc.addons import spatial2spin
import warnings

class FermionicHamiltonian:
    """Class to build Fermionic molecular hamiltonians.

      Holds fermionic operators + integrals
      coefficients assume a particular convention which depends on how integrals are labeled:
      h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
      h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
      In this labelling convention, the molecular Hamiltonian becomes:
      H =\sum_{p,q} h[p,q] a_p^\dagger a_q
        + 0.5 * \sum_{p,q,r,s} h[p,q,r,s] a_p^\dagger a_q^\dagger a_r a_s

    """

    def __init__(
        self,
        scf_method: StreamObject,
    ) -> None:
        self.scf_method = scf_method
        self.fermionic_molecular_hamiltonian=None

        self.n_electrons = self.scf_method.mol.nelectron
        self.n_qubits = 2*self.scf_method.mol.nao

    @cached_property
    def _one_body_integrals(self) -> np.ndarray:
        """Get the one electron integrals: An N by N array storing h_{pq}
        Note N is number of orbitals"""

        c_matrix = self.scf_method.mo_coeff

        # one body terms
        one_body_integrals = (
            c_matrix.T @ self.scf_method.get_hcore() @ c_matrix
        )
        return one_body_integrals

    @cached_property
    def _two_body_integrals(self) -> np.ndarray:
        """Get the two electron integrals: An N by N by N by N array storing h_{pqrs}
        Note N is number of orbitals. Note indexing in physist notation!"""
        c_matrix = self.scf_method.mo_coeff
        n_orbs = c_matrix.shape[1]

        two_body_compressed = ao2mo.kernel(self.scf_method.mol, c_matrix)

        # get electron repulsion integrals
        eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

        # Openfermion uses physicist notation whereas pyscf uses chemists
        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
        return two_body_integrals

    @cached_property
    def fermionic_fock_operator(self):
        """
        (Hartree-Fock Hamiltonian!)

        Szabo pg 351
        """
        n_spin_orbs = self.eri_spin_basis.shape[0]

        ib_jb = np.einsum('ibjb->ij', self.eri_spin_basis[:, :self.n_electrons, :, :self.n_electrons])
        ib_bj = np.einsum('ibbj->ij', self.eri_spin_basis[:, :self.n_electrons, :self.n_electrons, :])
        V_hf = ib_jb - ib_bj

        FOCK = FermionOperator((), 0)
        for p in range(n_spin_orbs):
            for q in range(n_spin_orbs):
                h_pq = self.core_h_spin_basis[p, q]
                v_hf_pq = V_hf[p, q]

                coeff = h_pq + v_hf_pq
                if not np.isclose(coeff,0):
                    FOCK += FermionOperator(f'{p}^ {q}', coeff)

        return FOCK

    @property
    def core_h_spin_basis(self):
        """
        Convert spatial-MO basis integrals into spin-MO basis!
        Returns:

        """
        n_spatial_orbs = self.scf_method.mol.nao
        n_spin_orbs = 2 * n_spatial_orbs
        h_core_mo_basis_spin = np.zeros((n_spin_orbs, n_spin_orbs))
        for p in range(n_spatial_orbs):
            for q in range(n_spatial_orbs):
                # populate 1-body terms (must have same spin, otherwise orthogonal)
                ## pg 82 Szabo
                h_core_mo_basis_spin[2 * p, 2 * q] = self._one_body_integrals[p, q]  # spin UP
                h_core_mo_basis_spin[(2 * p + 1), (2 * q + 1)] = self._one_body_integrals[p, q]  # spin DOWN
        return h_core_mo_basis_spin

    @cached_property
    def eri_spin_basis(self):
        n_spatial_orbs = self.scf_method.mol.nao
        n_spin_orbs = 2 * n_spatial_orbs
        ERI_mo_basis_spin = np.zeros((n_spin_orbs, n_spin_orbs, n_spin_orbs, n_spin_orbs))

        two_body_integrals = np.einsum('ijkl -> ikjl', ao2mo.restore(1,  ao2mo.kernel(self.scf_method.mol,
                                                                                      self.scf_method.mo_coeff),
                                                                     self.scf_method.mo_coeff.shape[1]))

        for p in range(n_spatial_orbs):
            for q in range(n_spatial_orbs):
                for r in range(n_spatial_orbs):
                    for s in range(n_spatial_orbs):
                        AO_term = two_body_integrals[p, q, r, s]
                        # up,up,up,up
                        ERI_mo_basis_spin[2 * p, 2 * q, 2 * r, 2 * s] = AO_term
                        # down,down, down, down
                        ERI_mo_basis_spin[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = AO_term

                        ## pg 82 Szabo
                        #  up down up down (physics notation)
                        ERI_mo_basis_spin[2 * p, 2 * q + 1,
                                          2 * r, 2 * s + 1] = AO_term
                        # down up down up  (physics notation)
                        ERI_mo_basis_spin[2 * p + 1, 2 * q,
                                          2 * r + 1, 2 * s] = AO_term

        return ERI_mo_basis_spin

    def get_perturbation_correlation_potential(self):
        """
        V = H_full - H_Hartree-Fock

        Szabo pg 350 and 351

        Returns:
            V_fermionic = perturbation correlation potential

        """
        if self.fermionic_molecular_hamiltonian is None:
            raise ValueError('please generate fermionic molecular H first')

        # commented out as not working due to interaction op - fermionic op
        # V_fermionic = self.fermionic_molecular_hamiltonian - self.fermionic_fock_operator

        n_qu = count_qubits(self.fermionic_molecular_hamiltonian)
        V_fermionic_mat = (get_sparse_operator(self.fermionic_molecular_hamiltonian, n_qubits=n_qu) -
                           get_sparse_operator(self.fermionic_fock_operator, n_qubits=n_qu))
        return V_fermionic_mat

    def build_fermionic_hamiltonian_operator(self, occupied_indices=None, active_indices=None) -> None:
        """Build fermionic Hamiltonian"""

        # nuclear energy
        core_constant = self.scf_method.energy_nuc()

        if active_indices is not None:
            # ACTIVE space reduction!
            (core_constant,
             one_body_integrals,
             two_body_integrals) = get_active_space_integrals(self._one_body_integrals,
                                           self._two_body_integrals,
                                           occupied_indices=occupied_indices,
                                           active_indices=active_indices)
        else:
            one_body_integrals = self._one_body_integrals
            two_body_integrals = self._two_body_integrals

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )

        self.fermionic_molecular_hamiltonian = InteractionOperator(core_constant,
                                                              one_body_coefficients,
                                                              0.5 * two_body_coefficients)

    def manual_fermionic_hamiltonian_operator(self) -> None:
        """Build fermionic Hamiltonian"""

        # nuclear energy
        fermionic_molecular_hamiltonian = FermionOperator((), self.scf_method.energy_nuc())

        for p in range(self.n_qubits):
            for q in range(self.n_qubits):

                h_pq = self.core_h_spin_basis[p, q]
                fermionic_molecular_hamiltonian += FermionOperator(f'{p}^ {q}', h_pq)
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):
                        ## note pyscf:  ⟨01|23⟩ ==> ⟨02|31⟩
                        g_pqrs = self.eri_spin_basis[p, q, r, s]
                        fermionic_molecular_hamiltonian += 0.5 * FermionOperator(f'{p}^ {r}^ {s} {q}', g_pqrs)

        return fermionic_molecular_hamiltonian

    @cached_property
    def fock_spin_mo_basis(self):

        # note this is in chemist notation (look at slices!)

        pm_qm = np.einsum('pmqm->pq', self.eri_spin_basis[:, :self.n_electrons,
                                                               :, :self.n_electrons])
        pm_mq = np.einsum('pmmq->pq', self.eri_spin_basis[:, :self.n_electrons,
                                                               :self.n_electrons, :])

        Fock_spin_mo = self.core_h_spin_basis + (pm_qm - pm_mq)
        return Fock_spin_mo


    def manual_T2_mp2(self) -> FermionOperator:
        """
        Get T2 mp2 fermionic operator

        T2 = 1/4 * t_{ijab} a†_{a} a†_{b} a_{i} a_{j}

        Returns:
            T2_phys (FermionOperator): mp2 fermionic doubles excitation operator

        """
        #TODO fix bg
        warnings.warn('currently note woring as well as taking t2 amps from pyscf object')

        # phys order
        e_orbs_occ = np.diag(self.fock_spin_mo_basis)[:self.n_electrons]

        e_i = e_orbs_occ.reshape(-1, 1, 1, 1)
        e_j = e_orbs_occ.reshape(1, -1, 1, 1)

        e_orbs_vir = np.diag(self.fock_spin_mo_basis)[self.n_electrons:]
        e_a = e_orbs_vir.reshape(1, 1, -1, 1)
        e_b = e_orbs_vir.reshape(1, 1, 1, -1)

        ij_ab = self.eri_spin_basis[:self.n_electrons, :self.n_electrons,
                                     self.n_electrons:, self.n_electrons:]
        ij_ba = np.einsum('ijab -> ijba', ij_ab)

        # mp2 amplitudes
        self.t_ijab_phy = (ij_ba-ij_ab) / (e_i + e_j - e_a - e_b)

        T2_phys = FermionOperator((),0)
        for i in range(self.t_ijab_phy.shape[0]):
            for j in range(self.t_ijab_phy.shape[1]):

                for a in range(self.t_ijab_phy.shape[2]):
                    for b in range(self.t_ijab_phy.shape[3]):

                        t_ijab = self.t_ijab_phy[i, j, a, b]
                        # t_ijab = self.t_ijab_phy[i, j, b, a]

                        # if not np.isclose(t_ijab, 0):
                        virt_a = a + self.n_electrons
                        virt_b = b + self.n_electrons
                        T2_phys += FermionOperator(f'{virt_a}^ {virt_b}^ {i} {j}', t_ijab / 4)

        return T2_phys


    @cached_property
    def hf_fermionic_basis_state(self):
        if self.scf_method.__class__.__name__.find('RHF') != -1:
            n_alpha = n_beta = self.scf_method.mol.nelectron//2
        else:
            n_alpha, n_beta = self.scf_method.nelec
        hf_array = np.zeros(self.n_qubits)
        hf_array[::2] = np.hstack([np.ones(n_alpha), np.zeros(self.n_qubits//2-n_alpha)])
        hf_array[1::2] = np.hstack([np.ones(n_beta), np.zeros(self.n_qubits//2-n_beta)])

        return hf_array.astype(int)


    @cached_property
    def hf_ket(self):
        binary_int_list = 1 << np.arange(self.n_qubits)[::-1]
        hf_ket = np.zeros(2 ** self.n_qubits, dtype=int)
        hf_ket[self.hf_fermionic_basis_state @ binary_int_list] = 1
        return hf_ket


    def mp2_ket(self, pyscf_mp2_t2_amps):
        T2_mp2_mat = get_sparse_operator(self.get_T2_mp2(pyscf_mp2_t2_amps), n_qubits=self.n_qubits)
        mp2_state = T2_mp2_mat @ self.hf_ket
        return mp2_state


    def get_sparse_ham(self):
        if self.fermionic_molecular_hamiltonian is None:
            raise ValueError('need to build operator first')
        return get_sparse_operator(self.fermionic_molecular_hamiltonian)


def get_T2_mp2(pyscf_mp2_t2_amps) -> FermionOperator:
    t2 = spatial2spin(pyscf_mp2_t2_amps)
    no, nv = t2.shape[1:3]
    nmo = no + nv
    double_amps = np.zeros((nmo, nmo, nmo, nmo))
    double_amps[no:, :no, no:, :no] = .25 * t2.transpose(2, 0, 3, 1)

    double_amplitudes_list = []
    double_amplitudes = double_amps
    for i, j, k, l in zip(*double_amplitudes.nonzero()):
        if not np.isclose(double_amplitudes[i, j, k, l], 0):
            double_amplitudes_list.append([[i, j, k, l],
                                           double_amplitudes[i, j, k, l]])

    generator = FermionOperator((), 0)
    # Add double excitations
    for (i, j, k, l), t_ijkl in double_amplitudes_list:
        i, j, k, l = int(i), int(j), int(k), int(l)
        generator += FermionOperator(((i, 1), (j, 0), (k, 1), (l, 0)), t_ijkl)
    #     if anti_hermitian:
    #         generator += FermionOperator(((l, 1), (k, 0), (j, 1), (i, 0)),
    #                                      -t_ijkl)
    return generator


class FermioniCC:
    """ Class for calculating Coupled-Cluster amplitudes 
    and building the corrsponding Fermionic operators
    """
    def __init__(
        self,
        cc_obj: StreamObject,
        ) -> None:
        """
        """
        self.cc_obj = cc_obj
        self.fermionic_cc_operator=None

        self.n_electrons = self.cc_obj.mol.nelectron
        self.n_qubits    = 2*self.cc_obj.mol.nao

    @property
    def _single_amplitudes(self) -> FermionOperator:
        """ Calculate CC amplitudes for single excitations
        """
        t1 = spatial2spin(self.cc_obj.t1, orbspin=self.orbspin)
        no, nv = t1.shape
        nmo = no + nv
        ccsd_single_amps = np.zeros((nmo, nmo))
        ccsd_single_amps[no:,:no] = t1.T

        single_amplitudes_list = []
        for i, j in zip(*ccsd_single_amps.nonzero()):
            single_amplitudes_list.append([[i, j], ccsd_single_amps[i, j]])

        generator_t1 = FermionOperator()
        for (i, j), t_ij in single_amplitudes_list:
            i, j = int(i), int(j)
            generator_t1 += FermionOperator(((i, 1), (j, 0)), t_ij)

        return generator_t1

    @property
    def _double_amplitudes(self) -> FermionOperator:
        """ Calculate CC amplitudes for double excitations
        """
        t2 = spatial2spin(self.cc_obj.t2, orbspin=self.orbspin)
        no, nv = t2.shape[1:3]
        nmo = no + nv
        double_amps = np.zeros((nmo, nmo, nmo, nmo))
        double_amps[no:,:no,no:,:no] = .5 * t2.transpose(2,0,3,1)

        double_amplitudes_list=[]
        double_amplitudes = double_amps
        for i, j, k, l in zip(*double_amplitudes.nonzero()):
            if not np.isclose(double_amplitudes[i, j, k, l], 0):
                double_amplitudes_list.append([[i, j, k, l],
                                            double_amplitudes[i, j, k, l]])
            
        generator_t2 = FermionOperator()
        for (i, j, k, l), t_ijkl in double_amplitudes_list:
            i, j, k, l = int(i), int(j), int(k), int(l)
            generator_t2 += FermionOperator(((i, 1), (j, 0), (k, 1), (l, 0)), t_ijkl)

        return generator_t2

    def build_operator(self, orbspin=None) -> None:
        """ builds the CCSD operator
        """
        self.orbspin = orbspin
        T1 = self._single_amplitudes
        T2 = self._double_amplitudes
        self.fermionic_cc_operator = T1 + 0.5 * T2


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

