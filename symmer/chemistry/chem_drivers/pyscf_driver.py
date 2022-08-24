from typing import Optional, List #,List, Tuple, Union
from pathlib import Path
import os
import numpy as np
from scipy.special import comb
from cached_property import cached_property
from openfermion import InteractionOperator, get_sparse_operator, FermionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import get_active_space_integrals
from pyscf import ao2mo, gto, scf, mp, ci, cc, fci
from pyscf.lib import StreamObject
from openfermion.chem.pubchem import geometry_from_pubchem
import py3Dmol
from pyscf.tools import cubegen
from pyscf.cc.addons import spatial2spin
import warnings


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

        run_mp2: Optional[bool] = False,
        run_cisd: Optional[bool] = False,
        run_ccsd: Optional[bool] = False,
        run_fci: Optional[bool] = False,
    ):

        self.geometry = geometry
        self.basis = basis.lower()
        self.convergence = convergence
        self.charge = charge
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.savefile = savefile
        self.unit = unit
        self.max_hf_cycles = max_hf_cycles

        self.run_mp2  = run_mp2
        self.run_cisd = run_cisd
        self.run_ccsd = run_ccsd
        self.run_fci = run_fci
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
                spin=self.spin
            ).build()
        else:
            # geometry is raw xyz string
            full_mol = gto.Mole(
                atom=self.geometry[3:],
                basis=self.basis,
                charge=self.charge,
                unit=self.unit,
                spin=self.spin,
            ).build()
        return full_mol


    @cached_property
    def pyscf_hf(self) -> StreamObject:
        """Run Hartree-Fock calculation."""
        mol_full = self._build_mol()
        # run Hartree-Fock

        if mol_full.spin:
            global_hf = scf.ROHF(mol_full)
        else:
            global_hf = scf.RHF(mol_full)

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