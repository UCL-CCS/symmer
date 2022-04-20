from typing import Optional, List #,List, Tuple, Union
from pathlib import Path
import os
import numpy as np
from cached_property import cached_property
from openfermion import InteractionOperator, get_sparse_operator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import get_active_space_integrals
from pyscf import ao2mo, gto, scf, mp, ci, cc, fci
from pyscf.lib import StreamObject
from openfermion.chem.pubchem import geometry_from_pubchem
import py3Dmol
from pyscf.tools import cubegen

class FermionicHamilt:
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

    @property
    def _one_body_integrals(self) -> np.ndarray:
        """Get the one electron integrals: An N by N array storing h_{pq}
        Note N is number of orbitals"""

        c_matrix = self.scf_method.mo_coeff

        # one body terms
        one_body_integrals = (
            c_matrix.T @ self.scf_method.get_hcore() @ c_matrix
        )
        return one_body_integrals

    @property
    def _two_body_integrals(self) -> np.ndarray:
        """Get the two electron integrals: An N by N by N by N array storing h_{pqrs}
        Note N is number of orbitals"""
        c_matrix = self.scf_method.mo_coeff
        n_orbs = c_matrix.shape[1]

        two_body_compressed = ao2mo.kernel(self.scf_method.mol, c_matrix)

        # get electron repulsion integrals
        eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

        # Openfermion uses physicist notation whereas pyscf uses chemists
        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
        return two_body_integrals

    def build_operator(self, occupied_indices=None, active_indices=None) -> None:
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


    @property
    def hf_comp_basis_state(self):
        hf_comp_basis_state = np.zeros(self.n_qubits, dtype=int)
        hf_comp_basis_state[:self.n_electrons] = 1
        return hf_comp_basis_state


    @property
    def hf_ket(self):
        binary_int_list = 1 << np.arange(self.n_qubits)[::-1]
        hf_ket = np.zeros(2 ** self.n_qubits, dtype=int)
        hf_ket[self.hf_comp_basis_state @ binary_int_list] = 1
        return hf_ket

    def get_sparse_ham(self):
        if self.fermionic_molecular_hamiltonian is None:
            raise ValueError('need to build operator first')
        return get_sparse_operator(self.fermionic_molecular_hamiltonian)


def xyz_from_pubchem(molecule_name):
    geometry_pubchem = geometry_from_pubchem(molecule_name, structure="3d")

    if geometry_pubchem is None:
        geometry_pubchem = geometry_from_pubchem(molecule_name, structure="2d")
        if geometry_pubchem is None:
            raise ValueError(
                f"""Could not find geometry of {molecule_name} on PubChem...
                     make sure molecule input is a correct path to an xyz file or real molecule
                                """)

    n_atoms = len(geometry_pubchem)
    xyz_file = f"{n_atoms}"
    xyz_file += "\n \n"
    for atom, xyz in geometry_pubchem:
            xyz_file += f"{atom}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n"

    return xyz_file


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

    def _build_mol(self) -> gto.mole:
        """Function to build PySCF molecule.

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        if os.path.exists(self.geometry):
            # geometry is an xyz file
            full_mol = gto.Mole(
                atom=self.geometry, basis=self.basis, charge=self.charge, unit=self.unit
            ).build()
        else:
            # geometry is raw xyz string
            full_mol = gto.Mole(
                atom=self.geometry[3:],
                basis=self.basis,
                charge=self.charge,
                unit=self.unit,
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


        if self.run_ccsd:
            self.pyscf_ccsd = cc.CCSD(self.pyscf_hf)
            self.pyscf_ccsd.verbose = self.pyscf_print_level
            # self.pyscf_ccsd.diis = False
            self.pyscf_ccsd.run()

        # Run FCI.
        if self.run_fci:
            self.pyscf_fci = fci.FCI(self.pyscf_hf.mol, self.pyscf_hf.mo_coeff)
            self.pyscf_fci.verbose = 0
            self.pyscf_fci.kernel()


def Draw_molecule(
    xyz_string: str, width: int = 400, height: int = 400, style: str = "sphere"
) -> py3Dmol.view:
    """Draw molecule from xyz string.

    Note if molecule has unrealistic bonds, then style should be sphere. Otherwise stick style can be used
    which shows bonds.

    TODO: more styles at http://3dmol.csb.pitt.edu/doc/$3Dmol.GLViewer.html

    Args:
        xyz_string (str): xyz string of molecule
        width (int): width of image
        height (int): Height of image
        style (str): py3Dmol style ('sphere' or 'stick')

    Returns:
        view (py3dmol.view object). Run view.show() method to print molecule.
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_string, "xyz")
    if style == "sphere":
        view.setStyle({'sphere': {"radius": 0.2}})
    elif style == "stick":
        view.setStyle({'stick': {}})
    else:
        raise ValueError(f"unknown py3dmol style: {style}")

    view.zoomTo()
    return view


def Draw_cube_orbital(
    PySCF_mol_obj: gto.Mole,
    xyz_string: str,
    C_matrix: np.ndarray,
    index_list: List[int],
    width: int = 400,
    height: int = 400,
    style: str = "sphere",
) -> List:
    """Draw orbials given a C_matrix and xyz string of molecule.

    This function writes orbitals to temporary cube files then deletes them.
    For standard use the C_matrix input should be C_matrix optimized by a self consistent field (SCF) run.

    Note if molecule has unrealistic bonds, then style should be set to sphere

    Args:
        PySCF_mol_obj (pyscf.mol): PySCF mol object. Required for pyscf.tools.cubegen function
        xyz_string (str): xyz string of molecule
        C_matrix (np.array): Numpy array of molecular orbitals (columns are MO).
        index_list (List): List of MO indices to plot
        width (int): width of image
        height (int): Height of image
        style (str): py3Dmol style ('sphere' or 'stick')

    Returns:
        plotted_orbitals (List): List of plotted orbitals (py3Dmol.view) ordered the same way as in index_list
    """

    if not set(index_list).issubset(set(range(C_matrix.shape[1]))):
        raise ValueError(
            "list of MO indices to plot is outside of C_matrix column indices"
        )

    plotted_orbitals = []
    for index in index_list:
        File_name = f"temp_MO_orbital_index{index}.cube"
        cubegen.orbital(PySCF_mol_obj, File_name, C_matrix[:, index])

        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz_string, "xyz")
        if style == "sphere":
            view.setStyle({"sphere": {"radius": 0.2}})
        elif style == "stick":
            view.setStyle({"stick": {}})
        else:
            raise ValueError(f"unknown py3dmol style: {style}")

        with open(File_name, "r") as f:
            view.addVolumetricData(
                f.read(), "cube", {"isoval": -0.02, "color": "red", "opacity": 0.75}
            )
        with open(File_name, "r") as f2:
            view.addVolumetricData(
                f2.read(), "cube", {"isoval": 0.02, "color": "blue", "opacity": 0.75}
            )

        plotted_orbitals.append(view.zoomTo())
        os.remove(File_name)  # delete file once orbital is drawn

    return plotted_orbitals
