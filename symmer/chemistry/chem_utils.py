from symmer.symplectic import array_to_QuantumState
from typing import List, Tuple
import os
import numpy as np
import scipy as sp
from pyscf import gto
from openfermion.chem.pubchem import geometry_from_pubchem
import py3Dmol
from pyscf.tools import cubegen
from openfermion import FermionOperator, count_qubits
from openfermion.transforms import jordan_wigner, bravyi_kitaev, parity_code
from symmer.utils import QubitOperator_to_dict
from symmer.symplectic import PauliwordOp

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


def exact_gs_energy(sparse_matrix, initial_guess=None, n_particles=None, number_operator=None, n_eigs=6) -> Tuple[float, np.array]:
    """ Return the ground state energy and corresponding ground statevector for the input operator
    Specifying a particle number will restrict to eigenvectors of that Hamming weight
    """
    # Note the eigenvectors are stored column-wise so need to transpose
    if sparse_matrix.shape[0] > 2**5:
        eigvals, eigvecs = sp.sparse.linalg.eigsh(
            sparse_matrix,k=n_eigs,v0=initial_guess,which='SA',maxiter=1e7
        )
    else:
        # for small matrices the dense representation can be more efficient than sparse!
        eigvals, eigvecs = np.linalg.eigh(sparse_matrix.toarray())
    
    # order the eigenvalues by increasing size
    order = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    if n_particles is None:
        # if no particle number is specified then return the smallest eigenvalue
        return eigvals[0], eigvecs.T[0]
    else:
        assert(number_operator is not None), 'Must specify the number operator.'
        # otherwise, search through the first n_eig eigenvalues and check the Hamming weight
        # of the the corresponding eigenvector - return the first match with n_particles
        for evl, evc in zip(eigvals, eigvecs.T):
            psi = array_to_QuantumState(evc).cleanup(zero_threshold=1e-5)
            assert(~np.any(number_operator.X_block)), 'Number operator not diagonal'
            expval_n_particle = 0
            for Z_symp, Z_coeff in zip(number_operator.Z_block, number_operator.coeff_vec):
                sign = (-1) ** np.einsum('ij->i', 
                    np.bitwise_and(
                        Z_symp, psi.state_matrix
                    )
                )
                expval_n_particle += Z_coeff * np.sum(sign * np.square(abs(psi.coeff_vector)))
            if round(expval_n_particle) == n_particles:
                return evl, evc
        # if a solution is not found within the first n_eig eigenvalues then error
        raise RuntimeError('No eigenvector of the correct particle number was identified - try increasing n_eigs.')


def get_fermionic_number_operator(N_qubits: int) -> FermionOperator:
    """

    Args:
        N_qubits(int): number of qubits (or number of molecular spin orbitals)

    Returns:
        N_op (FermionOperator): number operator
    """

    N_op = FermionOperator()
    for spin_orb_ind in range(N_qubits):
        N_op += FermionOperator(f'{spin_orb_ind}^ {spin_orb_ind}', 1)

    return N_op


def get_fermionic_up_down_parity_operators(N_qubits: int) -> Tuple[FermionOperator, FermionOperator]:
    """
    note order is assumed to be spin up, spin down, spin up, spin down ... etc

    each op is built as product of parities of individual sites!
    https://arxiv.org/pdf/1008.4346.pdf

    Args:
        N_qubits (int): number of qubits (or molecular spin orbitals)

    Returns:
        parity_up (FermionOperator): parity operator of spin up fermions
        parity_down (FermionOperator): parity operator of spin down fermions
    """

    parity_up = FermionOperator('', 1)
    for spin_up_ind in np.arange(0, N_qubits, 2):
        parity_up *= FermionOperator('', 1) - 2 * FermionOperator(f'{spin_up_ind}^ {spin_up_ind}', 1)  #

    parity_down = FermionOperator('', 1)
    for spin_down_ind in np.arange(1, N_qubits, 2):
        parity_down *= FermionOperator('', 1) - 2 * FermionOperator(f'{spin_down_ind}^ {spin_down_ind}', 1)

    return parity_up, parity_down


def fermion_to_qubit_operator(Fermionic_operator: FermionOperator,
                              qubit_mapping_str: str,
                              N_qubits: int = None):
    """
    Function to convert from fermion operators to qubit operators.
    Note see openfermion.transforms for different fermion to qubit mappings

    Args:
        Fermionic_operator(FermionOperator): any fermionic operator (openfermion)
        qubit_mapping_str (str): fermion to qubit mapping
        N_qubits (int): number of qubits (or spin orbitals)

    Returns:
        qubit_operator (PauliwordOp): qubit operator of fermonic operator (under certain mapping)
    """
    fermonic_to_qubit_map = {'jordan_wigner': jordan_wigner,
                             'bravyi_kitaev': bravyi_kitaev,
                             'parity_code': parity_code}

    if qubit_mapping_str.lower() not in fermonic_to_qubit_map.keys():
        print(f'valid qubit mappings : {list(fermonic_to_qubit_map.keys())}')
        raise ValueError(f'unknown qubit mapping: {qubit_mapping_str}')

    mapping = fermonic_to_qubit_map[qubit_mapping_str.lower()]
    qubit_operator = mapping(Fermionic_operator)

    if N_qubits is None:
        N_qubits = count_qubits(qubit_operator)

    q_op_dict = QubitOperator_to_dict(qubit_operator, N_qubits)

    # want to return PauliWordOp (but results in circular import!)
    ## aka PauliWordOp base class imports utils and so import here causes problems.
    return PauliwordOp.from_dictionary(q_op_dict)
