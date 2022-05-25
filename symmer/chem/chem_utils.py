from typing import List #,List, Tuple, Union
import os
import numpy as np
from pyscf import gto
from openfermion.chem.pubchem import geometry_from_pubchem
import py3Dmol
from pyscf.tools import cubegen

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
