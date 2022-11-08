from typing import List, Tuple
import os
import numpy as np
from pyscf import gto
from openfermion.chem.pubchem import geometry_from_pubchem
import py3Dmol
from pyscf.tools import cubegen
from openfermion import FermionOperator, count_qubits, hermitian_conjugated
from openfermion.transforms import jordan_wigner, bravyi_kitaev#, parity_code
from symmer.symplectic.utils import QubitOperator_to_dict
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


def get_fermionic_spin_operators(N_qubits: int) -> Tuple[FermionOperator, FermionOperator]:
    """ https://aip.scitation.org/doi/pdf/10.1063/1.5110682 eq 35-40
    
    The multiplicity of a given state is 2<Sz> + 1, but this is only valid for singlets
    The reason why is consider a triplet solution (has two underpaired e-)
    therefore (up,up), (up, down) or (down, down) giving Sz of -1,0,+1
    
    When measuring Sz on superposition, combinitionos of -1,0,1 possible leading to meaningless info
    
    As Sz can only be 0 in singlet states this is still useful.
    
    Likewise for doublets (aka one unpaired electron), we can have (up) or (down)
    thus Sz can be -0.5 or +0.5 and in a superposition state measuring Sz will lead to weird results
    
    NOTE: measuring the invidual kets of a superposition state we must get an allowed value of Sz
    that depends on the multiplicity
    (e.g. for triplets: -1, 0, +1  AND for doublets: -0.5 or +0.5)
    
    
    In general, given a spin quantum number s, we may observe 
    the values {s, s-1, ..., -s+1, -s} == 2S+1 possible values
    
    Therefore, projecting out Sz=0 ensures you CANNOT get a singlet solution
    (e.g. for triplets: now we can only get -1, +1 solutions (as 0 part projected out))
    
    HOWEVER, projecting onto Sz=0 does NOT! ensure you cannot get a singlet solution
    (e.g. you can get triplet terms with Sz=0... == [up, down] unpaired combo)

    """
    Sz = FermionOperator()
    S_plus = FermionOperator()

    for p in range(N_qubits//2):
        # Sz term
        ap = FermionOperator(f'{2*p}^ {2*p}')
        bp = FermionOperator(f'{2*p+1}^ {2*p+1}')
        Sz += (ap-bp)
        # S_plus term
        S_plus += FermionOperator(f'{2*p}^ {2*p+1}')
    Sz /= 2

    S_minus = hermitian_conjugated(S_plus)
    S2 = S_plus * S_minus + Sz + Sz ** 2
    
    return S2, Sz


def build_bk_matrix(n_qubits):
    """ Implemented from https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24969
    """
    assert n_qubits > 0
    B = np.array([[1]])
    
    n_q_power2 = int(np.ceil(np.log2(n_qubits)))
    for _ in range(n_q_power2):
        zero = np.zeros_like(B)
        zero_ones = zero.copy()
        zero_ones[-1,:] =1 
        
        B = np.block([ 
            [ B,         zero ],
            [ zero_ones, B    ]    
        ])
    
    return B[:n_qubits, :n_qubits]


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
    fermonic_to_qubit_map = {
        'jordan_wigner': jordan_wigner,
        'bravyi_kitaev': bravyi_kitaev,
        #'parity_code': parity_code
    }

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


def get_parity_operators_JW(n_qubits):
    """ Assumes alternating up/down spin orbitals
    """
    spin_up_parity_Z_block = np.arange(1, n_qubits+1) % 2
    spin_up_parity_op = PauliwordOp(np.hstack(
        [np.zeros_like(spin_up_parity_Z_block), spin_up_parity_Z_block]), [1])

    spin_down_parity_Z_block = np.arange(n_qubits) % 2
    spin_down_parity_op = PauliwordOp(np.hstack(
        [np.zeros_like(spin_down_parity_Z_block), spin_down_parity_Z_block]), [1])

    return spin_up_parity_op, spin_down_parity_op 


def get_parity_operators_BK(n_qubits):
    """ Assumes alternating up/down spin orbitals
    """
    spin_up_parity_Z_block = np.arange(1, n_qubits+1) % 2
    spin_up_parity_op = PauliwordOp(np.hstack(
        [np.zeros_like(spin_up_parity_Z_block), spin_up_parity_Z_block]), [1])

    parity_mat = build_bk_matrix(n_qubits)
    temp_array = parity_mat.copy()
    argmax_list = []
    counter = 0
    while counter<n_qubits:
        new_index = np.argmax(np.sum(temp_array, axis=1))
        if argmax_list == []:
            old = 0
        else:
            old = argmax_list[-1]+1
        argmax_list.append(old+new_index)
        temp_array = temp_array[new_index+1:]
        if len(temp_array)==0:
            break
        counter+=1
    full_parity_str = ['I']*n_qubits
    for i in argmax_list:
        full_parity_str[i] = 'Z'
    
    full_parity_op = PauliwordOp.from_list([full_parity_str])
    spin_down_parity_op = full_parity_op * spin_up_parity_op

    return spin_up_parity_op, spin_down_parity_op 
