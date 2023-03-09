from typing import Optional, Union
from pathlib import Path
import os
from scipy.sparse import csr_matrix
import numpy as np
from scipy.special import comb
from cached_property import cached_property
from openfermion import InteractionOperator, get_sparse_operator, FermionOperator, count_qubits
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import get_active_space_integrals
from pyscf import ao2mo, gto, scf, mp, ci, cc, fci
from pyscf.lib import StreamObject
from symmer.chemistry.utils import get_excitations
from symmer.symplectic import QuantumState, PauliwordOp
from symmer.utils import exact_gs_energy
from pyscf.cc.addons import spatial2spin
import warnings
from openfermion.transforms.opconversions.jordan_wigner import _jordan_wigner_interaction_op
from openfermion.transforms.opconversions.bravyi_kitaev import _bravyi_kitaev_interaction_operator

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
    def __init__(self, scf_obj, index_ordering='physicist') -> None:
        self.scf_obj = scf_obj
        self.c_matrix = scf_obj.mo_coeff
        self.n_spatial_orbs = scf_obj.mol.nao
        self.index_ordering = index_ordering
        self.n_spin_orbs = self.n_qubits = 2 * self.n_spatial_orbs
        self.n_electrons = scf_obj.mol.nelectron
        self.interaction_operator = InteractionOperator(
                constant = self.scf_obj.energy_nuc(), 
                one_body_tensor = self.core_h_spin_basis, 
                two_body_tensor = self.eri_spin_basis*.5
            )
        self.fermionic_molecular_hamiltonian=None
        self.qubit_molecular_hamiltonian=None
        
    def get_ci_fermionic(self, method:str='CISD',S:float=0):
        """

        Note does NOT INCLUDE NUCLEAR ENERGY (maybe add constant to diagonal elements in for loop)

        See page 6 of https://iopscience.iop.org/article/10.1088/2058-9565/aa9463/pdf
        for how to determine matrix element of FCI H given particular Slater determinants

        # TODO can add higher order excitations to get_excitations function for higher order excitations

        Args:
            method (str): CIS, CID, CISD
            S (float): The S in multiplicity (2S+1)
        """

        if method == 'CISD':
            HF, singles, doubles = get_excitations(self.hf_fermionic_basis_state,
                                                   self.n_qubits, S=S, excitations='sd')
            det_list = [HF, *singles, *doubles]
        elif method == 'CIS':
            # does NOT include HF array
            # det_list = self._gen_single_excitations_singlet()
            HF, singles, doubles = get_excitations(self.hf_fermionic_basis_state,
                                                   self.n_qubits, S=S, excitations='s')
            det_list = [HF, *singles]
        elif method == 'CID':
            # include HF array
            HF, singles, doubles = get_excitations(self.hf_fermionic_basis_state,
                                                   self.n_qubits, S=S, excitations='d')
            det_list = [HF, *doubles]
        else:
            raise ValueError(f'unknown / not implemented CI method: {method}')

        data = []
        row = []
        col = []
        for det_i in det_list:
            for det_j in det_list:

                bit_diff = np.logical_xor(det_i, det_j)
                n_diff = int(sum(bit_diff))

                if n_diff > 4:
                    pass
                else:
                    index_i = int(''.join(det_i.astype(str)), 2)
                    index_j = int(''.join(det_j.astype(str)), 2)

                    mat_element = 0
                    if n_diff == 0:
                        # <i | H | i>
                        occ_inds_i = np.where(det_i)[0]
                        for i in occ_inds_i:
                            mat_element += self.core_h_spin_basis[i, i]

                        for ind1, i in enumerate(occ_inds_i[:-1]):
                            for ind2 in range(ind1 + 1, len(occ_inds_i)):
                                j = occ_inds_i[ind2]
                                mat_element += (self.eri_spin_basis[i, j, i, j] - self.eri_spin_basis[i, j, j, i])
                        sign = 1

                    elif n_diff == 2:
                        # order matters!
                        k = np.logical_and(det_i, bit_diff).nonzero()[0]
                        l = np.logical_and(det_j, bit_diff).nonzero()[0]
                        mat_element += self.core_h_spin_basis[k, l]

                        common_bits = np.where(np.logical_and(det_i, det_j))[0]
                        for i in common_bits:
                            mat_element += (self.eri_spin_basis[k, i, l, i]
                                            - self.eri_spin_basis[k, i, i, l])

                        sign = get_sign(det_i, det_j, bit_diff)

                    elif n_diff == 4:
                        ij = np.logical_and(det_i, bit_diff).nonzero()[0]
                        kl = np.logical_and(det_j, bit_diff).nonzero()[0]
                        i, j = ij[0], ij[1]
                        k, l = kl[0], kl[1]
                        mat_element += (self.eri_spin_basis[i, j, k, l] - self.eri_spin_basis[i, j, l, k])
                        sign = get_sign(det_i, det_j, bit_diff)

                    data.append(float(mat_element * sign))
                    row.append(index_i)
                    col.append(index_j)

        # TODO: build smaller FCI matrix (aka on subspace of allowed determinants rather than 2^n by 2^n !!!
        H_ci = csr_matrix((data, (row, col)), shape=(2 ** (self.n_qubits), 2 ** (self.n_qubits)))
        return H_ci

    def get_fermionic_CI_ansatz(self, 
            method:str='CISD',
            S:float=0, 
            zero_threshold:float=1e-10
        ):
        """
        """
        H_CI_matrix = self.get_ci_fermionic(S=S, method=method)
        e_ci, psi = exact_gs_energy(H_CI_matrix)
        total_CI_energy = e_ci + self.scf_obj.energy_nuc()

        CI_ferm=FermionOperator()

        for det, coeff in zip(psi.state_matrix, psi.state_op.coeff_vec):

            bit_diff = np.logical_xor(self.hf_fermionic_basis_state, det)
            indices = bit_diff.nonzero()[0]
            
            if len(indices)==0:
                new_term = FermionOperator('', coeff)
            elif len(indices)==2:
                sign = get_sign(self.hf_fermionic_basis_state, det, bit_diff)
                ferm_string = f'{indices[1]}^ {indices[0]}' 
                new_term = FermionOperator(ferm_string, coeff*sign)
            elif len(indices)==4:
                sign = get_sign(self.hf_fermionic_basis_state, det, bit_diff)
                ferm_string_1 = f'{indices[2]}^ {indices[3]}^ {indices[1]} {indices[0]}'
                new_term = FermionOperator(ferm_string_1, coeff*sign)
            else:
                raise NotImplementedError('Excitations above doubles not currently implemented.')
            # add extra elif statements for higher order excitations
            
            CI_ferm += new_term

        return CI_ferm, total_CI_energy, psi


def sort_det(array):
    """
    sort list of numbers counting number of swaps (bubble sort)
    """

    arr = np.asarray(array)
    n_sites = arr.shape[0]
    sign_dict = {0: +1, 1: -1}
    # Traverse through all array elements
    swap_counter = 0
    for i in range(n_sites):
        swapped = False
        for j in range(0, n_sites - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                swap_counter += 1

        if swapped == False:
            break

    #     return np.abs(arr.tolist()), sign_dict[swap_counter%2]
    return sign_dict[swap_counter % 2]


def get_sign(i_det, j_det, bit_differ):
    # first put unique part to start of list (count if swap needed!)
    # then order RHS that contains common elements!

    nonzero_i = i_det.nonzero()[0]
    nonzero_j = j_det.nonzero()[0]

    unique = bit_differ.nonzero()[0]
    i_unique = np.intersect1d(nonzero_i, unique)
    j_unique = np.intersect1d(nonzero_j, unique)

    count_i = 0
    count_j = 0
    for ind, unique_i in enumerate(i_unique):
        swap_ind_i = np.where(unique_i == nonzero_i)[0]

        unique_j = j_unique[ind]
        swap_ind_j = np.where(unique_j == nonzero_j)[0]

        # swap
        if ind != swap_ind_i:
            count_i += 1
            nonzero_i[ind], nonzero_i[swap_ind_i] = nonzero_i[swap_ind_i], nonzero_i[ind]
        if ind != swap_ind_j:
            count_j += 1
            nonzero_j[ind], nonzero_j[swap_ind_j] = nonzero_j[swap_ind_j], nonzero_j[ind]

    sign_i = sort_det(nonzero_i[len(i_unique):])
    sign_j = sort_det(nonzero_j[len(j_unique):])
    return sign_i * sign_j * (-1) ** (count_i + count_j)