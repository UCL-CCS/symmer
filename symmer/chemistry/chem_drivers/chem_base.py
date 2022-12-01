from typing import Optional, Union, List
from scipy.sparse import csr_matrix
import numpy as np
from cached_property import cached_property
from openfermion import get_sparse_operator, FermionOperator, count_qubits
import warnings
from symmer.chemistry.fermionic_ham import get_sign


class ClosedShellChemistry:
    """Class to generate MO spin basis given some C matrix (orthogonal MOs in atomic orbital basis)
    """

    def __init__(
            self,
            C_matrix: np.array,
            electron_repulsion_integrals_spatial_MO: np.array,
            nuclear_energy: float,
            n_alpha: int,
            n_beta: int,
            electron_repulsion_integral_type: str,
            H_core_ao: np.array,
            aux_information=dict(),
    ) -> None:
        """

        Args:
            C_matrix (np.array): matrix of MO coefficients (columns are orthogonal MOs). Note must either be
                                 C_
            electron_repulsion_integrals_spatial_MO (np.array): eri in molecular orbitals basis (NOT in atomic orbital basis!)
            n_alpha (int): number of spin up electrons
            n_beta (int): number of spin down electrons
            calc_type (str): open or closed shell
            H_core_ao (np.array): core Hamiltonian (T+K) in atomic orbital basis
            electron_repulsion_integral_type (str): chemist or physist indexing (class can interconvert!)
            aux_information (dict): dictionary containing any extra information about calculation
        """

        # closed shell (alpha and beta e- treated the same)
        self.C_matrix = C_matrix

        if electron_repulsion_integral_type == 'chem':
            self.chem_eri_spatial = electron_repulsion_integrals_spatial_MO
            self.phys_eri_spatial = None
        elif electron_repulsion_integral_type == 'phys':
            self.chem_eri_spatial = None
            self.phys_eri_spatial = electron_repulsion_integrals_spatial_MO

        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.H_core_ao = H_core_ao
        self.nuclear_energy = nuclear_energy

        self.n_electrons = n_alpha + n_beta
        self.n_qubits = 2 * self.C_matrix.shape[0]
        self.aux_information = aux_information

        assert self.chem_eri.shape == (C_matrix.shape[0],
                                       C_matrix.shape[0],
                                       C_matrix.shape[0],
                                       C_matrix.shape[0]), 'electron repulsion integrals (spatial) not correct shape'

    @cached_property
    def hcore_spatial_MO(self) -> np.ndarray:
        """Get the one electron integrals: An N by N array storing h_{pq}
        Note N is number of orbitals"""
        hcore_ij_alpha = self.C_matrix.conj().T @ self.H_core_ao @ self.C_matrix

        return hcore_ij_alpha

    def _hcore_spatial_to_spin(self, hcore_spatial):
        n_spatial_orbs = hcore_spatial.shape[0]*2
        n_spin_orbs = 2 * n_spatial_orbs
        h_core_mo_basis_spin = np.zeros((n_spin_orbs, n_spin_orbs))
        for p in range(n_spatial_orbs):
            for q in range(n_spatial_orbs):
                # populate 1-body terms (must have same spin, otherwise orthogonal)
                ## pg 82 Szabo
                h_core_mo_basis_spin[2 * p, 2 * q] = hcore_spatial[p, q]  # spin UP
                h_core_mo_basis_spin[(2 * p + 1), (2 * q + 1)] = hcore_spatial[p, q]  # spin DOWN
        return h_core_mo_basis_spin

    @cached_property
    def hcore_spin_MO(self):
        """
        Convert spatial-MO basis integrals into spin-MO basis!
        Returns:

        """
        h_core_mo_basis_spin = self._hcore_spatial_to_spin(self.hcore_mo_spatial)
        return h_core_mo_basis_spin

    @cached_property
    def phys_eri_spatial_MO(self) -> np.ndarray:
        """Get the two electron integrals: An N by N by N by N array storing h_{pqrs}
        Note N is number of orbitals. Note indexing in physist notation!"""
        if self.phys_eri is None:
            return np.asarray(self.chem_eri.transpose(0, 2, 3, 1), order="C")
        else:
            return self.phys_eri

    @cached_property
    def chem_eri_spatial_MO(self) -> np.ndarray:
        """Get the two electron integrals: An N by N by N by N array storing h_{pqrs}
        Note N is number of orbitals. Note indexing in physist notation!"""
        if self.chem_eri_spatial is None:
            return np.asarray(self.phys_eri_spatial.transpose(0, 2, 3, 1), order="C")
        else:
            return self.chem_eri_spatial

    def _chem_eri_spatial_to_spin(self, chem_eri_spatial, zero_threshold=1e-12):
        n_spatial_orbs = self.chem_eri_spatial.shape[0]
        n_spin_orbs = 2 * n_spatial_orbs
        ERI_MO_basis_spin_chemist = np.zeros((n_spin_orbs, n_spin_orbs, n_spin_orbs, n_spin_orbs))

        for p in range(n_spatial_orbs):
            for q in range(n_spatial_orbs):
                for r in range(n_spatial_orbs):
                    for s in range(n_spatial_orbs):
                        AO_term = chem_eri_spatial[p, q, r, s]

                        if abs(AO_term)>zero_threshold:
                            # only populate if not zero!

                            # up,up,up,up
                            ERI_MO_basis_spin_chemist[2 * p, 2 * q, 2 * r, 2 * s] = AO_term
                            # down,down, down, down
                            ERI_MO_basis_spin_chemist[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = AO_term

                            ## pg 82 Szabo
                            #  up up down down (chemist notation)
                            ERI_MO_basis_spin_chemist[2*p,2*q,
                                              2*r+1,2*s+1] = AO_term
                            # down, down, up, up  (chemist notation)
                            ERI_MO_basis_spin_chemist[2*p+1,2*q+1,
                                                      2*r,2*s] = AO_term
        return ERI_MO_basis_spin_chemist

    @cached_property
    def chem_eri_spin_MO(self, zero_threshold=1e-12):
        ERI_MO_basis_spin_chemist = self._chem_eri_spatial_to_spin(self.chem_eri_spatial_MO,
                                                                   zero_threshold=zero_threshold)
        return ERI_MO_basis_spin_chemist

    @cached_property
    def phys_eri_spin_MO(self, zero_threshold=1e-12):
        n_spatial_orbs = self.phys_eri_spatial_MO.shape[0]
        n_spin_orbs = 2 * n_spatial_orbs
        ERI_mo_basis_spin_phys = np.zeros((n_spin_orbs, n_spin_orbs, n_spin_orbs, n_spin_orbs))

        for p in range(n_spatial_orbs):
            for q in range(n_spatial_orbs):
                for r in range(n_spatial_orbs):
                    for s in range(n_spatial_orbs):
                        # phys term
                        AO_term = self.phys_eri_spatial_MO[p, q, r, s]
                        if abs(AO_term)>zero_threshold:

                            # up,up,up,up
                            ERI_mo_basis_spin_phys[2 * p, 2 * q, 2 * r, 2 * s] = AO_term
                            # down,down, down, down
                            ERI_mo_basis_spin_phys[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = AO_term

                            ## pg 82 Szabo
                            #  up down up down (physics notation)
                            ERI_mo_basis_spin_phys[2 * p, 2 * q + 1,
                                              2 * r, 2 * s + 1] = AO_term
                            # down up down up  (physics notation)
                            ERI_mo_basis_spin_phys[2 * p + 1, 2 * q,
                                              2 * r + 1, 2 * s] = AO_term

        return ERI_mo_basis_spin_phys

    def active_space(self,
                      active_spatial_MOs_inds,
                      spatial_occupied_inds):
        """
        Note this is the restricted case (spatial orbitals contain 2e-)

        Args:
            active_spatial_MOs_inds:
            spatial_occupied_inds:

        Returns:

        """
        active_spatial_MOs_inds = np.asarray(active_spatial_MOs_inds)
        spatial_occupied_inds = np.asarray(spatial_occupied_inds)

        occupied_and_frozen = np.union1d(active_spatial_MOs_inds, spatial_occupied_inds)

        frozen_energy = 0
        for i in occupied_and_frozen:
            frozen_energy+= self.hcore_spin_MO[i,i]
            for b in occupied_and_frozen:

                # Szabo page 135 eqution 3.128
                frozen_energy+= 2*self.chem_eri_spatial_MO[i,i,b,b] - self.chem_eri_spatial_MO[i,b,b,i]


        # slice active parts only!
        hcore_spatial_MO_active_space = self.hcore_spatial_MO[np.ix_(active_spatial_MOs_inds, active_spatial_MOs_inds)]
        chem_eri_spatial_MO_active_space = self.chem_eri_spatial_MO[np.ix_(active_spatial_MOs_inds, active_spatial_MOs_inds,
                                                                  active_spatial_MOs_inds, active_spatial_MOs_inds)]

        hcore_spin_MO_active_space = self._hcore_spatial_to_spin(hcore_spatial_MO_active_space)
        chem_eri_spin_MO_active_space = self._chem_eri_spatial_to_spin(chem_eri_spatial_MO_active_space)


        return frozen_energy, hcore_spin_MO_active_space, chem_eri_spin_MO_active_space

    @cached_property
    def fermionic_fock_operator(self):
        """
        (Hartree-Fock Hamiltonian!)

        Szabo pg 351
        """
        n_spin_orbs = self.chem_eri.shape[0]

        ib_jb = np.einsum('ibjb->ij', self.chem_eri_spin_MO[:, :self.n_electrons, :, :self.n_electrons])
        ib_bj = np.einsum('ibbj->ij', self.chem_eri_spin_MO[:, :self.n_electrons, :self.n_electrons, :])
        V_hf = ib_jb - ib_bj

        FOCK = FermionOperator((), 0)
        for p in range(n_spin_orbs):
            for q in range(n_spin_orbs):
                h_pq = self.hcore_spin_MO[p, q]
                v_hf_pq = V_hf[p, q]

                coeff = h_pq + v_hf_pq
                if not np.isclose(coeff, 0):
                    FOCK += FermionOperator(f'{p}^ {q}', coeff)

        return FOCK

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

    def build_fermionic_hamiltonian_operator(self, spatial_occupied_inds=None, active_spatial_MOs_inds=None) -> None:
        """Build fermionic Hamiltonian"""

        # nuclear energy
        core_constant = self.nuclear_energy

        if spatial_occupied_inds is not None:
            # ACTIVE space reduction!
            (frozen_energy,
             h_pq,
             g_pqrs) = self.active_space(active_spatial_MOs_inds,
                                                     spatial_occupied_inds)

            core_constant+=frozen_energy # <-- adding frozen term
        else:
            h_pq = self.hcore_spin_MO
            g_pqrs = self.chem_eri_spin_MO

        # nuclear energy ( and frozen energy if active space included)
        fermionic_molecular_hamiltonian = FermionOperator((), core_constant)

        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                fermionic_molecular_hamiltonian += FermionOperator(f'{p}^ {q}',
                                                                   h_pq[p,q])
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):
                        fermionic_molecular_hamiltonian += 0.5 * FermionOperator(f'{p}^ {r}^ {s} {q}',
                                                                                 g_pqrs[p, q, r, s])

        return fermionic_molecular_hamiltonian

    @cached_property
    def fock_spin_mo_basis(self):

        # note this is in chemist notation (look at slices!)

        pm_qm = np.einsum('pmqm->pq', self.chem_eri_spin_MO[:, :self.n_electrons,
                                      :, :self.n_electrons])
        pm_mq = np.einsum('pmmq->pq', self.chem_eri_spin_MO[:, :self.n_electrons,
                                      :self.n_electrons, :])

        Fock_spin_mo = self.hcore_spin_MO + (pm_qm - pm_mq)
        return Fock_spin_mo

    def manual_T2_mp2(self) -> FermionOperator:
        """
        Get T2 mp2 fermionic operator

        T2 = 1/4 * t_{ijab} a†_{a} a†_{b} a_{i} a_{j}

        Returns:
            T2_phys (FermionOperator): mp2 fermionic doubles excitation operator

        """
        # TODO fix bg
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
        self.t_ijab_phy = (ij_ba - ij_ab) / (e_i + e_j - e_a - e_b)

        T2_phys = FermionOperator((), 0)
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
        hf_array = np.zeros(self.n_qubits)
        hf_array[::2] = np.hstack([np.ones(self.n_alpha), np.zeros(self.n_qubits // 2 - self.n_alpha)])
        hf_array[1::2] = np.hstack([np.ones(self.n_beta), np.zeros(self.n_qubits // 2 - self.n_beta)])

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

    def get_cisd_fermionic(self):
        """

        Note does NOT INCLUDE NUCLEAR ENERGY (maybe add constant to diagonal elements in for loop)

        See page 6 of https://iopscience.iop.org/article/10.1088/2058-9565/aa9463/pdf
        for how to determine matrix element of FCI H given particular Slater determinants

        """
        double_dets = []
        double_excitations = []
        single_dets = []
        single_excitations = []
        for i in range(self.n_electrons):
            for a in range(self.n_electrons, self.n_qubits):
                single_excitations.append((i, a))

                det = self.hf_fermionic_basis_state.copy()
                det[[i, a]] = det[[a, i]]
                single_dets.append(det)
                for j in range(i + 1, self.n_electrons):
                    for b in range(a + 1, self.n_qubits):
                        double_excitations.append((i, j, a, b))

                        det = self.hf_fermionic_basis_state.copy()
                        det[[i, a]] = det[[a, i]]
                        det[[j, b]] = det[[b, j]]
                        double_dets.append(det)

        allowed_dets = [self.hf_fermionic_basis_state, *single_dets, *double_dets]

        data = []
        row = []
        col = []
        for det_i in allowed_dets:
            for det_j in allowed_dets:

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

        H_ci = csr_matrix((data, (row, col)), shape=(2 ** (self.n_qubits), 2 ** (self.n_qubits)))
        #
        return H_ci

class OpenShellChemistry:
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
        Ca: np.array,
        Cb: np.array,
        eri_aaaa,
        eri_bbbb,
        eri_aabb,
        eri_bbaa,

        n_alpha: int,
        n_beta: int,
        calc_type:str,
        electron_repulsion_integral_type: str,
        H_core_ao: np.array,
        aux_information = dict(),

    ) -> None:
        """

        Args:
            C_matrix (np.array):
        """
        raise NotImplementedError('need to write openshell cose')


class FermionicOp:
    """
    TODO: build class that holds all operator info

    .number_op
    .spinz_op
    .spin2_op
    .second_quant_H_op
    .MP2_ansatz_op
    .CCSD_ansatz_op
    ...etc...

    """

    def __init__(
        self,
    ) -> None:
        """

        Args:
            C_matrix (np.array):
        """
        raise NotImplementedError('need to write openshell cose')

    def molecular_H(self):
        pass
    def number_op(self):
        pass
    def spinz_op(self):
        pass
    def spinz2_op(self):
        pass
    def hf_fermionic_array(self):
        pass


