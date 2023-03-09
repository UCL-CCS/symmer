import numpy as np
from symmer.symplectic import QuantumState, PauliwordOp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from symmer.chemistry.utils import build_bk_matrix
from symmer.chemistry.utils import get_excitations

class CI_from_qubit_H():
    """
    class to get CI Hamiltonian from a qubit H (note fermoinic function is FASTER, see fermionic_ham.py)
    """
    def __init__(self, qubitH: PauliwordOp, hf_array_number: np.array):
        self.H = qubitH
        self.hf_array = hf_array_number # HF in number basis (not BK / parity / ...)
        self.n_electrons = np.sum(self.hf_array)

    # def _gen_single_excitations_general(self):
    #     # does not maintain spin symmetry
    #     single_dets = []
    #     single_excitations = []
    #     for i in range(self.n_electrons):
    #         for a in range(self.n_electrons, self.H.n_qubits):
    #             single_excitations.append((i, a))
    #
    #             det = self.hf_array.copy()
    #             det[[i ,a]] = det[[a ,i]]
    #             single_dets.append(det)
    #     return single_dets  # , single_excitations

    def _gen_single_excitations_singlet(self):
        # only allow spin allowed transitions
        single_dets = []
        single_excitations = []
        for i in range(self.n_electrons//2):
            alpha_occ = 2*i
            beta_occ = 2*i + 1
            for a in range(self.n_electrons, self.H.n_qubits, 2):
                alpha_virt = a
                ## spin up
                single_excitations.append((alpha_occ, alpha_virt))
                det = self.hf_array.copy()
                det[[alpha_occ , alpha_virt]] = det[[alpha_virt ,alpha_occ]]
                single_dets.append(det)

                beta_virt = a + 1
                ## spin down
                single_excitations.append((alpha_occ, beta_virt))
                det = self.hf_array.copy()
                det[[beta_occ , beta_virt]] = det[[beta_virt ,beta_occ]]
                single_dets.append(det)

        return single_dets  # , single_excitations

    # def _gen_double_excitations(self):
    #     double_dets = []
    #     double_excitations = []
    #     for i in range(self.n_electrons):
    #         for j in range( i +1, self.n_electrons):
    #             for a in range(self.n_electrons, self.H.n_qubits):
    #                 for b in range( a +1, self.H.n_qubits):
    #                     double_excitations.append((i ,j, a ,b))
    #
    #                     det = self.hf_array.copy()
    #                     det[[i ,a]] = det[[a ,i]]
    #                     det[[j ,b]] = det[[b ,j]]
    #                     double_dets.append(det)
    #     return double_dets  # , double_excitations

    def _gen_double_excitations_singlet(self):
        double_dets = []
        double_excitations = []
        for i in range(self.n_electrons):
            for j in range( i +1, self.n_electrons):
                for a in range(self.n_electrons, self.H.n_qubits):
                    for b in range( a +1, self.H.n_qubits):
                        # if (i%2== 0 and a%2  == 0) and (j%2== 1 and b%2 == 1):
                        double_excitations.append((i ,j, a ,b))
                        det = self.hf_array.copy()
                        det[[i ,a]] = det[[a ,i]]
                        det[[j ,b]] = det[[b ,j]]
                        double_dets.append(det)
        return double_dets  # , double_excitations

    def _gen_single_double_excitations(self):
        double_dets = []
        double_excitations = []
        single_dets = []
        single_excitations = []
        for i in range(self.n_electrons):
            for a in range(self.n_electrons, self.H.n_qubits):
                single_excitations.append((i, a))

                det = self.hf_array.copy()
                det[[i ,a]] = det[[a ,i]]
                single_dets.append(det)
                for j in range( i +1, self.n_electrons):
                    for b in range( a +1, self.H.n_qubits):
                        double_excitations.append((i ,j, a ,b))

                        det = self.hf_array.copy()
                        det[[i ,a]] = det[[a ,i]]
                        det[[j ,b]] = det[[b ,j]]
                        double_dets.append(det)

        # return single_dets, single_excitations, double_dets, double_excitations
        return [*single_dets, *double_dets]

    def _perform_CI_JW(self, det_list):

        data =[]
        row = []
        col = []
        for i, det_i in enumerate(det_list):
            for det_j in det_list:
                index_i = int(''.join(det_i.astype(str)), 2)
                q_state_i = QuantumState(np.array(det_i).reshape([1, -1]), [1])

                index_j = int(''.join(det_j.astype(str)), 2)
                q_state_j = QuantumState(np.array(det_j).astype(int).reshape([1, -1]), [1])

                mat_ij_element = q_state_j.dagger * self.H * q_state_i
                data.append(mat_ij_element)
                row.append(index_i)
                col.append(index_j)

        #                 # ij == ji ! (symmetry)
        #                 data.append(mat_ij_element)
        #                 row.append(index_j)
        #                 col.append(index_i)

        H_CI_JW = csr_matrix((data, (row, col)), shape=(2 ** self.H.n_qubits, 2 ** self.H.n_qubits))
        #         H_CI_JW = csr_matrix((data, (row, col)))
        return H_CI_JW

    def _perform_CI_BK(self, det_list):
        raise NotImplemented('not working correctly')
        bk_conv_mat = build_bk_matrix(self.H.n_qubits)

        # convert dets to BK representation
        det_list_new = [(bk_conv_mat @ det.reshape([-1,1])).reshape([-1])%2 for det in det_list]
        return self._perform_CI_JW(det_list_new)

    def perform_CI(self, method='CISD', encoding='JW', S=0):

        if method == 'CISD':
            HF, singles, doubles = get_excitations(self.hf_array, self.H.n_qubits, S=S, excitations='sd')
            det_list = [HF, *singles, *doubles]
        elif method == 'CIS':
            # does NOT include HF array
            # det_list = self._gen_single_excitations_singlet()
            HF, singles, doubles = get_excitations(self.hf_array, self.H.n_qubits, S=S, excitations='s')
            det_list = [HF, *singles]
        elif method == 'CID':
            # include HF array
            HF, singles, doubles = get_excitations(self.hf_array, self.H.n_qubits, S=S, excitations='d')
            det_list = [HF, *doubles]
        else:
            raise ValueError(f'unknown / not implemented CI method: {method}')

        if encoding == 'JW':
            H_CI = self._perform_CI_JW(det_list)
        elif encoding == 'BK':
            H_CI = self._perform_CI_BK(det_list)
        #         print(H_CI.shape)
        E_CI, vec_CI = eigsh(H_CI, which='SA', k=1)
        ci_qstate = QuantumState.from_array(vec_CI)
        del vec_CI
        return H_CI, E_CI, ci_qstate

