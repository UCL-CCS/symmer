from openfermion.ops import FermionOperator

def Get_ia_terms(n_electrons, n_orbitals, single_cc_amplitudes=None,  singles_hamiltonian=None,
                               tol_filter_small_terms = None):
    """

    Get ia excitation terms as fermionic creation and annihilation operators for UCCSD.
    ia terms are standard single excitation terms (aka only occupied -> unoccupied transitions allowed)
    (faster and only marginally less accurate.)
    #TODO can add method to get pqrs terms
    #TODO these are all non-degenerate excitations which can possibly non-zero, including nocc->nocc, occ->occ, and spin-flips.
    #TODO EXPENSIVE, but will likely  get a slightly better answer.

    Args:
        n_electrons (int): number of electrons
        n_orbitals (int): number of orbitals
        singles_hamiltonian (numpy.ndarray, optional): h_pq (n_qubits x n_qubits) matrix.
        tol_filter_small_terms (bool, optional):  Whether to filter small terms in Hamiltonian (threshold currently hardcoded)
        single_cc_amplitudes (numpy.ndarray, optional): A 2-dimension array t[a,i] for CCSD single excitation amplitudes
                                where a is virtual index and i is occupied index

    returns:
        Sec_Quant_CC_ia_ops (list): list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator)
        theta_parameters (list): list of theta values (parameterisation of excitation amplitudes)

     e.g.:

     n_electrons=2
     n_orbitals=4
     Sec_Quant_CC_ops, theta_parameters = Get_ia_terms(n_electrons, n_orbitals)

     Sec_Quant_CC_ops=  [
                         -1.0[0 ^ 2] + 1.0[2 ^ 0],            # -(a†0 a2) + (a†2 a0)
                         -1.0[1 ^ 3] + 1.0[3 ^ 1],            # -(a†1 a3) + (a†3 a1)
                        ]
    theta_parameters = [0,0,0]

    """

    Sec_Quant_CC_ia_ops = []  # second quantised single e- CC operators
    theta_parameters_ia = []

    # single_amplitudes and double_amplitudes from Get_CCSD_Amplitudes Hamiltonian function!
    orbitals_index = range(0, n_orbitals)

    alph_occs = [k for k in orbitals_index if k % 2 == 0 and k < n_electrons]  # spin up occupied
    beta_occs = [k for k in orbitals_index if k % 2 == 1 and k < n_electrons]  # spin down UN-occupied
    alph_noccs = [k for k in orbitals_index if k % 2 == 0 and k >= n_electrons]  # spin down occupied
    beta_noccs = [k for k in orbitals_index if k % 2 == 1 and k >= n_electrons]  # spin up UN-occupied


    # SINGLE electron excitation: spin UP transition
    for i in alph_occs:
        for a in alph_noccs:
            if tol_filter_small_terms:
                if abs(singles_hamiltonian[i][a]) > tol_filter_small_terms or abs(
                        singles_hamiltonian[a][i]) > tol_filter_small_terms:
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if single_cc_amplitudes is not None:
                        theta_parameters_ia.append(single_cc_amplitudes[a][i])
                    else:
                        theta_parameters_ia.append(0)

                    Sec_Quant_CC_ia_ops.append(one_elec)
            else:
                # NO filtering
                one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                if single_cc_amplitudes is not None:
                    theta_parameters_ia.append(single_cc_amplitudes[a][i])
                else:
                    theta_parameters_ia.append(0)

                Sec_Quant_CC_ia_ops.append(one_elec)

    # SINGLE electron excitation: spin DOWN transition
    for i in beta_occs:
        for a in beta_noccs:
            if tol_filter_small_terms:
                # uses Hamiltonian to ignore small terms!
                if abs(singles_hamiltonian[i][a]) > tol_filter_small_terms or abs(
                        singles_hamiltonian[a][i]) > tol_filter_small_terms:
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if single_cc_amplitudes is not None:
                        theta_parameters_ia.append(single_cc_amplitudes[a][i])
                    else:
                        theta_parameters_ia.append(0)

                    Sec_Quant_CC_ia_ops.append(one_elec)
            else:
                # NO filtering
                one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                if single_cc_amplitudes is not None:
                    theta_parameters_ia.append(single_cc_amplitudes[a][i])
                else:
                    theta_parameters_ia.append(0)

                Sec_Quant_CC_ia_ops.append(one_elec)

    return Sec_Quant_CC_ia_ops, theta_parameters_ia


def Get_ijab_terms(n_electrons, n_orbitals, double_cc_amplitudes=None, doubles_hamiltonian=None,
                 tol_filter_small_terms=None):
    """

    Get ijab excitation terms as fermionic creation and annihilation operators for UCCSD.
    ijab terms are standard double excitation terms (aka only occupied -> unoccupied transitions allowed)
    (faster and only marginally less accurate.)
    #TODO can add method to get pqrs terms
    #TODO these are all non-degenerate excitations which can possibly non-zero, including nocc->nocc, occ->occ, and spin-flips.
    #TODO EXPENSIVE, but will likely  get a slightly better answer.

    Args:
        n_electrons (int): number of electrons
        n_orbitals (int): number of orbitals
        doubles_hamiltonian (numpy.ndarray, optional): h_pqrs (n_qubits x n_qubits x n_qubits x n_qubits) matrix
        tol_filter_small_terms (bool, optional):  Whether to filter small terms in Hamiltonian (threshold currently hardcoded)
        double_cc_amplitudes (numpy.ndarray, optional): A 4-dimension array t[a,i,b,j] for CCSD double excitation amplitudes
                                                        where a, b are virtual indices and i, j are occupied indices.

    returns:
        Sec_Quant_CC_ijab_ops (list): list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator)
        theta_parameters (list): list of theta values (parameterisation of excitation amplitudes)

     e.g.:

     n_electrons=2
     n_orbitals=4
     Sec_Quant_CC_ops, theta_parameters = Get_ijab_terms(n_electrons, n_orbitals)

     Sec_Quant_CC_ops=  [
                            -1.0[0 ^ 1 ^ 2 3] + 1.0 [3^ 2^ 1 0]  # -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                        ]
    theta_parameters = [0]
    """


    # single_amplitudes and double_amplitudes from Get_CCSD_Amplitudes Hamiltonian function!
    orbitals_index = range(0, n_orbitals)

    alph_occs = [k for k in orbitals_index if k % 2 == 0 and k < n_electrons]  # spin up occupied
    beta_occs = [k for k in orbitals_index if k % 2 == 1 and k < n_electrons]  # spin down UN-occupied
    alph_noccs = [k for k in orbitals_index if k % 2 == 0 and k >= n_electrons]  # spin down occupied
    beta_noccs = [k for k in orbitals_index if k % 2 == 1 and k >= n_electrons]  # spin up UN-occupied

    Sec_Quant_CC_ijab_ops = []  # second quantised two e- CC operators
    theta_parameters_ijab = []

    # DOUBLE excitation: UP + UP
    for i in alph_occs:
        for j in [k for k in alph_occs if k > i]:
            for a in alph_noccs:
                for b in [k for k in alph_noccs if k > a]:

                    if tol_filter_small_terms:
                        # uses Hamiltonian to ignore small terms!
                        if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
                                doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                            else:
                                theta_parameters_ijab.append(0)
                        Sec_Quant_CC_ijab_ops.append(two_elec)
                    else:
                        # NO filtering
                        two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                   FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                        if double_cc_amplitudes is not None:
                            theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                        else:
                            theta_parameters_ijab.append(0)

                        Sec_Quant_CC_ijab_ops.append(two_elec)

    # DOUBLE excitation: DOWN + DOWN
    for i in beta_occs:
        for j in [k for k in beta_occs if k > i]:
            for a in beta_noccs:
                for b in [k for k in beta_noccs if k > a]:

                    if tol_filter_small_terms:
                        # uses Hamiltonian to ignore small terms!
                        if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
                                doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                            else:
                                theta_parameters_ijab.append(0)
                        Sec_Quant_CC_ijab_ops.append(two_elec)
                    else:
                        # NO filtering
                        two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                   FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                        if double_cc_amplitudes is not None:
                            theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                        else:
                            theta_parameters_ijab.append(0)

                        Sec_Quant_CC_ijab_ops.append(two_elec)

    # DOUBLE excitation: up + DOWN
    for i in alph_occs:
        for j in [k for k in beta_occs if k > i]:
            for a in alph_noccs:
                for b in [k for k in beta_noccs if k > a]:

                    if tol_filter_small_terms:
                        # uses Hamiltonian to ignore small terms!
                        if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
                                doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                            else:
                                theta_parameters_ijab.append(0)
                        Sec_Quant_CC_ijab_ops.append(two_elec)
                    else:
                        # NO filtering
                        two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                   FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                        if double_cc_amplitudes is not None:
                            theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                        else:
                            theta_parameters_ijab.append(0)

                        Sec_Quant_CC_ijab_ops.append(two_elec)

    return Sec_Quant_CC_ijab_ops, theta_parameters_ijab

