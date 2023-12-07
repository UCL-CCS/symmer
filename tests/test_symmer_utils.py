import numpy as np
import py3Dmol
from openfermion import QubitOperator

from symmer.operators import PauliwordOp, QuantumState
from symmer.utils import (
    Draw_molecule,
    exact_gs_energy,
    get_sparse_matrix_large_pauliwordop,
    gram_schmidt_from_quantum_state,
    matrix_allclose,
    product_list,
    random_anitcomm_2n_1_PauliwordOp,
    tensor_list,
)

H2_sto3g = {
    "qubit_encoding": "jordan_wigner",
    "unit": "angstrom",
    "geometry": "2\n \nH\t0\t0\t0\nH\t0\t0\t0.74",
    "basis": "STO-3G",
    "charge": 0,
    "spin": 0,
    "hf_array": [1, 1, 0, 0],
    "hf_method": "pyscf.scf.hf_symm.SymAdaptedRHF",
    "n_particles": {"total": 2, "alpha": 1, "beta": 1},
    "n_qubits": 4,
    "convergence_threshold": 1e-06,
    "point_group": {"groupname": "Dooh", "topgroup": "Dooh"},
    "calculated_properties": {
        "HF": {"energy": -1.1167593073964255, "converged": True},
        "MP2": {"energy": -1.1298973809859585, "converged": True},
        "CCSD": {"energy": -1.13728399861044, "converged": True},
        "FCI": {"energy": -1.137283834488502, "converged": True},
    },
    "auxiliary_operators": {
        "number_operator": {
            "IIII": (2.0, 0.0),
            "IIIZ": (-0.5, 0.0),
            "IIZI": (-0.5, 0.0),
            "IZII": (-0.5, 0.0),
            "ZIII": (-0.5, 0.0),
        },
        "S^2_operator": {
            "IIII": (0.75, 0.0),
            "IIIZ": (0.5, 0.0),
            "IIZI": (-0.5, 0.0),
            "IIZZ": (-0.375, 0.0),
            "IZII": (0.5, 0.0),
            "IZIZ": (0.125, 0.0),
            "IZZI": (-0.125, 0.0),
            "ZIII": (-0.5, 0.0),
            "ZIIZ": (-0.125, 0.0),
            "ZIZI": (0.125, 0.0),
            "ZZII": (-0.375, 0.0),
            "XXXX": (0.125, 0.0),
            "XXYY": (0.125, 0.0),
            "XYXY": (0.125, 0.0),
            "XYYX": (-0.125, 0.0),
            "YXXY": (-0.125, 0.0),
            "YXYX": (0.125, 0.0),
            "YYXX": (0.125, 0.0),
            "YYYY": (0.125, 0.0),
        },
        "Sz_operator": {
            "IIIZ": (0.25, 0.0),
            "IIZI": (-0.25, 0.0),
            "IZII": (0.25, 0.0),
            "ZIII": (-0.25, 0.0),
        },
        "alpha_parity_operator": {"ZIZI": (1.0, 0.0)},
        "beta_parity_operator": {"IZIZ": (1.0, 0.0)},
        "MP2_operator": {
            "XXXX": (-0.004531358676614097, 0.0),
            "XXXY": (0.0, 0.004531358676614097),
            "XXYX": (0.0, 0.004531358676614097),
            "XXYY": (0.004531358676614097, 0.0),
            "XYXX": (0.0, -0.004531358676614097),
            "XYXY": (-0.004531358676614097, 0.0),
            "XYYX": (-0.004531358676614097, 0.0),
            "XYYY": (0.0, 0.004531358676614097),
            "YXXX": (0.0, -0.004531358676614097),
            "YXXY": (-0.004531358676614097, 0.0),
            "YXYX": (-0.004531358676614097, 0.0),
            "YXYY": (0.0, 0.004531358676614097),
            "YYXX": (0.004531358676614097, 0.0),
            "YYXY": (0.0, -0.004531358676614097),
            "YYYX": (0.0, -0.004531358676614097),
            "YYYY": (-0.004531358676614097, 0.0),
        },
        "CCSD_operator": {
            "XXXX": (-0.007079023951543804, 0.0),
            "XXXY": (0.0, 0.007079023951543804),
            "XXYX": (0.0, 0.007079023951543804),
            "XXYY": (0.007079023951543804, 0.0),
            "XYXX": (0.0, -0.007079023951543804),
            "XYXY": (-0.007079023951543804, 0.0),
            "XYYX": (-0.007079023951543804, 0.0),
            "XYYY": (0.0, 0.007079023951543804),
            "YXXX": (0.0, -0.007079023951543804),
            "YXXY": (-0.007079023951543804, 0.0),
            "YXYX": (-0.007079023951543804, 0.0),
            "YXYY": (0.0, 0.007079023951543804),
            "YYXX": (0.007079023951543804, 0.0),
            "YYXY": (0.0, -0.007079023951543804),
            "YYYX": (0.0, -0.007079023951543804),
            "YYYY": (-0.007079023951543804, 0.0),
        },
    },
    "H_dict": {
        "IIII": (-0.09706626816763123 + 0j),
        "IIIZ": (-0.22343153690813441 + 0j),
        "IIZI": (-0.22343153690813441 + 0j),
        "IIZZ": (0.17441287612261588 + 0j),
        "IZII": (0.17141282644776915 + 0j),
        "IZIZ": (0.12062523483390411 + 0j),
        "IZZI": (0.1659278503377034 + 0j),
        "ZIII": (0.17141282644776912 + 0j),
        "ZIIZ": (0.1659278503377034 + 0j),
        "ZIZI": (0.12062523483390411 + 0j),
        "ZZII": (0.16868898170361207 + 0j),
        "XXYY": (-0.04530261550379927 + 0j),
        "XYYX": (0.04530261550379927 + 0j),
        "YXXY": (0.04530261550379927 + 0j),
        "YYXX": (-0.04530261550379927 + 0j),
    },
}

He3_plus = {
    "qubit_encoding": "jordan_wigner",
    "unit": "angstrom",
    "geometry": "3\n \nH\t0\t0\t0\nH\t0\t0\t0.74\nH\t0\t0\t1.48",
    "basis": "STO-3G",
    "charge": 1,
    "spin": 0,
    "hf_array": [1, 1, 0, 0, 0, 0],
    "hf_method": "pyscf.scf.hf_symm.SymAdaptedRHF",
    "n_particles": {"total": 2, "alpha": 1, "beta": 1},
    "n_qubits": 6,
    "convergence_threshold": 1e-06,
    "point_group": {"groupname": "Dooh", "topgroup": "Dooh"},
    "calculated_properties": {
        "HF": {"energy": -1.189999028637302, "converged": True},
        "MP2": {"energy": -1.206775423813649, "converged": True},
        "CCSD": {"energy": -1.214628907244379, "converged": True},
        "FCI": {"energy": -1.214628846262647, "converged": True},
    },
    "auxiliary_operators": {
        "number_operator": {
            "IIIIII": (3.0, 0.0),
            "IIIIIZ": (-0.5, 0.0),
            "IIIIZI": (-0.5, 0.0),
            "IIIZII": (-0.5, 0.0),
            "IIZIII": (-0.5, 0.0),
            "IZIIII": (-0.5, 0.0),
            "ZIIIII": (-0.5, 0.0),
        },
        "S^2_operator": {
            "IIIIII": (1.125, 0.0),
            "IIIIIZ": (0.5, 0.0),
            "IIIIZI": (-0.5, 0.0),
            "IIIIZZ": (-0.375, 0.0),
            "IIIZII": (0.5, 0.0),
            "IIIZIZ": (0.125, 0.0),
            "IIIZZI": (-0.125, 0.0),
            "IIZIII": (-0.5, 0.0),
            "IIZIIZ": (-0.125, 0.0),
            "IIZIZI": (0.125, 0.0),
            "IIZZII": (-0.375, 0.0),
            "IZIIII": (0.5, 0.0),
            "IZIIIZ": (0.125, 0.0),
            "IZIIZI": (-0.125, 0.0),
            "IZIZII": (0.125, 0.0),
            "IZZIII": (-0.125, 0.0),
            "ZIIIII": (-0.5, 0.0),
            "ZIIIIZ": (-0.125, 0.0),
            "ZIIIZI": (0.125, 0.0),
            "ZIIZII": (-0.125, 0.0),
            "ZIZIII": (0.125, 0.0),
            "ZZIIII": (-0.375, 0.0),
            "IIXXXX": (0.125, 0.0),
            "IIXXYY": (0.125, 0.0),
            "IIXYXY": (0.125, 0.0),
            "IIXYYX": (-0.125, 0.0),
            "IIYXXY": (-0.125, 0.0),
            "IIYXYX": (0.125, 0.0),
            "IIYYXX": (0.125, 0.0),
            "IIYYYY": (0.125, 0.0),
            "XXIIXX": (0.125, 0.0),
            "XXIIYY": (0.125, 0.0),
            "XYIIXY": (0.125, 0.0),
            "XYIIYX": (-0.125, 0.0),
            "YXIIXY": (-0.125, 0.0),
            "YXIIYX": (0.125, 0.0),
            "YYIIXX": (0.125, 0.0),
            "YYIIYY": (0.125, 0.0),
            "XXXXII": (0.125, 0.0),
            "XXYYII": (0.125, 0.0),
            "XYXYII": (0.125, 0.0),
            "XYYXII": (-0.125, 0.0),
            "YXXYII": (-0.125, 0.0),
            "YXYXII": (0.125, 0.0),
            "YYXXII": (0.125, 0.0),
            "YYYYII": (0.125, 0.0),
        },
        "Sz_operator": {
            "IIIIIZ": (0.25, 0.0),
            "IIIIZI": (-0.25, 0.0),
            "IIIZII": (0.25, 0.0),
            "IIZIII": (-0.25, 0.0),
            "IZIIII": (0.25, 0.0),
            "ZIIIII": (-0.25, 0.0),
        },
        "alpha_parity_operator": {"ZIZIZI": (1.0, 0.0)},
        "beta_parity_operator": {"IZIZIZ": (1.0, 0.0)},
        "MP2_operator": {
            "XXIIXX": (-0.0022135760877754116, 0.0),
            "XXIIXY": (0.0, 0.0022135760877754116),
            "XXIIYX": (0.0, 0.0022135760877754116),
            "XXIIYY": (0.0022135760877754116, 0.0),
            "XYIIXX": (0.0, -0.0022135760877754116),
            "XYIIXY": (-0.0022135760877754116, 0.0),
            "XYIIYX": (-0.0022135760877754116, 0.0),
            "XYIIYY": (0.0, 0.0022135760877754116),
            "YXIIXX": (0.0, -0.0022135760877754116),
            "YXIIXY": (-0.0022135760877754116, 0.0),
            "YXIIYX": (-0.0022135760877754116, 0.0),
            "YXIIYY": (0.0, 0.0022135760877754116),
            "YYIIXX": (0.0022135760877754116, 0.0),
            "YYIIXY": (0.0, -0.0022135760877754116),
            "YYIIYX": (0.0, -0.0022135760877754116),
            "YYIIYY": (-0.0022135760877754116, 0.0),
            "XXXXII": (-0.005258398458007929, 0.0),
            "XXXYII": (0.0, 0.005258398458007929),
            "XXYXII": (0.0, 0.005258398458007929),
            "XXYYII": (0.005258398458007929, 0.0),
            "XYXXII": (0.0, -0.005258398458007929),
            "XYXYII": (-0.005258398458007929, 0.0),
            "XYYXII": (-0.005258398458007929, 0.0),
            "XYYYII": (0.0, 0.005258398458007929),
            "YXXXII": (0.0, -0.005258398458007929),
            "YXXYII": (-0.005258398458007929, 0.0),
            "YXYXII": (-0.005258398458007929, 0.0),
            "YXYYII": (0.0, 0.005258398458007929),
            "YYXXII": (0.005258398458007929, 0.0),
            "YYXYII": (0.0, -0.005258398458007929),
            "YYYXII": (0.0, -0.005258398458007929),
            "YYYYII": (-0.005258398458007929, 0.0),
        },
        "CCSD_operator": {
            "IXZZZX": (-0.0031353836158880214, 0.0),
            "IXZZZY": (0.0, 0.0031353836158880214),
            "IYZZZX": (0.0, -0.0031353836158880214),
            "IYZZZY": (-0.0031353836158880214, 0.0),
            "XZZZXI": (-0.0031353836158880214, 0.0),
            "XZZZYI": (0.0, 0.0031353836158880214),
            "YZZZXI": (0.0, -0.0031353836158880214),
            "YZZZYI": (-0.0031353836158880214, 0.0),
            "XXIIXX": (-0.0025494280723394984, 0.0),
            "XXIIXY": (0.0, 0.0025494280723394984),
            "XXIIYX": (0.0, 0.0025494280723394984),
            "XXIIYY": (0.0025494280723394984, 0.0),
            "XYIIXX": (0.0, -0.0025494280723394984),
            "XYIIXY": (-0.0025494280723394984, 0.0),
            "XYIIYX": (-0.0025494280723394984, 0.0),
            "XYIIYY": (0.0, 0.0025494280723394984),
            "YXIIXX": (0.0, -0.0025494280723394984),
            "YXIIXY": (-0.0025494280723394984, 0.0),
            "YXIIYX": (-0.0025494280723394984, 0.0),
            "YXIIYY": (0.0, 0.0025494280723394984),
            "YYIIXX": (0.0025494280723394984, 0.0),
            "YYIIXY": (0.0, -0.0025494280723394984),
            "YYIIYX": (0.0, -0.0025494280723394984),
            "YYIIYY": (-0.0025494280723394984, 0.0),
            "XXXXII": (-0.008347686124399265, 0.0),
            "XXXYII": (0.0, 0.008347686124399265),
            "XXYXII": (0.0, 0.008347686124399265),
            "XXYYII": (0.008347686124399265, 0.0),
            "XYXXII": (0.0, -0.008347686124399265),
            "XYXYII": (-0.008347686124399265, 0.0),
            "XYYXII": (-0.008347686124399265, 0.0),
            "XYYYII": (0.0, 0.008347686124399265),
            "YXXXII": (0.0, -0.008347686124399265),
            "YXXYII": (-0.008347686124399265, 0.0),
            "YXYXII": (-0.008347686124399265, 0.0),
            "YXYYII": (0.0, 0.008347686124399265),
            "YYXXII": (0.008347686124399265, 0.0),
            "YYXYII": (0.0, -0.008347686124399265),
            "YYYXII": (0.0, -0.008347686124399265),
            "YYYYII": (-0.008347686124399265, 0.0),
        },
    },
    "H_dict": {
        "IIIIII": (0.24747487013571695 + 0j),
        "IIIIIZ": (-0.460046379318107 + 0j),
        "IIIIZI": (-0.460046379318107 + 0j),
        "IIIIZZ": (0.17501414371183724 + 0j),
        "IIIZII": (-0.008325684680054832 + 0j),
        "IIIZIZ": (0.10794008354812167 + 0j),
        "IIIZZI": (0.14596700512786398 + 0j),
        "IIZIII": (-0.008325684680054846 + 0j),
        "IIZIIZ": (0.14596700512786398 + 0j),
        "IIZIZI": (0.10794008354812167 + 0j),
        "IIZZII": (0.1464490594631947 + 0j),
        "IZIIII": (0.21618381471527337 + 0j),
        "IZIIIZ": (0.12892083226845963 + 0j),
        "IZIIZI": (0.16103166954501724 + 0j),
        "IZIZII": (0.10098551626926347 + 0j),
        "IZZIII": (0.13731813628698764 + 0j),
        "ZIIIII": (0.21618381471527337 + 0j),
        "ZIIIIZ": (0.16103166954501724 + 0j),
        "ZIIIZI": (0.12892083226845963 + 0j),
        "ZIIZII": (0.13731813628698764 + 0j),
        "ZIZIII": (0.10098551626926347 + 0j),
        "ZZIIII": (0.15887278686630402 + 0j),
        "IIXXYY": (-0.03802692157974233 + 0j),
        "IIXYYX": (0.03802692157974233 + 0j),
        "IIYXXY": (0.03802692157974233 + 0j),
        "IIYYXX": (-0.03802692157974233 + 0j),
        "IXIZZX": (0.0035462180491050393 + 0j),
        "IXZIZX": (-0.029756134618662153 + 0j),
        "IXZZIX": (-0.024481114159709064 + 0j),
        "IXZZZX": (0.02373733926074963 + 0j),
        "IYIZZY": (0.0035462180491050393 + 0j),
        "IYZIZY": (-0.029756134618662153 + 0j),
        "IYZZIY": (-0.024481114159709064 + 0j),
        "IYZZZY": (0.02373733926074963 + 0j),
        "ZXZZZX": (-0.026953621359169865 + 0j),
        "ZYZZZY": (-0.026953621359169865 + 0j),
        "IXXYYI": (0.03330235266776719 + 0j),
        "IXYYXI": (-0.03330235266776719 + 0j),
        "IYXXYI": (-0.03330235266776719 + 0j),
        "IYYXXI": (0.03330235266776719 + 0j),
        "XIZZXI": (-0.026953621359169868 + 0j),
        "XZIZXI": (-0.029756134618662153 + 0j),
        "XZZIXI": (0.0035462180491050393 + 0j),
        "XZZZXI": (0.023737339260749633 + 0j),
        "XZZZXZ": (-0.024481114159709064 + 0j),
        "YIZZYI": (-0.026953621359169868 + 0j),
        "YZIZYI": (-0.029756134618662153 + 0j),
        "YZZIYI": (0.0035462180491050393 + 0j),
        "YZZZYI": (0.023737339260749633 + 0j),
        "YZZZYZ": (-0.024481114159709064 + 0j),
        "XZXXZX": (-0.03330235266776719 + 0j),
        "XZXYZY": (-0.03330235266776719 + 0j),
        "YZYXZX": (-0.03330235266776719 + 0j),
        "YZYYZY": (-0.03330235266776719 + 0j),
        "XXIIYY": (-0.032110837276557606 + 0j),
        "XYIIYX": (0.032110837276557606 + 0j),
        "YXIIXY": (0.032110837276557606 + 0j),
        "YYIIXX": (-0.032110837276557606 + 0j),
        "XXYYII": (-0.036332620017724165 + 0j),
        "XYYXII": (0.036332620017724165 + 0j),
        "YXXYII": (0.036332620017724165 + 0j),
        "YYXXII": (-0.036332620017724165 + 0j),
    },
}


def test_exact_gs_energy_H2():
    H = PauliwordOp.from_dictionary(H2_sto3g["H_dict"])
    E_ref = H2_sto3g["calculated_properties"]["FCI"]["energy"]

    gs_energy, gs_state = exact_gs_energy(H.to_sparse_matrix)
    assert np.isclose(gs_energy, E_ref), "reference energy does NOT match true gs"

    assert isinstance(gs_state, QuantumState)
    assert np.isclose(gs_state.dagger * H * gs_state, E_ref)


def test_exact_gs_energy_H3_plus():
    H = PauliwordOp.from_dictionary(He3_plus["H_dict"])
    E_ref = He3_plus["calculated_properties"]["FCI"]["energy"]

    num_operator = PauliwordOp.from_dictionary(
        He3_plus["auxiliary_operators"]["number_operator"]
    )
    n_part = He3_plus["n_particles"]["total"]

    gs_energy, gs_state = exact_gs_energy(
        H.to_sparse_matrix, number_operator=num_operator, n_particles=n_part
    )

    assert np.isclose(gs_energy, E_ref), "reference energy does NOT match true gs"
    assert isinstance(gs_state, QuantumState)
    assert np.isclose(gs_state.dagger * H * gs_state, E_ref)


def test_random_anitcomm_2n_1_PauliwordOp_method():
    n_qubits = 5
    complex_coeff = False
    apply_clifford = False

    AC = random_anitcomm_2n_1_PauliwordOp(
        n_qubits, apply_clifford=apply_clifford, complex_coeff=complex_coeff
    )

    # underlying method manually coded:
    base = "Z" * n_qubits
    I_term = "I" * n_qubits
    P_list = [base]
    for i in range(n_qubits):
        # Z_term
        P_list.append(base[:i] + "X" + I_term[i + 1 :])
        # Y_term
        P_list.append(base[:i] + "Y" + I_term[i + 1 :])

    P_anticomm = PauliwordOp.from_list(P_list)
    AC.coeff_vec = np.ones(AC.n_terms)

    output = P_anticomm - AC

    assert output.n_terms == 0
    assert np.allclose(output.coeff_vec, np.zeros_like(output.coeff_vec))


def test_random_anitcomm_2n_1_PauliwordOp_real_with_clifford():
    n_qubits = 4
    complex_coeff = False
    apply_clifford = True

    AC = random_anitcomm_2n_1_PauliwordOp(
        n_qubits, apply_clifford=apply_clifford, complex_coeff=complex_coeff
    )

    assert AC.n_terms == 2 * n_qubits + 1
    anti_comm_check = AC.adjacency_matrix.astype(int) - np.eye(
        AC.adjacency_matrix.shape[0]
    )
    assert (
        np.sum(anti_comm_check) == 0
    ), "operator not made up of pairwisie anticommuting Pauli operators"


def test_random_anitcomm_2n_1_PauliwordOp_complex_with_clifford():
    n_qubits = 4
    complex_coeff = True
    apply_clifford = True

    AC = random_anitcomm_2n_1_PauliwordOp(
        n_qubits, apply_clifford=apply_clifford, complex_coeff=complex_coeff
    )

    assert AC.n_terms == 2 * n_qubits + 1
    anti_comm_check = AC.adjacency_matrix.astype(int) - np.eye(
        AC.adjacency_matrix.shape[0]
    )
    assert (
        np.sum(anti_comm_check) == 0
    ), "operator not made up of pairwisie anticommuting Pauli operators"


def test_tensor_list():

    X_term = PauliwordOp.from_list(["X"], [0.25])
    Y_term = PauliwordOp.from_list(["Y"], [0.25])
    Z_term = PauliwordOp.from_list(["Z"], [0.25])

    targ = PauliwordOp.from_list(["XYZ"], [0.25 * 0.25 * 0.25])

    P_out = tensor_list([X_term, Y_term, Z_term])

    assert P_out == targ


def test_product_list():
    n_qubits = 4
    P_random = PauliwordOp.random(n_qubits=n_qubits, n_terms=10)

    open_F_prod = QubitOperator("", 1)
    for p in P_random:
        open_F_prod *= p.to_openfermion

    symmer_prod = product_list(P_random)
    open_F_prod = PauliwordOp.from_openfermion(open_F_prod, n_qubits=n_qubits)

    assert open_F_prod == symmer_prod


def test_gram_schmidt_from_quantum_state_QuantumState():
    nq = 3
    psi = QuantumState.haar_random(nq)

    U_gram = gram_schmidt_from_quantum_state(psi)
    assert np.allclose(
        U_gram[:, 0], psi.to_sparse_matrix.toarray().reshape([-1])
    ), "first column of U_gram not correct"
    assert np.allclose(U_gram @ U_gram.conj().T, np.eye(2**nq)), "U_gram not unitary"


def test_gram_schmidt_from_quantum_state_numpy_array():
    nq = 3
    psi = np.arange(0, 2**nq)
    psi_norm = psi / np.linalg.norm(psi)

    U_gram = gram_schmidt_from_quantum_state(psi_norm)
    assert np.allclose(U_gram[:, 0], psi_norm), "first column of U_gram not correct"
    assert np.allclose(U_gram @ U_gram.conj().T, np.eye(2**nq)), "U_gram not unitary"


def test_Draw_molecule():

    xyz = H2_sto3g["geometry"]
    viewer_sphere = Draw_molecule(xyz, width=400, height=400, style="sphere")
    assert isinstance(viewer_sphere, py3Dmol.view)

    viewer_stick = Draw_molecule(xyz, width=400, height=400, style="stick")
    assert isinstance(viewer_stick, py3Dmol.view)


def test_get_sparse_matrix_large_pauliwordop():
    for nq in range(2, 6):
        n_terms = 10 * nq
        random_P = PauliwordOp.random(nq, n_terms)
        sparse_mat = get_sparse_matrix_large_pauliwordop(random_P)
        assert np.allclose(random_P.to_sparse_matrix.toarray(), sparse_mat.toarray())


def test_matrix_allclose_sparse():
    for nq in range(2, 6):
        n_terms = 10 * nq
        random_P = PauliwordOp.random(nq, n_terms)
        sparse_mat = get_sparse_matrix_large_pauliwordop(random_P)
        assert matrix_allclose(random_P.to_sparse_matrix, sparse_mat)

    # assert false output
    Pop_XI = PauliwordOp.from_list(["XI"]).to_sparse_matrix
    Pop_ZI = PauliwordOp.from_list(["ZI"]).to_sparse_matrix
    assert not matrix_allclose(Pop_XI, Pop_ZI)


def test_matrix_allclose_dense():
    for nq in range(2, 6):
        n_terms = 10 * nq
        random_P = PauliwordOp.random(nq, n_terms)
        sparse_mat = get_sparse_matrix_large_pauliwordop(random_P)
        assert matrix_allclose(
            random_P.to_sparse_matrix.toarray(), sparse_mat.toarray()
        )

    # assert false output
    Pop_XI = PauliwordOp.from_list(["XI"]).to_sparse_matrix
    Pop_ZI = PauliwordOp.from_list(["ZI"]).to_sparse_matrix
    assert not matrix_allclose(Pop_XI.toarray(), Pop_ZI.toarray())
