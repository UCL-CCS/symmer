"""
Scalable construction of Matrix Product Operators (MPOs) from sums of Pauli
strings, plus a DMRG ground-state solver built on ``quimb``.

Motivation
----------
The original :mod:`symmer.approximate.tensor_network` builds the *exact* MPO of a
sum of ``K`` Pauli strings before compressing it. The exact MPO has a bond
dimension equal to ``K`` in the bulk, so the middle site tensors have shape
``(2, 2, K, K)`` (dense ``np.diag`` matrices). This uses ``O(K**2)`` memory per
site and requires dense SVDs of ``(4K, K)`` matrices - both of which blow up for
``K`` in the hundreds-of-thousands-to-millions range.

This module instead performs a single left-to-right *compression sweep* that
never materialises the full ``K``-dimensional bond. It keeps a running
"carry" matrix of shape ``(D, K)`` (``D`` = current bond dimension, capped at
``max_bond``) and truncates each bond using the small ``(4D, 4D)`` reduced
density matrix (Gram matrix). Building that Gram matrix, and updating the carry,
both cost ``O(K * D**2)`` per site by grouping the ``K`` terms according to
which of the four single-qubit Paulis (I, X, Y, Z) they carry at that site.

Complexity
----------
* time:   ``O(n_qubits * K * D**2)`` (+ per-site dense ``eigh`` on a ``4D x 4D``
          matrix)
* memory: ``O(D * K)`` for the carry, ``O(K * n_qubits)`` for the term codes,
          and ``O(sum_i D_i**2)`` for the resulting MPO tensors.

with ``D <= max_bond``. This is what makes ``n_qubits ~ 40`` and
``K ~ 10**5 - 10**6`` tractable (subject to available RAM for the carry matrix,
which is ``max_bond * K * itemsize`` bytes).
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing heavy/optional deps at module import time
    from symmer.operators import PauliwordOp, QuantumState

__all__ = [
    "PAULI_MATS",
    "PAULI_VECS",
    "pauliwordop_to_codes",
    "pauli_strings_to_codes",
    "build_mpo_tensors",
    "tensors_to_quimb_mpo",
    "mpo_from_pauliwordop",
    "mpo_from_pauli_strings",
    "mpo_tensors_to_dense",
    "find_groundstate_quimb",
]

from quimb.tensor.tn1d.core import MatrixProductOperator, MatrixProductState
from quimb.tensor.tn1d.dmrg import DMRG2

# Single-qubit Paulis, indexed I=0, X=1, Y=2, Z=3.
# Row index = physical "out" leg (sigma), column index = physical "in" leg (l).
PAULI_MATS = np.array(
    [
        [[1, 0], [0, 1]],       # I
        [[0, 1], [1, 0]],       # X
        [[0, -1j], [1j, 0]],    # Y
        [[1, 0], [0, -1]],      # Z
    ],
    dtype=np.complex128,
)

# Vectorised Paulis, flattened in C-order as index = sigma * 2 + l.
PAULI_VECS = PAULI_MATS.reshape(4, 4)

_CHAR_TO_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}


# ---------------------------------------------------------------------------
# Encoding Pauli operators as compact integer "codes"
# ---------------------------------------------------------------------------
def pauliwordop_to_codes(
    operator: "PauliwordOp",
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a :class:`PauliwordOp` as integer Pauli codes + coefficients.

    This reads the symplectic representation directly (no per-term string
    building), so it is fast even for millions of terms.

    Args:
        operator: The operator (a linear combination of Paulis).

    Returns:
        codes: ``int8`` array of shape ``(n_terms, n_qubits)`` with entries in
            ``{0: I, 1: X, 2: Y, 3: Z}``.
        coeffs: ``complex128`` array of shape ``(n_terms,)``.
    """
    X = np.asarray(operator.X_block, dtype=bool)
    Z = np.asarray(operator.Z_block, dtype=bool)
    codes = np.zeros(X.shape, dtype=np.int8)
    # Convention matches symmer.operators.utils.symplectic_to_string:
    #   Y where X&Z, X where X&~Z, Z where Z&~X, I otherwise.
    codes[X & ~Z] = 1
    codes[X & Z] = 2
    codes[Z & ~X] = 3
    coeffs = np.asarray(operator.coeff_vec, dtype=np.complex128)
    return codes, coeffs


def pauli_strings_to_codes(
    pstrings: Sequence[str],
    coeffs: Optional[Sequence[complex]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a list of Pauli strings (e.g. ``"IXYZ"``) as codes + coefficients.

    Args:
        pstrings: Iterable of equal-length Pauli strings over ``{I, X, Y, Z}``.
        coeffs: Optional iterable of coefficients (defaults to all ones).

    Returns:
        codes: ``int8`` array of shape ``(n_terms, n_qubits)``.
        coeffs: ``complex128`` array of shape ``(n_terms,)``.
    """
    pstrings = list(pstrings)
    n_terms = len(pstrings)
    if n_terms == 0:
        raise ValueError("Received an empty list of Pauli strings.")
    n_qubits = len(pstrings[0])
    codes = np.empty((n_terms, n_qubits), dtype=np.int8)
    for row, pstr in enumerate(pstrings):
        if len(pstr) != n_qubits:
            raise ValueError("All Pauli strings must have the same length.")
        codes[row] = [_CHAR_TO_CODE[c] for c in pstr]
    if coeffs is None:
        coeffs = np.ones(n_terms, dtype=np.complex128)
    else:
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        if coeffs.shape[0] != n_terms:
            raise ValueError("coeffs and pstrings must have the same length.")
    return codes, coeffs


# ---------------------------------------------------------------------------
# Core: scalable left-to-right compression sweep
# ---------------------------------------------------------------------------
def build_mpo_tensors(
    codes: np.ndarray,
    coeffs: np.ndarray,
    max_bond: Optional[int] = None,
    cutoff: float = 1e-12,
    dtype: np.dtype = np.complex128,
    verbose: bool = False,
) -> List[np.ndarray]:
    """Build a compressed MPO for ``sum_k coeffs[k] * prod_i Pauli(codes[k, i])``.

    The MPO is produced by a single left-to-right sweep that truncates each
    bond via the reduced density matrix, so the bond dimension never exceeds
    ``max_bond`` and no ``O(K**2)`` intermediate is ever formed.

    Args:
        codes: ``int8`` array ``(n_terms, n_qubits)`` of Pauli codes.
        coeffs: complex array ``(n_terms,)`` of coefficients.
        max_bond: Maximum bond dimension (``None`` = no cap, exact up to
            ``cutoff``).
        cutoff: Relative singular-value threshold. Singular values below
            ``cutoff * max_singular_value`` at a bond are discarded.
        dtype: Working dtype for the tensors and carry. ``np.complex64`` halves
            memory/time at reduced precision; ``np.complex128`` is the default.
        verbose: If True, print the bond dimension found at each site.

    Returns:
        List of ``n_qubits`` MPO tensors, each of shape
        ``(2, 2, D_left, D_right)`` and ordered as ``(sigma_out, sigma_in,
        left_bond, right_bond)``. The first tensor has ``D_left == 1`` and the
        last has ``D_right == 1``.
    """
    codes = np.ascontiguousarray(codes)
    K, n = codes.shape
    if K == 0:
        raise ValueError("Need at least one term to build an MPO.")
    if len(coeffs) != K:
        raise ValueError("coeffs and codes must have matching first dimension.")

    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    pmats = PAULI_MATS.astype(dtype)
    pvecs = PAULI_VECS.astype(dtype)

    if max_bond is None:
        max_bond = 2 ** 62  # effectively unbounded

    # Single-qubit operator: just sum the (scaled) Paulis.
    if n == 1:
        W = np.zeros((2, 2), dtype=dtype)
        col = codes[:, 0]
        for p in range(4):
            mask = col == p
            if mask.any():
                W += coeffs[mask].sum() * pmats[p]
        return [W.reshape(2, 2, 1, 1)]

    # Carry matrix C: shape (D_left, K). The coefficients live on the first site.
    C = coeffs.astype(dtype).reshape(1, K).copy()
    Dl = 1
    tensors: List[np.ndarray] = []

    for i in range(n - 1):
        col = codes[:, i]
        masks = [col == p for p in range(4)]

        # Reduced density matrix G = A A^H of shape (4*Dl, 4*Dl), assembled
        # without ever forming the full A (shape (4*Dl, K)):
        #   G = sum_p kron( outer(vec_p, conj(vec_p)),  C_p C_p^H )
        G = np.zeros((4 * Dl, 4 * Dl), dtype=dtype)
        for p in range(4):
            mask = masks[p]
            if not mask.any():
                continue
            Cp = C[:, mask]                       # (Dl, K_p)
            gram_p = Cp @ Cp.conj().T             # (Dl, Dl)
            outer_p = np.outer(pvecs[p], pvecs[p].conj())  # (4, 4)
            G += np.kron(outer_p, gram_p)

        G = 0.5 * (G + G.conj().T)                # enforce Hermiticity
        evals, evecs = np.linalg.eigh(G)          # ascending, real evals
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
        evals = np.clip(evals, 0.0, None)

        smax = np.sqrt(evals[0]) if evals[0] > 0 else 0.0
        if smax == 0.0:
            keep = 1
        else:
            thresh = (cutoff * smax) ** 2
            keep = int(np.count_nonzero(evals > thresh))
            keep = max(1, min(keep, max_bond, evals.size))

        L = evecs[:, :keep]                       # (4*Dl, Dr) isometry
        Dr = keep
        W = L.reshape(2, 2, Dl, Dr)
        tensors.append(W)

        if verbose:
            print(f"  site {i:>4d}: bond {Dl} -> {Dr}")

        # Update the carry: C_new = L^H A, computed per-Pauli group.
        Wc = W.conj()
        C_new = np.zeros((Dr, K), dtype=dtype)
        for p in range(4):
            mask = masks[p]
            if not mask.any():
                continue
            # T_p[a', a] = sum_{s,l} conj(W[s,l,a,a']) * pmat[s,l]
            T_p = np.einsum("slad,sl->da", Wc, pmats[p], optimize=True)  # (Dr, Dl)
            C_new[:, mask] = T_p @ C[:, mask]
        C = C_new
        Dl = Dr

    # Final site: W[s, l, a] = sum_k C[a, k] * Pauli(codes[k, -1])[s, l]
    col = codes[:, n - 1]
    W_last = np.zeros((2, 2, Dl, 1), dtype=dtype)
    for p in range(4):
        mask = col == p
        if not mask.any():
            continue
        rowsum = C[:, mask].sum(axis=1)           # (Dl,)
        W_last[:, :, :, 0] += np.multiply.outer(pmats[p], rowsum)
    tensors.append(W_last)

    if verbose:
        print(f"  site {n - 1:>4d}: bond {Dl} -> 1")

    return tensors


# ---------------------------------------------------------------------------
# quimb interoperability
# ---------------------------------------------------------------------------
def tensors_to_quimb_mpo(tensors: List[np.ndarray]):
    """Wrap a list of MPO tensors as a ``quimb`` ``MatrixProductOperator``.

    Args:
        tensors: List of ``(2, 2, D_left, D_right)`` arrays from
            :func:`build_mpo_tensors`.

    Returns:
        ``quimb.tensor.MatrixProductOperator`` with index order ``'dulr'``.
    """
    n = len(tensors)
    if n == 1:
        # Single site operator: drop both trivial bonds -> shape 'du'.
        return MatrixProductOperator([tensors[0][:, :, 0, 0]], shape="du")

    arrays = []
    for i, W in enumerate(tensors):
        if i == 0:
            arrays.append(W[:, :, 0, :])   # (d, u, r)
        elif i == n - 1:
            arrays.append(W[:, :, :, 0])   # (d, u, l)
        else:
            arrays.append(W)               # (d, u, l, r)
    return MatrixProductOperator(arrays, shape="dulr")


def mpo_from_pauliwordop(
    operator: "PauliwordOp",
    max_bond: Optional[int] = None,
    cutoff: float = 1e-12,
    dtype: np.dtype = np.complex128,
    verbose: bool = False,
    return_tensors: bool = False,
):
    """Build a (compressed) quimb MPO directly from a :class:`PauliwordOp`.

    Args:
        operator: The operator to represent.
        max_bond: Maximum bond dimension (``None`` = uncapped).
        cutoff: Relative singular-value truncation threshold.
        dtype: Working dtype (``np.complex64`` or ``np.complex128``).
        verbose: Print per-site bond dimensions.
        return_tensors: If True, also return the raw tensor list.

    Returns:
        The quimb ``MatrixProductOperator`` (and, if ``return_tensors``, the
        list of tensors).
    """
    codes, coeffs = pauliwordop_to_codes(operator)
    tensors = build_mpo_tensors(
        codes, coeffs, max_bond=max_bond, cutoff=cutoff, dtype=dtype, verbose=verbose
    )
    mpo = tensors_to_quimb_mpo(tensors)
    if return_tensors:
        return mpo, tensors
    return mpo


def mpo_from_pauli_strings(
    pstrings: Sequence[str],
    coeffs: Optional[Sequence[complex]] = None,
    max_bond: Optional[int] = None,
    cutoff: float = 1e-12,
    dtype: np.dtype = np.complex128,
    verbose: bool = False,
    return_tensors: bool = False,
):
    """Build a (compressed) quimb MPO from Pauli strings + coefficients.

    See :func:`mpo_from_pauliwordop` for the meaning of the shared arguments.
    """
    codes, coeffs = pauli_strings_to_codes(pstrings, coeffs)
    tensors = build_mpo_tensors(
        codes, coeffs, max_bond=max_bond, cutoff=cutoff, dtype=dtype, verbose=verbose
    )
    mpo = tensors_to_quimb_mpo(tensors)
    if return_tensors:
        return mpo, tensors
    return mpo


# ---------------------------------------------------------------------------
# Utilities / validation helpers (small systems only!)
# ---------------------------------------------------------------------------
def mpo_tensors_to_dense(tensors: List[np.ndarray]) -> np.ndarray:
    """Contract MPO tensors into a dense ``(2**n, 2**n)`` matrix.

    Intended for validation on small systems only - the result has
    ``4**n_qubits`` entries.

    Args:
        tensors: List of ``(2, 2, D_left, D_right)`` MPO tensors.

    Returns:
        Dense operator matrix.
    """
    contr = tensors[0]  # (S, L, a, b) with S = L = 2, a = 1
    for W in tensors[1:]:
        s2, l2, i2, j2 = W.shape
        S, L, a, b = contr.shape
        contr = np.einsum("SLab,slbc->SsLlac", contr, W, optimize=True)
        contr = contr.reshape(S * s2, L * l2, a, j2)
    return contr[:, :, 0, 0]


# ---------------------------------------------------------------------------
# DMRG ground-state solver
# ---------------------------------------------------------------------------
def find_groundstate_quimb(
    mpo,
    bond_dims: Union[int, Sequence[int]] = (10, 20, 50, 100, 100, 200),
    cutoffs: Union[float, Sequence[float]] = 1e-10,
    tol: float = 1e-6,
    max_sweeps: int = 20,
    verbosity: int = 0,
    p0=None,
    as_quantum_state: bool = False,
    max_qubits_for_dense: int = 24,
):
    """Approximate the ground state of an MPO with quimb's ``DMRG2``.

    Accepts either a quimb ``MatrixProductOperator`` or a list of MPO tensors
    (as returned by :func:`build_mpo_tensors`).

    Args:
        mpo: quimb MPO or list of ``(2, 2, D_left, D_right)`` tensors.
        bond_dims: DMRG bond-dimension schedule (or a single int).
        cutoffs: DMRG truncation cutoff schedule (or a single float).
        tol: Convergence tolerance on the energy.
        max_sweeps: Maximum number of DMRG sweeps.
        verbosity: quimb DMRG print level.
        p0: Optional initial state (dense array or quimb MPS).
        as_quantum_state: If True, densify the result to a
            :class:`symmer.operators.QuantumState`. Only allowed for small
            systems (see ``max_qubits_for_dense``) since it materialises a
            ``2**n`` vector.
        max_qubits_for_dense: Guard against accidentally densifying huge states.

    Returns:
        A dict with keys ``'energy'`` (float), ``'mps'`` (the quimb ground-state
        MPS) and, if requested, ``'state'`` (a ``QuantumState``).
    """
    if isinstance(mpo, list):
        mpo = tensors_to_quimb_mpo(mpo)

    n_qubits = mpo.L

    if p0 is not None and not isinstance(p0, MatrixProductState):
        p0 = np.asarray(p0)
        dims = [2] * int(np.log2(p0.shape[0]))
        p0 = MatrixProductState.from_dense(p0, dims)

    dmrg = DMRG2(mpo, bond_dims=bond_dims, cutoffs=cutoffs, p0=p0)
    dmrg.solve(tol=tol, max_sweeps=max_sweeps, verbosity=verbosity)

    result = {"energy": float(np.real(dmrg.energy)), "mps": dmrg.state}

    if as_quantum_state:
        if n_qubits > max_qubits_for_dense:
            raise ValueError(
                f"Refusing to densify a {n_qubits}-qubit state "
                f"(> max_qubits_for_dense={max_qubits_for_dense}); a dense "
                f"vector would need 2**{n_qubits} amplitudes. Use the returned "
                f"'mps' instead."
            )
        from symmer.operators import QuantumState

        dense = dmrg.state.to_dense()
        result["state"] = QuantumState.from_array(dense).cleanup(zero_threshold=1e-5)

    return result
