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
    "computational_basis_mps",
    "mps_to_arrays",
    "top_amplitude_bitstrings",
]

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
    method: str = "svd",
    verbose: bool = False,
) -> List[np.ndarray]:
    """Build a compressed MPO for ``sum_k coeffs[k] * prod_i Pauli(codes[k, i])``.

    The MPO is produced by a single left-to-right sweep that truncates each
    bond via an SVD, so the bond dimension never exceeds ``max_bond`` and no
    ``O(K**2)`` intermediate is ever formed. With ``max_bond=None`` and a small
    ``cutoff`` the MPO is *exact* (up to ``cutoff``).

    Args:
        codes: ``int8`` array ``(n_terms, n_qubits)`` of Pauli codes.
        coeffs: complex array ``(n_terms,)`` of coefficients.
        max_bond: Maximum bond dimension (``None`` = no cap, exact up to
            ``cutoff``). NOTE: this truncates the *Hamiltonian operator*, not the
            state. Truncating an operator is not variational - for accurate
            energies keep the MPO essentially exact (``None`` / large ``max_bond``
            and small ``cutoff``) and control accuracy via the DMRG *state* bond
            dimension instead.
        cutoff: Relative singular-value threshold; singular values below
            ``cutoff * max_singular_value`` at a bond are discarded.
        dtype: Working dtype. ``np.complex128`` (default) is recommended,
            especially for chemistry Hamiltonians. ``np.complex64`` halves
            memory but limits accuracy to ~1e-6.
        method: ``'svd'`` (default) forms ``A`` (shape ``(4*D, K)``) and takes a
            thin SVD - accurate and robust, memory ``O(D * K)``. ``'gram'``
            truncates via the ``(4D, 4D)`` reduced density matrix without forming
            ``A`` (lowest memory for enormous ``K``), but squares the singular
            values so its usable precision is limited to ~``sqrt(eps)`` and its
            effective cutoff is floored accordingly.
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
    if method not in ("svd", "gram"):
        raise ValueError(f"Unknown method {method!r}; use 'svd' or 'gram'.")

    pmats = PAULI_MATS.astype(dtype)
    pvecs = PAULI_VECS.astype(dtype)
    # Noise floor for the Gram method: eigenvalues (= sigma**2) below
    # ~eps * sigma_max**2 are numerical zeros and must be dropped, else
    # rank-deficient directions inflate the bond dimension.
    eps = np.finfo(np.float64 if dtype == np.complex128 else np.float32).eps
    gram_floor = 100.0 * eps  # relative eigenvalue floor

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

        if method == "svd":
            # Form A (shape (4*Dl, K)):  A[(sl, a), k] = vec_p(sl) * C[a, k]
            # for k of Pauli-type p, then take a thin SVD.  A = U S V^H.
            A = np.zeros((4 * Dl, K), dtype=dtype)
            for p in range(4):
                mask = masks[p]
                if mask.any():
                    A[:, mask] = np.kron(pvecs[p].reshape(4, 1), C[:, mask])
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            smax = S[0] if S.size else 0.0
            if smax <= 0.0:
                keep = 1
            else:
                keep = int(np.count_nonzero(S > cutoff * smax))
                keep = max(1, min(keep, max_bond, S.size))
            L = U[:, :keep]                                   # (4*Dl, Dr)
            Dr = keep
            C = S[:keep, None] * Vh[:keep, :]                 # (Dr, K) = L^H A
        else:  # method == "gram"
            # Reduced density matrix G = A A^H (4*Dl, 4*Dl), without forming A.
            G = np.zeros((4 * Dl, 4 * Dl), dtype=dtype)
            for p in range(4):
                mask = masks[p]
                if not mask.any():
                    continue
                Cp = C[:, mask]
                gram_p = Cp @ Cp.conj().T
                outer_p = np.outer(pvecs[p], pvecs[p].conj())
                G += np.kron(outer_p, gram_p)
            G = 0.5 * (G + G.conj().T)
            evals, evecs = np.linalg.eigh(G)
            evals = np.clip(evals[::-1], 0.0, None)
            evecs = evecs[:, ::-1]
            emax = evals[0]
            if emax <= 0.0:
                keep = 1
            else:
                thresh = max((cutoff ** 2), gram_floor) * emax
                keep = int(np.count_nonzero(evals > thresh))
                keep = max(1, min(keep, max_bond, evals.size))
            L = evecs[:, :keep]
            Dr = keep
            W = L.reshape(2, 2, Dl, Dr)
            Wc = W.conj()
            C_new = np.zeros((Dr, K), dtype=dtype)
            for p in range(4):
                mask = masks[p]
                if not mask.any():
                    continue
                T_p = np.einsum("slad,sl->da", Wc, pmats[p], optimize=True)
                C_new[:, mask] = T_p @ C[:, mask]
            C = C_new

        tensors.append(L.reshape(2, 2, Dl, Dr))
        if verbose:
            print(f"  site {i:>4d}: bond {Dl} -> {Dr}")
        Dl = Dr

    # Final site: W[s, l, a] = sum_k C[a, k] * Pauli(codes[k, -1])[s, l]
    col = codes[:, n - 1]
    W_last = np.zeros((2, 2, Dl, 1), dtype=dtype)
    for p in range(4):
        mask = col == p
        if not mask.any():
            continue
        rowsum = C[:, mask].sum(axis=1)
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
    from quimb.tensor import MatrixProductOperator

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
    method: str = "svd",
    verbose: bool = False,
    return_tensors: bool = False,
):
    """Build a (compressed) quimb MPO directly from a :class:`PauliwordOp`.

    Args:
        operator: The operator to represent.
        max_bond: Maximum bond dimension (``None`` = uncapped/exact). This
            truncates the *operator*; for accurate energies leave it ``None``
            (see :func:`build_mpo_tensors`).
        cutoff: Relative singular-value truncation threshold.
        dtype: Working dtype (``np.complex128`` recommended).
        method: ``'svd'`` (default, accurate) or ``'gram'`` (low memory for
            enormous term counts).
        verbose: Print per-site bond dimensions.
        return_tensors: If True, also return the raw tensor list.

    Returns:
        The quimb ``MatrixProductOperator`` (and, if ``return_tensors``, the
        list of tensors).
    """
    codes, coeffs = pauliwordop_to_codes(operator)
    tensors = build_mpo_tensors(
        codes, coeffs, max_bond=max_bond, cutoff=cutoff, dtype=dtype,
        method=method, verbose=verbose
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
    method: str = "svd",
    verbose: bool = False,
    return_tensors: bool = False,
):
    """Build a (compressed) quimb MPO from Pauli strings + coefficients.

    See :func:`mpo_from_pauliwordop` for the meaning of the shared arguments.
    """
    codes, coeffs = pauli_strings_to_codes(pstrings, coeffs)
    tensors = build_mpo_tensors(
        codes, coeffs, max_bond=max_bond, cutoff=cutoff, dtype=dtype,
        method=method, verbose=verbose
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
def computational_basis_mps(bitstring: Union[str, Sequence[int]], dtype=np.complex128):
    """Build a bond-dimension-1 MPS for a single computational basis state.

    No dense ``2**n`` vector is ever formed - the result is an exact product-state
    MPS. Qubit ``i`` corresponds to site ``i`` (same ordering as the Pauli
    strings / :func:`top_amplitude_bitstrings`).

    Args:
        bitstring: e.g. ``"0011"`` or ``[0, 0, 1, 1]``. ``bitstring[i]`` is the
            value of qubit ``i``.
        dtype: MPS data dtype.

    Returns:
        A quimb ``MatrixProductState`` representing ``|bitstring>``.
    """
    from quimb.tensor import MPS_computational_state

    if not isinstance(bitstring, str):
        bitstring = "".join(str(int(b)) for b in bitstring)
    if any(c not in "01" for c in bitstring):
        raise ValueError(f"bitstring must contain only 0/1, got {bitstring!r}.")
    return MPS_computational_state(bitstring, dtype=np.dtype(dtype).name)


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
        p0: Optional initial state. May be a quimb MPS, a dense array, or - to
            start from a single computational basis state without ever forming a
            dense vector - a bitstring such as ``"0011"`` or a sequence of bits
            ``[0, 0, 1, 1]`` (``p0[i]`` is the value of qubit ``i``).
        as_quantum_state: If True, densify the result to a
            :class:`symmer.operators.QuantumState`. Only allowed for small
            systems (see ``max_qubits_for_dense``) since it materialises a
            ``2**n`` vector.
        max_qubits_for_dense: Guard against accidentally densifying huge states.

    Returns:
        A dict with keys ``'energy'`` (float), ``'mps'`` (the quimb ground-state
        MPS) and, if requested, ``'state'`` (a ``QuantumState``).
    """
    from quimb.tensor import DMRG2, MatrixProductState

    if isinstance(mpo, list):
        mpo = tensors_to_quimb_mpo(mpo)

    n_qubits = mpo.L

    if p0 is not None and not isinstance(p0, MatrixProductState):
        is_bitstring = isinstance(p0, str) or (
            isinstance(p0, (list, tuple, np.ndarray))
            and np.asarray(p0).ndim == 1
            and len(p0) == n_qubits
            and set(np.unique(np.asarray(p0)).tolist()) <= {0, 1}
        )
        if is_bitstring:
            # single computational basis state -> bond-1 product MPS (no dense)
            p0 = computational_basis_mps(p0, dtype=mpo.dtype)
        else:
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


# ---------------------------------------------------------------------------
# Extracting the largest-amplitude bitstrings from an MPS
# ---------------------------------------------------------------------------
def mps_to_arrays(mps) -> List[np.ndarray]:
    """Extract a quimb MPS as a list of ``(D_left, 2, D_right)`` numpy arrays.

    The arrays are ordered ``(left_bond, physical, right_bond)`` with the
    physical dimension indexed as ``0 -> |0>``, ``1 -> |1>``. Boundary sites get
    singleton bonds so the first array is ``(1, 2, D)`` and the last ``(D, 2, 1)``.

    Args:
        mps: A quimb ``MatrixProductState``.

    Returns:
        List of site tensors as numpy arrays.
    """
    n = mps.L
    arrays: List[np.ndarray] = []
    for i in range(n):
        t = mps[i]
        pind = mps.site_ind(i)
        bond_inds = [ix for ix in t.inds if ix != pind]
        left = right = None
        if i > 0:
            prev = set(mps[i - 1].inds)
            left = next(ix for ix in bond_inds if ix in prev)
        if i < n - 1:
            nxt = set(mps[i + 1].inds)
            right = next(ix for ix in bond_inds if ix in nxt)
        order = [ix for ix in (left, pind, right) if ix is not None]
        arr = np.asarray(t.transpose(*order).data)
        d = arr.shape[order.index(pind)]
        if left is None and right is None:
            arr = arr.reshape(1, d, 1)
        elif left is None:
            arr = arr.reshape(1, d, -1)
        elif right is None:
            arr = arr.reshape(-1, d, 1)
        arrays.append(np.ascontiguousarray(arr))
    return arrays


def _right_canonicalize(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Put a list of ``(Dl, d, Dr)`` MPS tensors into right-canonical form.

    After this, every site ``i >= 1`` is a right isometry
    (``sum_{p,r} A[l,p,r] conj(A[l',p,r]) = delta_{l,l'}``) and the norm of the
    state is carried by site 0. Uses an RQ sweep from right to left.
    """
    A = [a.astype(np.complex128, copy=True) for a in arrays]
    n = len(A)
    for i in range(n - 1, 0, -1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl, d * Dr)
        # RQ via QR of M^H:  M = R @ Q, with Q having orthonormal rows.
        Q1, R1 = np.linalg.qr(M.conj().T, mode="reduced")
        Q = Q1.conj().T
        R = R1.conj().T
        A[i] = Q.reshape(Q.shape[0], d, Dr)
        A[i - 1] = np.einsum("lpr,rs->lps", A[i - 1], R, optimize=True)
    return A


def _search_beam(A: List[np.ndarray], k: int, beam_width: int):
    """Beam search over an MPS for the largest-amplitude bitstrings.

    Fast and always terminating: keeps the ``beam_width`` highest-probability
    prefixes at each site. Exact when ``beam_width`` is large enough (in
    particular for peaked states); otherwise an excellent approximation.
    """
    n = len(A)
    beam = [((), np.ones((1, 1), dtype=A[0].dtype))]  # (prefix, row-vector v)
    for d in range(n):
        A_site = A[d]
        cand = []
        for prefix, v in beam:
            for bit in (0, 1):
                v2 = v @ A_site[:, bit, :]
                nb = float(np.linalg.norm(v2))
                if nb > 0:
                    cand.append((nb, prefix + (bit,), v2))
        cand.sort(key=lambda x: x[0], reverse=True)
        beam = [(p, v) for _, p, v in cand[:beam_width]]
    finals = [(prefix, complex(v.reshape(()))) for prefix, v in beam]
    finals.sort(key=lambda x: abs(x[1]), reverse=True)
    return finals[:k]


def _search_exact(A: List[np.ndarray], k: int, max_nodes: int):
    """Exact best-first branch-and-bound for the top-``k`` amplitudes.

    Guaranteed correct (the bound ``sqrt(P(prefix))`` is admissible), but the
    frontier can grow exponentially for near-uniform / highly entangled states.
    """
    import heapq
    import warnings

    n = len(A)
    counter = 0
    heap: list = []  # (-bound, tiebreak, prefix, v)
    for bit in (0, 1):
        v = A[0][:, bit, :]
        b = float(np.linalg.norm(v))
        if b > 0:
            heapq.heappush(heap, (-b, counter, (bit,), v))
            counter += 1

    results = []
    nodes = 0
    while heap and len(results) < k:
        _, _, prefix, v = heapq.heappop(heap)
        if len(prefix) == n:
            results.append((prefix, complex(v.reshape(()))))
            continue
        nodes += 1
        if nodes > max_nodes:
            warnings.warn(
                f"top_amplitude_bitstrings(method='exact') hit "
                f"max_nodes={max_nodes} after finding {len(results)} of {k} "
                f"bitstrings. The state is likely too entangled/flat for exact "
                f"search; use method='beam' (default) instead.",
                RuntimeWarning,
            )
            break
        A_next = A[len(prefix)]
        for bit in (0, 1):
            v2 = v @ A_next[:, bit, :]
            b = float(np.linalg.norm(v2))
            if b > 0:
                heapq.heappush(heap, (-b, counter, prefix + (bit,), v2))
                counter += 1
    return results


def top_amplitude_bitstrings(
    mps,
    k: int = 20,
    method: str = "beam",
    beam_width: Optional[int] = None,
    max_nodes: int = 2_000_000,
    normalize: bool = True,
    return_probabilities: bool = True,
):
    """Return the ``k`` largest-amplitude computational-basis bitstrings of an MPS.

    The full ``2**n`` state is never formed. Two strategies are available:

    * ``method='beam'`` (default): beam search keeping the ``beam_width``
      most-probable prefixes at each site. Always fast (``O(n * beam_width *
      D**2)``) and terminating. Exact for peaked states (e.g. chemistry ground
      states dominated by a few determinants) once ``beam_width`` is large
      enough; otherwise a high-quality approximation. Increase ``beam_width``
      for more accuracy.
    * ``method='exact'``: best-first branch-and-bound using the admissible bound
      ``sqrt(P(prefix))``, guaranteed to return the true top-``k``. Efficient
      only for peaked/low-entanglement states - for near-uniform states the
      frontier explodes and the search is capped by ``max_nodes`` (with a
      warning).

    Args:
        mps: A quimb ``MatrixProductState`` (e.g.
            ``find_groundstate_quimb(...)['mps']``) or a list of ``(Dl, 2, Dr)``
            numpy arrays.
        k: Number of bitstrings to return.
        method: ``'beam'`` or ``'exact'``.
        beam_width: Beam size (``method='beam'`` only). Defaults to
            ``max(4 * k, 64)``.
        max_nodes: Node cap for ``method='exact'``.
        normalize: If True, normalise the state so probabilities sum to 1.
        return_probabilities: Include the probability ``|amplitude|**2`` in the
            output tuples.

    Returns:
        List of ``(bitstring, amplitude[, probability])`` tuples sorted by
        descending amplitude magnitude. ``bitstring[i]`` is the value of qubit
        ``i`` (same site/qubit ordering as the Pauli strings used to build the
        MPO).
    """
    arrays = mps if isinstance(mps, list) else mps_to_arrays(mps)
    A = _right_canonicalize(arrays)

    if normalize:
        nrm = np.linalg.norm(A[0])
        if nrm > 0:
            A[0] = A[0] / nrm

    if method == "beam":
        if beam_width is None:
            beam_width = max(4 * k, 64)
        results = _search_beam(A, k, beam_width)
    elif method == "exact":
        results = _search_exact(A, k, max_nodes)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'beam' or 'exact'.")

    out = []
    for prefix, amp in results:
        bitstring = "".join(str(b) for b in prefix)
        if return_probabilities:
            out.append((bitstring, amp, abs(amp) ** 2))
        else:
            out.append((bitstring, amp))
    return out
