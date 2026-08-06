"""Estimator kernels.  This is the code that ships.

Every kernel here is written directly against flopscope, so the algorithm that
is researched is byte-for-byte the algorithm that is graded — there is no
second "fast" implementation that could drift from it.  Ground-truth Monte
Carlo lives in :mod:`whestfloor.mc` and uses raw NumPy; it is never shipped.

A kernel has the signature ``f(weights, ctx=None) -> fnp.ndarray`` of shape
``(depth, width)``, where ``weights`` is the list of ``(width, width)`` float32
matrices and the forward convention is ``x @ W``.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp

# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------


def _relu_gauss(mu_pre, var_pre, sig):
    """Exact E[relu], E[relu^2] and Var[relu] for z ~ N(mu_pre, var_pre)."""
    alpha = mu_pre / sig
    ph = flops.stats.norm.pdf(alpha)
    Ph = flops.stats.norm.cdf(alpha)
    mu = mu_pre * Ph + sig * ph
    ez2 = (mu_pre * mu_pre + var_pre) * Ph + mu_pre * sig * ph
    var = fnp.maximum(ez2 - mu * mu, 0.0)
    return mu, var, alpha, ph, Ph


def _hermite_coeffs(alpha, sig, ph, Ph, kmax):
    """``a_k = E[relu(m + s t) He_k(t)]``, k = 1 … kmax, as a list of vectors.

    Closed form (see :mod:`whestfloor.relu_moments` for the derivation):
        a_1 = s Phi(alpha)
        a_k = (-1)^k s He_{k-2}(alpha) phi(alpha),  k >= 2
    Cost is O(kmax * width) — negligible beside the layer matmul.
    """
    out = [None, sig * Ph]
    if kmax >= 2:
        s_phi = sig * ph
        h_prev = None  # He_{j-1}
        h = None       # He_j
        for k in range(2, kmax + 1):
            j = k - 2
            if j == 0:
                hj = None  # He_0 == 1
            elif j == 1:
                h_prev, h = None, alpha
                hj = h
            else:
                prev = h_prev if h_prev is not None else None
                base = alpha * h
                hj = base if prev is None else base - float(j - 1) * prev
                h_prev, h = h, hj
            term = s_phi if hj is None else s_phi * hj
            out.append(term if k % 2 == 0 else -term)
    return out


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------


def mean_prop(weights, ctx=None):
    """Diagonal-variance mean propagation (starter-kit example 02 replica)."""
    n = weights[0].shape[0]
    mu = fnp.zeros(n, dtype=fnp.float32)
    var = fnp.ones(n, dtype=fnp.float32)
    rows = []
    for w in weights:
        mu_pre = w.T @ mu
        var_pre = fnp.maximum((w * w).T @ var, 1e-12)
        sig = fnp.sqrt(var_pre)
        mu, var, _, _, _ = _relu_gauss(mu_pre, var_pre, sig)
        rows.append(mu)
    return fnp.stack(rows, axis=0)


def cov_prop_gain(weights, ctx=None):
    """Full covariance propagation with the "gain" off-diagonal rule.

    Exact replica of starter-kit example 03 — the strongest bundled baseline
    and the reference point every improvement in this repo is measured against.
    """
    n = weights[0].shape[0]
    mu = fnp.zeros(n, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
    rows = []
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        mu, var_post, _, _, Ph = _relu_gauss(mu_pre, var_pre, sig)
        cov = fnp.multiply(fnp.outer(Ph, Ph), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        rows.append(mu)
    return fnp.stack(rows, axis=0)


def cov_prop_mehler(weights, ctx=None, kmax: int = 4):
    """Full covariance propagation with the **exact** post-ReLU covariance.

    Replaces the gain rule ``C_ij ≈ Φ_i Φ_j Σ_ij`` — which is only the k=1 term
    of Mehler's expansion — with the convergent series

        C_ij = Σ_{k≥1} a^i_k a^j_k ρ_ij^k / k!

    truncated at ``kmax``.  The added cost is ``O(kmax · n²)`` per layer against
    the layer's ``O(n³)`` einsum, so accuracy here is bought at roughly 2% of
    the layer cost.
    """
    n = weights[0].shape[0]
    mu = fnp.zeros(n, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
    rows = []
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        mu, var_post, alpha, ph, Ph = _relu_gauss(mu_pre, var_pre, sig)

        inv_sig = 1.0 / sig
        rho = cov_pre * fnp.outer(inv_sig, inv_sig)
        rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)

        a = _hermite_coeffs(alpha, sig, ph, Ph, kmax)
        rho_k = rho
        acc = fnp.outer(a[1], a[1]) * rho
        fact = 1.0
        for k in range(2, kmax + 1):
            rho_k = rho_k * rho
            fact *= k
            acc = acc + fnp.outer(a[k], a[k]) * (rho_k * (1.0 / fact))

        cov = acc
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        rows.append(mu)
    return fnp.stack(rows, axis=0)


KERNELS = {
    "mean_prop": mean_prop,
    "cov_prop_gain": cov_prop_gain,
    "cov_prop_mehler": cov_prop_mehler,
}
