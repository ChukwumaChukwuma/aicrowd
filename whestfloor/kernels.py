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


def _relu_gauss(mu_pre, var_pre, sig, f32: bool = False):
    """Exact E[relu], E[relu^2] and Var[relu] for z ~ N(mu_pre, var_pre).

    ``flops.stats.norm`` promotes its float32 input to float64 to match scipy,
    and float64 is billed at **twice** the float32 rate — which, since every
    downstream array inherits the dtype, doubles the cost of the entire layer.
    ``f32=True`` casts the two transcendentals back; it is off by default so
    the baseline kernels stay byte-identical to their published numbers.
    """
    alpha = mu_pre / sig
    ph = flops.stats.norm.pdf(alpha)
    Ph = flops.stats.norm.cdf(alpha)
    if f32:
        ph = ph.astype(alpha.dtype)
        Ph = Ph.astype(alpha.dtype)
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
        # He_j by ``He_j = alpha He_{j-1} - (j-1) He_{j-2}``.  ``None`` stands
        # for the CONSTANT He_0 = 1, so it must still contribute ``(j-1)*1`` to
        # the recurrence -- treating it as an absent term (the bug this
        # replaces) drops the -1 in He_2 and corrupts every a_k from k = 4 up.
        h_prev = None  # He_{j-1}
        h = None       # He_j
        for k in range(2, kmax + 1):
            j = k - 2
            if j == 0:
                hj = None                      # He_0 == 1
                h_prev, h = None, None
            elif j == 1:
                hj = alpha                     # He_1
                h_prev, h = None, hj           # He_0 == 1 stays implicit
            else:
                base = alpha * h
                hj = (base - float(j - 1) if h_prev is None
                      else base - float(j - 1) * h_prev)
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
        cov = flops.symmetrize(cov, symmetry=(0, 1))
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
        cov = flops.symmetrize(cov, symmetry=(0, 1))
        rows.append(mu)
    return fnp.stack(rows, axis=0)


KERNELS = {
    "mean_prop": mean_prop,
    "cov_prop_gain": cov_prop_gain,
    "cov_prop_mehler": cov_prop_mehler,
}


def _kappa3_star(w, a, rho, umax):
    """Star-diagram third cumulant of ``z^{l}_j = sum_i w_ij relu(z^{l-1}_i)``.

    ``a`` are the previous layer's ReLU Hermite coefficients and ``rho`` its
    pre-activation correlation.  See :mod:`whestfloor.cumulants` for the
    derivation; the whole thing is ``umax`` matmuls of n^3 for all j at once,
    because every diagram with a zero edge multiplicity factorises through the
    rank-one ``R^(0)``.
    """
    G = [None] + [a[m][:, None] * w for m in range(1, 2 * umax + 1)]
    Rp = [None, rho]
    for u in range(2, umax + 1):
        Rp.append(Rp[u - 1] * rho)
    RG = [None] + [Rp[u] @ G[u] for u in range(1, umax + 1)]
    fact = [1.0]
    for i in range(1, 2 * umax + 2):
        fact.append(fact[-1] * i)
    acc = None
    for u in range(1, umax + 1):
        for v in range(1, umax + 1):
            term = fnp.sum(G[u + v] * RG[u] * RG[v], axis=0) * (
                1.0 / (fact[u] * fact[v]))
            acc = term if acc is None else acc + term
    return 3.0 * acc


def cov_prop_edgeworth(weights, ctx=None, kmax: int = 4, umax: int = 2,
                       damp: float = 1.0, g: float = 1.0, f32: bool = True):
    """Mehler covariance propagation plus an analytic third-cumulant correction.

    The Gaussian assumption is the entire error of covariance propagation
    (measured: ~1.3e-3 RMS per layer against a 1.3e-6 budget).  Its leading
    correction is the Edgeworth term in the third cumulant,

        E[relu(z)] = m Phi(a) + s phi(a) - (kappa_3 / 6) (m / s^3) phi(a) + ...

    and ``kappa_3`` is obtained analytically from the previous layer's Hermite
    coefficients and correlation matrix by the star-diagram contraction, which
    costs ``umax`` extra n^3 matmuls per layer.  ``damp`` scales the correction
    (1.0 = full); it exists so the correction's contribution can be ablated
    without changing any other code path.
    """
    n = weights[0].shape[0]
    mu = fnp.zeros(n, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
    prev = None
    rows = []
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        mu, var_post, alpha, ph, Ph = _relu_gauss(mu_pre, var_pre, sig, f32)

        if prev is not None:
            k3 = _kappa3_star(w, prev[0], prev[1], umax)
            mu = mu - (damp / 6.0) * k3 * (mu_pre / (var_pre * sig)) * ph
        if g != 1.0:
            # composes the coherent-bias shrink of cov_prop_shrink with the
            # per-neuron skew correction; they are different mechanisms.
            mu = fnp.maximum(mu * g, 0.0)

        inv_sig = 1.0 / sig
        rho = cov_pre * fnp.outer(inv_sig, inv_sig)
        rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)

        a = _hermite_coeffs(alpha, sig, ph, Ph, max(kmax, 2 * umax))
        rho_k = rho
        acc = fnp.outer(a[1], a[1]) * rho
        fact = 1.0
        for k in range(2, kmax + 1):
            rho_k = rho_k * rho
            fact *= k
            acc = acc + fnp.outer(a[k], a[k]) * (rho_k * (1.0 / fact))

        cov = acc
        fnp.fill_diagonal(cov, var_post)
        cov = flops.symmetrize(cov, symmetry=(0, 1))
        prev = (a, rho)
        rows.append(mu)
    return fnp.stack(rows, axis=0)


KERNELS["cov_prop_edgeworth"] = cov_prop_edgeworth


# --------------------------------------------------------------------------
# Corrected cumulant catalogue.  See docs/cumulant_expansion.md.
#
# ``_kappa3_star`` above is the *incomplete* form: it contracts with ``R``
# rather than ``rhohat = R - I``, and it keeps only the ``(1,1,1)`` slot
# partition with no injectivity correction.  The complete tree catalogue adds
#
#   * the ``(3)`` all-coincident partition,  ``(W^o3)^T kappa_3(x)``;
#   * the ``(2,1)`` bundle, edge-kernel folded into ONE matmul at any order;
#   * the leaf-coincidence (injectivity) correction of the ``(1,1,1)`` path,
#     which is 76% of that diagram;
#
# and uses ``rhohat``, whose zero diagonal is what makes the coincidences
# separable in the first place.  Measured: 98.3-98.8% of the true kappa_3
# against brute force, vs 84-88% for the star form.
# --------------------------------------------------------------------------


def _relu_aux(alpha, sig, ph, Ph, mu_g, kmax, want_g: bool = True):
    """Hermite coefficients, size-2/3 block weights and single-site cumulants.

    Returns ``(a, c, g, k3x, k4x)`` where ``a[k] = E[relu He_k]`` (``a[0]`` is
    the rectified mean), ``c[k] = beta_{2,k} = E[(x-mu)^2 He_k]``,
    ``g[k] = beta_{3,k}``, and ``k3x``/``k4x`` are the third and fourth
    cumulants of ``x = relu(z)``.  All ``O(kmax * n)``.
    """
    a = _hermite_coeffs(alpha, sig, ph, Ph, kmax)
    a[0] = mu_g
    I0 = Ph
    I1 = alpha * Ph + ph
    I2 = alpha * I1 + I0
    I3 = alpha * I2 + 2.0 * I1
    I4 = alpha * I3 + 3.0 * I2
    s2 = sig * sig
    m2 = s2 * I2
    m3 = s2 * sig * I3
    m4 = s2 * s2 * I4
    mu2 = mu_g * mu_g
    k3x = m3 - 3.0 * mu_g * m2 + 2.0 * mu2 * mu_g
    k4x = (m4 - 4.0 * mu_g * m3 - 3.0 * m2 * m2 + 12.0 * mu2 * m2
           - 6.0 * mu2 * mu2)
    # A^(2)_k = 2 s a_{k-1} (k>=1); A^(3)_k = 6 s^2 a_{k-2} (k>=2).
    c = [None]
    g = [None, None]
    gc = 6.0 * mu2 - 3.0 * m2
    for k in range(1, kmax + 1):
        A2k = 2.0 * sig * a[k - 1]
        c.append(A2k - 2.0 * mu_g * a[k])
        if want_g and k >= 2:
            g.append(6.0 * s2 * a[k - 2] - 3.0 * mu_g * A2k + gc * a[k])
    if want_g and kmax >= 1:
        g[1] = (3.0 * s2 * sig * I2 - 6.0 * mu_g * sig * a[0] + gc * a[1])
    return a, c, g, k3x, k4x


def _rhohat_powers(rho, kmax):
    """``E[e] = (R - I)^{oe}`` for ``e = 1..kmax``; ``E[0]`` is unused."""
    n = rho.shape[0]
    rhoh = rho - fnp.eye(n, dtype=rho.dtype)
    E = [None, rhoh]
    for e in range(2, kmax + 1):
        E.append(E[-1] * rhoh)
    return E


_FACT = [1.0]
for _i in range(1, 32):
    _FACT.append(_FACT[-1] * _i)


def _kappa3_tree(w, a, c, k3x, E, K2, T3):
    """Complete tree catalogue for ``kappa_3(z'_j)``, cycles dropped.

    ``T3`` caps the TOTAL edge multiplicity of the three-block diagrams (the
    only ones that cost a matmul per order); ``K2`` caps the two-block bundle,
    which is edge-kernel folded and therefore costs one matmul at any order.
    Matmuls: ``T3 - 1`` for the leaves, plus 2 for the two folded kernels.
    """
    w2 = w * w
    w3 = w2 * w
    out = fnp.sum(w3 * k3x[:, None], axis=0)                       # (3)

    Xi = None                                                      # (2,1) folded
    for e in range(1, K2 + 1):
        t = fnp.outer(c[e] * (1.0 / _FACT[e]), a[e]) * E[e]
        Xi = t if Xi is None else Xi + t
    out = out + 3.0 * fnp.sum(w2 * (Xi @ w), axis=0)

    AW = {}
    for d in range(1, T3 + 1):
        AW[d] = a[d][:, None] * w
    M = [None] + [E[e] @ AW[e] for e in range(1, T3)]
    acc = None                                                     # (1,1,1) path
    for p in range(1, T3):
        for q in range(1, T3 - p + 1):
            t = fnp.sum(AW[p + q] * M[p] * M[q], axis=0) * (
                1.0 / (_FACT[p] * _FACT[q]))
            acc = t if acc is None else acc + t
    out = out + 3.0 * acc

    Psi = None                                                     # leaf coincidence
    for D in range(2, T3 + 1):
        eta = None
        for p in range(1, D):
            t = a[p] * a[D - p] * (1.0 / (_FACT[p] * _FACT[D - p]))
            eta = t if eta is None else eta + t
        t = E[D] * fnp.outer(a[D], eta)
        Psi = t if Psi is None else Psi + t
    out = out - 3.0 * fnp.sum(w * (Psi @ w2), axis=0)
    return out, M


def cov_prop_edge3(weights, ctx=None, kmax: int = 4, K2: int = 6, T3: int = 3,
                   damp: float = 1.0, f32: bool = True):
    """Mehler covariance propagation + the COMPLETE tree-diagram ``kappa_3``.

    Same Edgeworth correction as :func:`cov_prop_edgeworth`, with the third
    cumulant supplied by :func:`_kappa3_tree` instead of the incomplete star
    form.  ``damp`` scales the correction (0 = exact ablation through the
    identical code path).
    """
    n = weights[0].shape[0]
    mu = fnp.zeros(n, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
    kc = max(kmax, K2, T3)
    prev = None
    rows = []
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        mu_g, var_post, alpha, ph, Ph = _relu_gauss(mu_pre, var_pre, sig, f32)
        mu = mu_g

        if prev is not None:
            pa, pc, pk3, pE = prev
            k3, _ = _kappa3_tree(w, pa, pc, pk3, pE, K2, T3)
            mu = mu - (damp / 6.0) * k3 * (mu_pre / (var_pre * sig)) * ph

        inv_sig = 1.0 / sig
        rho = cov_pre * fnp.outer(inv_sig, inv_sig)
        rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)

        a, c, g, k3x, k4x = _relu_aux(alpha, sig, ph, Ph, mu_g, kc,
                                      want_g=False)
        E = _rhohat_powers(rho, kc)

        # Mehler off-diagonal.  ``E[k]`` differs from ``rho^k`` only on the
        # diagonal, which ``fill_diagonal`` overwrites, so the rhohat powers
        # already built for the diagrams are reused here for free.
        acc = None
        for k in range(1, kmax + 1):
            t = fnp.outer(a[k] * (1.0 / _FACT[k]), a[k]) * E[k]
            acc = t if acc is None else acc + t
        cov = acc
        fnp.fill_diagonal(cov, var_post)
        cov = flops.symmetrize(cov, symmetry=(0, 1))
        prev = (a, c, k3x, E)
        rows.append(mu)
    return fnp.stack(rows, axis=0)


KERNELS["cov_prop_edge3"] = cov_prop_edge3


def cov_prop_shrink(weights, ctx=None, kmax: int = 4, g: float = 1.0,
                    f32: bool = True):
    """Mehler covariance propagation with a per-layer multiplicative shrink.

    Motivation, and it is mechanical rather than empirical.  ``scripts/04``
    measured the one-step error of the Gaussian assumption and found it is not
    zero-mean: the mean over neurons is POSITIVE at every single layer
    (+1.34e-3 at layer 2 decaying to +4.96e-4 at layer 32), i.e. the Gaussian
    model systematically OVER-estimates the rectified mean.  A coherent
    multiplicative bias of that kind compounds through the chain, and one
    constant per layer removes it.

    ``g`` is a calibrated constant, not a derived one, so it is only legitimate
    if it transfers.  ``scripts/17_fit_shrink.py`` fits it on one suite and
    scores it on a disjoint one, in both directions.
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
        mu, var_post, alpha, ph, Ph = _relu_gauss(mu_pre, var_pre, sig, f32)
        if g != 1.0:
            # mu is a mean of a ReLU: clip to its feasible range.
            mu = fnp.maximum(mu * g, 0.0)
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
        cov = flops.symmetrize(cov, symmetry=(0, 1))
        rows.append(mu)
    return fnp.stack(rows, axis=0)


KERNELS["cov_prop_shrink"] = cov_prop_shrink


def _kappa4_tree(w, a, c, g, k4x, E, Chat, M, K2, T4):
    """Complete tree catalogue for ``kappa_4(z'_j)``, cycles dropped.

    Both Ursell terms are summed **in the same expression as their graph
    partners**: the ``(2,1,1)`` order-2 tree and its ``-2 C C`` partner are each
    ~100x the total and cancel to four digits (docs sec 9.3), so they must not be
    accumulated separately.  ``M`` is the leaf-matmul list already built for
    ``kappa_3`` and is reused here for free.
    """
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    out = fnp.sum(w4 * k4x[:, None], axis=0)                        # (4)

    X31 = X22 = None                                                # bundles
    for e in range(1, K2 + 1):
        Ee = E[e] * (1.0 / _FACT[e])
        t = Ee * fnp.outer(g[e], a[e])
        X31 = t if X31 is None else X31 + t
        t = Ee * fnp.outer(c[e], c[e])
        X22 = t if X22 is None else X22 + t
    out = out + 4.0 * fnp.sum(w3 * (X31 @ w), axis=0)               # (3,1)
    CC = fnp.sum(w2 * ((Chat * Chat) @ w2), axis=0)                 # shared Ursell
    out = out + 3.0 * fnp.sum(w2 * (X22 @ w2), axis=0) - 6.0 * CC   # (2,2)+ursell
    out = out - 12.0 * (fnp.sum(w2 * ((Chat @ w) ** 2), axis=0) - CC)

    P = [None] + [E[e] @ (c[e][:, None] * w2) for e in range(1, T4)]
    A = B = None                                                    # (2,1,1) paths
    for p in range(1, T4):
        for q in range(1, T4 - p + 1):
            sc = 1.0 / (_FACT[p] * _FACT[q])
            t = fnp.sum((c[p + q][:, None] * w2) * M[p] * M[q], axis=0) * sc
            A = t if A is None else A + t
            t = fnp.sum((a[p + q][:, None] * w) * P[p] * M[q], axis=0) * sc
            B = t if B is None else B + t
    out = out + 6.0 * A + 12.0 * B
    Psi2 = V = None                                                 # leaf coincidences
    for D in range(2, T4 + 1):
        eta = zeta = None
        for p in range(1, D):
            sc = 1.0 / (_FACT[p] * _FACT[D - p])
            t = a[p] * a[D - p] * sc
            eta = t if eta is None else eta + t
            t = c[p] * a[D - p] * sc
            zeta = t if zeta is None else zeta + t
        t = E[D] * fnp.outer(c[D], eta)
        Psi2 = t if Psi2 is None else Psi2 + t
        t = E[D] * fnp.outer(a[D], zeta)
        V = t if V is None else V + t
    out = (out - 6.0 * fnp.sum(w2 * (Psi2 @ w2), axis=0)
           - 12.0 * fnp.sum(w * (V @ w3), axis=0))

    S = None                                                        # (1,1,1,1) star
    for p in range(1, T4 - 1):
        for q in range(1, T4 - p):
            for r in range(1, T4 - p - q + 1):
                t = fnp.sum((a[p + q + r][:, None] * w) * M[p] * M[q] * M[r],
                            axis=0) * (1.0 / (_FACT[p] * _FACT[q] * _FACT[r]))
                S = t if S is None else S + t
    if S is not None:
        out = out + 4.0 * S
    S2 = None                                          # merge two leaves (3, -1 each)
    for D in range(2, T4):
        eta = None
        for p in range(1, D):
            t = a[p] * a[D - p] * (1.0 / (_FACT[p] * _FACT[D - p]))
            eta = t if eta is None else eta + t
        YD = E[D] @ (eta[:, None] * w2)
        for r in range(1, T4 - D + 1):
            t = fnp.sum((a[D + r][:, None] * w) * YD * M[r], axis=0) * (1.0 / _FACT[r])
            S2 = t if S2 is None else S2 + t
    if S2 is not None:
        out = out - 12.0 * S2
    Z = None                                           # merge all three leaves (+2)
    for T in range(3, T4 + 1):
        th = None
        for p in range(1, T - 1):
            for q in range(1, T - p):
                r = T - p - q
                t = a[p] * a[q] * a[r] * (1.0 / (_FACT[p] * _FACT[q] * _FACT[r]))
                th = t if th is None else th + t
        t = E[T] * fnp.outer(a[T], th)
        Z = t if Z is None else Z + t
    if Z is not None:
        out = out + 8.0 * fnp.sum(w * (Z @ w3), axis=0)

    Pth = None                                         # (1,1,1,1) 4-vertex path
    for q in range(1, T4 - 1):
        Ub = None
        for p in range(1, T4 - q):
            t = (a[p + q][:, None] * w) * M[p] * (1.0 / _FACT[p])
            Ub = t if Ub is None else Ub + t
        t = fnp.sum(Ub * (E[q] @ Ub), axis=0) * (1.0 / _FACT[q])
        Pth = t if Pth is None else Pth + t
    if Pth is not None:
        out = out + 12.0 * Pth
    inj = None
    for p in range(1, T4 - 1):
        for q in range(1, T4 - p):
            for r in range(1, T4 - p - q + 1):
                sc = 1.0 / (_FACT[p] * _FACT[q] * _FACT[r])
                v2 = (a[p] * a[q + r])[:, None] * w2
                t = (-2.0) * fnp.sum(v2 * M[p + q] * M[r], axis=0) * sc
                t = t + fnp.sum(
                    v2 * (E[p + q + r] @ ((a[p + q] * a[r])[:, None] * w2)),
                    axis=0) * sc
                inj = t if inj is None else inj + t
    if inj is not None:
        out = out + 12.0 * inj
    return out


def cov_prop_edgeworth4(weights, ctx=None, kmax: int = 4, K2: int = 6,
                        T3: int = 3, T4: int = 3, damp: float = 1.0,
                        damp4: float = 1.0, g1sq: bool = False,
                        f32: bool = True):
    """Mehler propagation + Edgeworth through the FOURTH cumulant.

        E[relu(z)] = m Phi(al) + s phi(al)
                     + s phi(al) [ -g1 al/6 + g2 He_2(al)/24 + g1^2 He_4(al)/72 ]

    with ``g1 = kappa_3/s^3``, ``g2 = kappa_4/s^4``, both from the complete
    tree catalogue.  ``damp``/``damp4`` scale the two corrections independently
    (0 = exact ablation through the identical code path); ``g1sq`` switches the
    ``gamma_1^2`` term on.  ``T4 = 0`` drops the fourth cumulant entirely, which
    reproduces :func:`cov_prop_edge3` exactly.
    """
    n = weights[0].shape[0]
    mu = fnp.zeros(n, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
    kc = max(kmax, K2, T3, T4)
    prev = None
    rows = []
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        mu_g, var_post, alpha, ph, Ph = _relu_gauss(mu_pre, var_pre, sig, f32)
        mu = mu_g

        if prev is not None:
            pa, pc, pg, pk3x, pk4x, pE, pChat = prev
            k3, M = _kappa3_tree(w, pa, pc, pk3x, pE, K2, T3)
            inv3 = 1.0 / (var_pre * sig)
            corr = (-damp / 6.0) * k3 * inv3 * alpha
            if T4:
                k4 = _kappa4_tree(w, pa, pc, pg, pk4x, pE, pChat, M, K2, T4)
                he2 = alpha * alpha - 1.0
                corr = corr + (damp4 / 24.0) * k4 * (inv3 / sig) * he2
            if g1sq:
                a2 = alpha * alpha
                he4 = a2 * a2 - 6.0 * a2 + 3.0
                corr = corr + (damp * damp / 72.0) * (k3 * inv3) ** 2 * he4
            mu = mu + sig * ph * corr

        inv_sig = 1.0 / sig
        rho = cov_pre * fnp.outer(inv_sig, inv_sig)
        rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)

        a, c, g, k3x, k4x = _relu_aux(alpha, sig, ph, Ph, mu_g, kc)
        E = _rhohat_powers(rho, kc)

        acc = Chat = None
        for k in range(1, max(kmax, K2 if T4 else 0) + 1):
            t = fnp.outer(a[k], a[k]) * (E[k] * (1.0 / _FACT[k]))
            if k <= kmax:
                acc = t if acc is None else acc + t
            if T4:
                Chat = t if Chat is None else Chat + t
        cov = acc
        fnp.fill_diagonal(cov, var_post)
        cov = flops.symmetrize(cov, symmetry=(0, 1))
        prev = (a, c, g, k3x, k4x, E, Chat)
        rows.append(mu)
    return fnp.stack(rows, axis=0)


KERNELS["cov_prop_edgeworth4"] = cov_prop_edgeworth4


def mc_kernel(weights, ctx=None, n_samples: int = 5800, seed: int = 0):
    """Plain Monte Carlo.  The baseline that actually matters.

    The grader reports a constant ``sampling_mse = 6.4695e-7`` on every
    submission: because ``mse x C/B`` is flat in N for a sampler, that IS the
    plain-MC adjusted plateau.  Every analytic scheme in this repo -- and the
    bundled covariance-propagation baseline it was measured against -- is
    WORSE than this.  Measuring improvements against covariance propagation
    was measuring against the wrong thing.
    """
    n = weights[0].shape[0]
    rng = fnp.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n), dtype=fnp.float32)
    rows = []
    for w in weights:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def blend_kernel(weights, ctx=None, n_samples: int = 4600, seed: int = 0,
                 wmc: float = 0.70, kmax: int = 4, g: float = 0.999825,
                 damp: float = 0.75, umax: int = 1):
    """Convex blend of Monte Carlo with the analytic estimator.

    They are independent error sources -- MC is unbiased with variance v/N, the
    analytic one is biased with (almost) no variance -- so a convex combination
    beats both.  ``wmc`` is calibrated out of sample.
    """
    a = cov_prop_edgeworth(weights, kmax=kmax, umax=umax, damp=damp, g=g)
    m = mc_kernel(weights, n_samples=n_samples, seed=seed)
    return a + (m - a) * wmc
