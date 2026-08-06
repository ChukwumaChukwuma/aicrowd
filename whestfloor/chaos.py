"""Wiener-chaos decomposition of a pre-activation, and the chaos-2 model.

The mechanism
=============

With ``z ~ N(m, Sigma)``, ``t_i = (z_i - m_i)/s_i`` and ``R`` the correlation of
``t``, expand every rectifier exactly in Hermite polynomials of its **own**
standardised pre-activation (coefficients in :mod:`whestfloor.relu_moments`):

    relu(z_i) = sum_k (a_k^i / k!) He_k(t_i)

The next pre-activation then splits exactly by Wiener chaos order::

    S_j = sum_i W_ij relu(z_i)
        = c_j                                        chaos 0   (= E[S_j])
        + sum_i (W_ij a_1^i) t_i                     chaos 1   (exactly Gaussian)
        + (1/2) sum_i (W_ij a_2^i)(t_i^2 - 1)        chaos 2   (a quadratic form)
        + eta_j                                      chaos 3+

Different chaos orders are uncorrelated, so the variances add.  The **chaos-2
model** keeps chaos 0-2 exactly and replaces ``eta_j`` by an independent
Gaussian of the correct variance.  Unlike an Edgeworth series this is a genuine
distribution — chaos 1 + chaos 2 is a generalized chi-square — so it cannot
diverge, and it is not a moment truncation.

Writing ``t = R^{1/2} u`` with ``u ~ N(0,I)``::

    S_j = c_j - tr(A_j) + b_j'u + u'A_j u + eta_j
    A_j = (1/2) R^{1/2} diag(d_j) R^{1/2},  d_j = W[:,j] * a_2
    b_j = R^{1/2} g_j,                      g_j = W[:,j] * a_1

``E[relu]`` of that law is obtained here **exactly** (to quadrature accuracy) by
inverting the characteristic function, written as a correction to the Gaussian
answer so that no cancellation is needed::

    E[relu(S)] = E[relu(G)] + (1/pi) int_0^inf [Re phi_G(th) - Re phi_S(th)]/th^2 dth

with ``G ~ N(c_j, v_j)`` carrying the same first two moments.  The identity is
``|x| = (2/pi) int_0^inf (1 - cos(th x))/th^2 dth`` plus ``relu = (x+|x|)/2``.

Everything here is plain NumPy research code.  It is never shipped; the graded
implementation lives in :mod:`whestfloor.kernels`.
"""

from __future__ import annotations

import numpy as np

from .relu_moments import Phi, phi, relu_mean, relu_var

# ---------------------------------------------------------------------------
# Hermite coefficients, normalised so nothing overflows at high order
# ---------------------------------------------------------------------------


def normalised_hermite_coeffs(m, s, kmax: int) -> np.ndarray:
    """``ahat[k] = a_k / sqrt(k!)`` for ``k = 0 … kmax``.

    ``a_k`` itself overflows float64 around ``k ≈ 170`` because
    ``He_{k-2}(alpha) ~ sqrt((k-2)!)``; the normalised coefficient is bounded by
    ``s phi(alpha) exp(alpha^2/4)`` and the variance decomposition only ever
    needs ``a_k^2/k! = ahat_k^2``.  ``ahat[0]`` is ``a_0 = E[relu]`` unchanged.
    """
    m = np.asarray(m, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    al = m / s
    Pa = Phi(al)
    pa = phi(al)
    out = np.empty((kmax + 1,) + m.shape, dtype=np.float64)
    out[0] = m * Pa + s * pa
    if kmax >= 1:
        out[1] = s * Pa
    if kmax >= 2:
        # htil[j] = He_j(alpha)/sqrt(j!)  by the stable recurrence
        #   htil_{j+1} = (alpha*htil_j - sqrt(j)*htil_{j-1}) / sqrt(j+1)
        htil = [np.ones_like(al), al.copy()]
        for j in range(1, kmax):
            htil.append((al * htil[j] - np.sqrt(j) * htil[j - 1]) / np.sqrt(j + 1.0))
        for k in range(2, kmax + 1):
            out[k] = ((-1.0) ** k) * s * pa * htil[k - 2] / np.sqrt(k * (k - 1.0))
    return out


# ---------------------------------------------------------------------------
# Chaos pieces of S_j = sum_i W_ij relu(z_i)
# ---------------------------------------------------------------------------


def chaos_pieces(W, m, s, R, kmax: int = 120) -> dict:
    """All quantities the chaos-2 model needs, for every output ``j`` at once.

    ``W`` is ``(n, n)`` with the forward convention ``x @ W``.  Returns a dict
    with ``c`` (= E[S]), ``G1``/``G2`` (the chaos-1 and chaos-2 weight columns),
    the chaos-1/2/3+ variances and their total.

    The chaos-3+ variance is ``sum_{k>=3} (1/k!) w_k' R^{ok} w_k`` with
    ``w_k = W[:,j] * a_k``.  It is summed exactly to ``kmax`` and then closed
    with its **exact diagonal tail**, using ``sum_{k>=1} ahat_k^2 = Var(relu)``;
    the omitted off-diagonal tail is ``O(rho^kmax)``.
    """
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[1]
    ah = normalised_hermite_coeffs(m, s, kmax)
    a1 = ah[1]
    a2 = ah[2] * np.sqrt(2.0)

    c = W.T @ ah[0]
    G1 = a1[:, None] * W
    G2 = a2[:, None] * W

    v_l = np.einsum("ij,ij->j", G1, R @ G1)
    R2 = R * R
    v_q = 0.5 * np.einsum("ij,ij->j", G2, R2 @ G2)

    Rk = R2.copy()
    v_eta = np.zeros(n)
    diag_sum = ah[1] ** 2 + ah[2] ** 2
    for k in range(3, kmax + 1):
        Rk *= R
        Gk = ah[k][:, None] * W
        v_eta += np.einsum("ij,ij->j", Gk, Rk @ Gk)
        diag_sum += ah[k] ** 2
    tail = np.maximum(relu_var(m, s) - diag_sum, 0.0)
    v_eta += (W * W).T @ tail

    return {
        "c": c, "G1": G1, "G2": G2, "a1": a1, "a2": a2, "ahat": ah,
        "v_l": v_l, "v_q": v_q, "v_eta": np.maximum(v_eta, 1e-30),
        "v_tot": v_l + v_q + np.maximum(v_eta, 1e-30),
    }


def gauss_relu_mean(pieces) -> np.ndarray:
    """The Gaussian model: match the exact first two moments and stop."""
    return relu_mean(pieces["c"], np.sqrt(pieces["v_tot"]))


# ---------------------------------------------------------------------------
# Exact E[relu] under the chaos-2 model, by characteristic-function inversion
# ---------------------------------------------------------------------------


def _gauss_legendre_panels(hi: float, n_panel: int, order: int):
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, hi, n_panel + 1)
    lo_e, hi_e = edges[:-1, None], edges[1:, None]
    mid, half = 0.5 * (lo_e + hi_e), 0.5 * (hi_e - lo_e)
    return (mid + half * x[None, :]).ravel(), (half * w[None, :]).ravel()


def chaos2_relu_mean(W, m, s, R, pieces=None, *, kmax: int = 120,
                     n_panel: int = 48, order: int = 24,
                     tol: float = 1e-14, report=None) -> np.ndarray:
    """``E[relu(S_j)]`` under the chaos-2 model, for every ``j``.

    Per ``j``: eigendecompose ``R^{1/2} diag(d_j) R^{1/2}`` (so the quadratic
    form becomes a weighted sum of independent non-central chi-squares) and
    integrate the CF-inversion correction on a composite Gauss-Legendre grid.
    Exact up to quadrature error, which is monitored against ``tol``.
    """
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[1]
    if pieces is None:
        pieces = chaos_pieces(W, m, s, R, kmax=kmax)
    ev, U = np.linalg.eigh(R)
    Rh = (U * np.sqrt(np.clip(ev, 0.0, None))) @ U.T

    c, G1, G2 = pieces["c"], pieces["G1"], pieces["G2"]
    v_eta, v_tot = pieces["v_eta"], pieces["v_tot"]
    out = np.empty(n)
    worst_tail = 0.0
    for j in range(n):
        d = G2[:, j]
        B = (Rh * d[None, :]) @ Rh
        mu_e, V = np.linalg.eigh(B)
        lam = 0.5 * mu_e
        beta = V.T @ (Rh @ G1[:, j])
        b2 = beta * beta
        c0 = c[j] - lam.sum()
        ve, vt = v_eta[j], v_tot[j]

        # choose the upper limit so both CFs are below tol there
        hi = 8.0 / np.sqrt(vt)
        for _ in range(40):
            zz = 1.0 - 2j * hi * lam
            mag = np.exp(-0.5 * hi * hi * ve - 0.5 * np.log(np.abs(zz)).sum())
            if max(mag, np.exp(-0.5 * hi * hi * vt)) < tol:
                break
            hi *= 1.35
        worst_tail = max(worst_tail, mag / hi)

        th, wt = _gauss_legendre_panels(hi, n_panel, order)
        zz = 1.0 - 2j * th[:, None] * lam[None, :]
        log_s = (1j * th * c0
                 - 0.5 * np.log(zz).sum(axis=1)
                 - 0.5 * th * th * (b2[None, :] / zz).sum(axis=1)
                 - 0.5 * th * th * ve)
        re_g = np.exp(-0.5 * th * th * vt) * np.cos(th * c[j])
        integ = (re_g - np.exp(log_s).real) / (th * th)
        out[j] = relu_mean(c[j], np.sqrt(vt)) + float(integ @ wt) / np.pi
    if report is not None:
        report["cf_tail_bound"] = worst_tail
    return out


# ---------------------------------------------------------------------------
# Brute-force target with a chaos-2 control variate
# ---------------------------------------------------------------------------

_A5 = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911


def _erf_fast(x):
    """Abramowitz & Stegun 7.1.26.  |err| < 1.5e-7, ~6x faster than the exact
    one in :mod:`whestfloor.backend`, which matters because the control variate
    needs one Phi per sample per neuron.  The residual bias it induces on the
    control variate's mean is measured, not assumed (see :func:`brute_mc`)."""
    ax = np.abs(x)
    t = 1.0 / (1.0 + _P * ax)
    y = 1.0 - (((((_A5[4] * t + _A5[3]) * t) + _A5[2]) * t + _A5[1]) * t
               + _A5[0]) * t * np.exp(-ax * ax)
    return np.sign(x) * y


_S2 = float(np.sqrt(2.0))
_ISP = float(1.0 / np.sqrt(2.0 * np.pi))


def _relu_mean_fast(mu, sd):
    a = mu / sd
    return mu * (0.5 * (1.0 + _erf_fast(a / _S2))) + sd * (_ISP * np.exp(-0.5 * a * a))


def _relu_mean_exact(mu, sd):
    a = mu / sd
    return mu * Phi(a) + sd * phi(a)


def brute_mc(W, m, cov, pieces, n_samples: int, seed: int, *,
             chunk: int = 16384, check_every: int = 64):
    """Monte-Carlo ``E[relu(S_j)]`` for ``z ~ N(m, cov)``, two ways.

    Returns ``(mean_Y, mean_D, bias_fast, n)`` where

    * ``mean_Y`` is the plain estimate of ``E[relu(S_j)]`` (high variance);
    * ``mean_D = E[Y - X]`` with ``X = E_eta[relu(P(u) + eta)]`` the chaos-2
      model's Rao-Blackwellised integrand and ``P`` the exact chaos-0/1/2 part
      of ``S`` evaluated pathwise.  Because ``E[X]`` is *exactly* the chaos-2
      prediction, ``mean_D`` **is** the chaos-2 error, measured directly and
      with ~25x less variance than ``mean_Y``;
    * ``bias_fast`` is ``E[X_fast - X_exact]``, measured on a subsample, so the
      fast-erf approximation is corrected rather than trusted.
    """
    W64 = np.asarray(W, dtype=np.float64)
    n = W64.shape[0]
    s = np.sqrt(np.diag(cov))
    evv, UU = np.linalg.eigh(cov)
    L = (UU * np.sqrt(np.clip(evv, 0.0, None))).astype(np.float32)
    Wf = W64.astype(np.float32)
    G1f = pieces["G1"].astype(np.float32)
    G2h = (0.5 * pieces["G2"]).astype(np.float32)
    cf = pieces["c"].astype(np.float32)
    sef = np.sqrt(pieces["v_eta"]).astype(np.float32)
    mf = m.astype(np.float32)
    sf = s.astype(np.float32)

    rng = np.random.default_rng(seed)
    acc_y = np.zeros(n)
    acc_d = np.zeros(n)
    acc_b = np.zeros(n)
    n_b = 0
    done = 0
    ci = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        t = rng.standard_normal((nb, n), dtype=np.float32)
        z = t @ L.T + mf
        tt = (z - mf) / sf
        y = np.maximum(z, np.float32(0.0)) @ Wf
        np.maximum(y, np.float32(0.0), out=y)          # Y = relu(S)
        p = cf + tt @ G1f + (tt * tt - np.float32(1.0)) @ G2h
        x = _relu_mean_fast(p, sef)
        acc_y += y.sum(axis=0, dtype=np.float64)
        acc_d += (y - x).sum(axis=0, dtype=np.float64)
        if ci % check_every == 0:
            xe = _relu_mean_exact(p.astype(np.float64), sef.astype(np.float64))
            acc_b += (x - xe).sum(axis=0, dtype=np.float64)
            n_b += nb
        ci += 1
        done += nb
    return acc_y / done, acc_d / done, acc_b / max(n_b, 1), done


def layer_state(W, upto: int, n_samples: int, seed: int, chunk: int = 2048):
    """Measured ``(m, cov)`` of ``z`` at layer ``upto`` (0-based) of a real MLP.

    Same construction as ``scripts/13_validate_kappa3.py``: the correlation the
    network actually produces, not a synthetic one.
    """
    n = W[0].shape[0]
    rng = np.random.default_rng(seed)
    s1 = np.zeros(n)
    g = np.zeros((n, n))
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        for li, w in enumerate(W):
            z = x @ w
            if li == upto:
                zf = z.astype(np.float64)
                s1 += zf.sum(axis=0)
                g += zf.T @ zf
                break
            x = np.maximum(z, np.float32(0.0))
        done += nb
    m = s1 / done
    return m, g / done - np.outer(m, m)
