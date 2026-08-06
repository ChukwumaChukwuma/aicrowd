"""Exact Gaussian ReLU moments and the Hermite coefficients of the rectifier.

The single identity everything in this repository is built on
=============================================================

Let ``z = m + s t`` with ``t ~ N(0,1)`` and ``α = m/s``.  Define the Hermite
coefficients of the rectifier about that neuron's own mean and scale:

    a_k := E[ relu(m + s t) · He_k(t) ]

Then, in closed form,

    a_0 = m Φ(α) + s φ(α)
    a_1 = s Φ(α)
    a_k = (-1)^k · s · He_{k-2}(α) · φ(α)          for k ≥ 2

*Derivation.*  For k ≥ 1, Stein's identity gives ``E[f(t) He_k(t)] =
E[f^{(k)}(t)]``.  With ``f(t) = relu(m+st)`` we have ``f'(t) = s·1{m+st>0}``
and ``f^{(k)}(t) = s^k δ^{(k-2)}(m+st)`` for ``k ≥ 2``.  Substituting
``u = m+st`` and using ``φ^{(j)}(x) = (-1)^j He_j(x) φ(x)`` together with
``He_j(-α) = (-1)^j He_j(α)`` collapses to the line above.

Why it matters
--------------

Mehler's formula turns those coefficients into the **exact** post-ReLU
covariance of two jointly Gaussian pre-activations with correlation ρ:

    Cov(relu(z_i), relu(z_j)) = Σ_{k≥1} a^i_k a^j_k ρ^k / k!
                              = Φ_i Φ_j Σ_ij
                                + s_i s_j φ_i φ_j Σ_{k≥2} He_{k-2}(α_i) He_{k-2}(α_j) ρ^k / k!

The first term is precisely the "gain" approximation used by the starter kit's
covariance-propagation baseline.  Everything after it is a correction the
baseline drops — obtainable for a few n² FLOPs against the n³ cost of the layer
matmul, i.e. essentially free.

*Verification.*  At ``m = 0`` the series must reproduce the arc-cosine kernel
``Cov = (s²/2π)[√(1-ρ²) - 1 + ρπ/2 + ρ arcsin ρ]``.  Expanding that gives
``s²[ρ/4 + ρ²/(4π) + ρ⁴/(48π) + ρ⁶/(160π) + …]`` and the series above gives
``s²[ρ/4 + ρ²/(4π) + ρ⁴/(48π) + 9ρ⁶/(1440π) + …]`` with ``9/1440 = 1/160``.
They agree term by term; ``tests/test_relu_moments.py`` pins this numerically
against adaptive quadrature.
"""

from __future__ import annotations

import numpy as np

SQRT_2 = float(np.sqrt(2.0))
INV_SQRT_2PI = float(1.0 / np.sqrt(2.0 * np.pi))


def phi(x: np.ndarray) -> np.ndarray:
    """Standard normal pdf."""
    return INV_SQRT_2PI * np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2)


def Phi(x: np.ndarray) -> np.ndarray:  # noqa: N802 - matches the maths
    """Standard normal cdf, via the erf in :mod:`whestfloor.backend`."""
    from .backend import _erf_np  # noqa: PLC0415

    return 0.5 * (1.0 + _erf_np(np.asarray(x, dtype=np.float64) / SQRT_2))


def relu_mean(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E[relu(z)] for z ~ N(m, s²).  Exact."""
    a = m / s
    return m * Phi(a) + s * phi(a)


def relu_second_moment(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E[relu(z)²] for z ~ N(m, s²).  Exact."""
    a = m / s
    return (m * m + s * s) * Phi(a) + m * s * phi(a)


def relu_var(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Var[relu(z)] for z ~ N(m, s²).  Exact."""
    mu = relu_mean(m, s)
    return np.maximum(relu_second_moment(m, s) - mu * mu, 0.0)


def hermite_prob(order: int, x: np.ndarray) -> np.ndarray:
    """Probabilists' Hermite polynomial He_order(x), by recurrence."""
    x = np.asarray(x, dtype=np.float64)
    if order == 0:
        return np.ones_like(x)
    h_prev = np.ones_like(x)
    h = x.copy()
    for n in range(1, order):
        h_prev, h = h, x * h - n * h_prev
    return h


def relu_hermite_coeffs(m: np.ndarray, s: np.ndarray, kmax: int) -> np.ndarray:
    """Return ``a[k]`` for ``k = 0 … kmax``, shape ``(kmax+1,) + m.shape``.

    ``a_k = E[relu(m + s t) He_k(t)]`` in the closed form derived above.
    """
    m = np.asarray(m, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    alpha = m / s
    Pa = Phi(alpha)
    pa = phi(alpha)
    out = np.empty((kmax + 1,) + m.shape, dtype=np.float64)
    out[0] = m * Pa + s * pa
    if kmax >= 1:
        out[1] = s * Pa
    # He_{k-2}(alpha) by the same recurrence, tracked in place.
    if kmax >= 2:
        h_prev = np.ones_like(alpha)       # He_0
        h = alpha.copy()                   # He_1
        out[2] = s * h_prev * pa           # k=2 -> (+1) * s * He_0 * phi
        for k in range(3, kmax + 1):
            j = k - 2                      # need He_j
            if j == 1:
                hj = h
            else:
                h_prev, h = h, alpha * h - (j - 1) * h_prev
                hj = h
            out[k] = ((-1.0) ** k) * s * hj * pa
    return out


def relu_cov_mehler(
    Sigma: np.ndarray,
    m: np.ndarray,
    s: np.ndarray,
    kmax: int = 8,
    *,
    coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Post-ReLU covariance via Mehler's formula, truncated at ``kmax``.

    ``Sigma`` is the pre-activation covariance, ``m`` its mean, ``s`` the
    square root of its diagonal.  The diagonal of the result is overwritten
    with the exact ``Var[relu]``, which the series only reaches asymptotically
    at ρ = 1.

    Cost: one ``n×n`` elementwise power per order plus one outer product per
    order — ``O(kmax · n²)`` against the layer's ``O(n³)`` matmul.
    """
    n = Sigma.shape[0]
    if coeffs is None:
        coeffs = relu_hermite_coeffs(m, s, kmax)
    rho = Sigma / np.outer(s, s)
    np.clip(rho, -1.0, 1.0, out=rho)

    out = np.zeros((n, n), dtype=np.float64)
    rho_k = np.ones_like(rho)
    fact = 1.0
    for k in range(1, kmax + 1):
        rho_k *= rho
        fact *= k
        out += np.outer(coeffs[k], coeffs[k]) * (rho_k / fact)
    np.fill_diagonal(out, relu_var(m, s))
    return out


def relu_cov_exact_centered(Sigma: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Arc-cosine kernel: exact post-ReLU covariance when all means are zero.

    Used only as a test oracle for :func:`relu_cov_mehler`.
    """
    rho = np.clip(Sigma / np.outer(s, s), -1.0, 1.0)
    ss = np.outer(s, s)
    second = (ss / (2.0 * np.pi)) * (
        np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) + rho * (np.pi / 2.0 + np.arcsin(rho))
    )
    mu = s * INV_SQRT_2PI
    return second - np.outer(mu, mu)
