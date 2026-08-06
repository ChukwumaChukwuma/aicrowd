"""The conditional-independence CGF model for ``S_j = sum_i W_ij relu(z_i)``.

For a single rectified Gaussian there is an exact CGF::

    E[exp(c relu(Y))] = Phi(-mu/sqrt(d))
                      + exp(c mu + c^2 d/2) Phi(mu/sqrt(d) + c sqrt(d)),
    Y ~ N(mu, d)

so **if the pre-activations were independent** the CGF of ``S_j`` would be the
exact, everywhere-valid function ``K_j(th) = sum_i log[...]`` with ``c = th W_ij``
— not a truncated series, so a saddlepoint on it cannot diverge the way an
Edgeworth expansion does.

They are not independent.  The construction tested here uses the exact mean and
variance from the *full* covariance and lets the independence CGF supply only
the **shape**::

    K~(th) = a th + K_ind(lam th),
    lam = sqrt(v_exact / v_ind),   a = m_exact - lam K_ind'(0)

Under that affine renormalisation only the *standardised* cumulants of the
independence model survive, so :func:`indep_std_cumulants` is the whole test of
whether its shape is right, and it costs two ``n^2`` contractions.

:func:`indep_relu_mean_fft` evaluates ``E[relu]`` of the renormalised model
*exactly* (to grid resolution) by FFT convolution, which separates the question
"is the model right?" from "is the saddlepoint accurate?".  Research code, plain
NumPy; never shipped.
"""

from __future__ import annotations

import numpy as np

from .relu_moments import Phi, phi


def relu_raw_moments(m, s, pmax: int = 4) -> list:
    """``M_p = E[relu(Y)^p]`` for ``Y ~ N(m, s^2)``, ``p = 0 … pmax``.

    ``M_0 = Phi(alpha)``, ``M_1 = m Phi + s phi`` and, by Stein's identity,
    ``M_p = m M_{p-1} + (p-1) s^2 M_{p-2}`` for ``p >= 2`` (the boundary term
    vanishes because the integrand has a zero of order ``p-1`` at the kink).
    """
    m = np.asarray(m, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    al = m / s
    M = [Phi(al), m * Phi(al) + s * phi(al)]
    for p in range(2, pmax + 1):
        M.append(m * M[p - 1] + (p - 1) * s * s * M[p - 2])
    return M


def relu_cumulants(m, s):
    """``(mean, k2, k3, k4)`` of ``relu(Y)``, ``Y ~ N(m, s^2)``.  Exact."""
    M = relu_raw_moments(m, s, 4)
    m1 = M[1]
    k2 = M[2] - m1**2
    k3 = M[3] - 3 * m1 * M[2] + 2 * m1**3
    k4 = (M[4] - 4 * m1 * M[3] - 3 * M[2]**2 + 12 * m1**2 * M[2] - 6 * m1**4)
    return m1, k2, k3, k4


def indep_std_cumulants(W, m, s):
    """``(gamma1, gamma2, k2_ind)`` of the independence model, per output ``j``.

    Cumulants of a sum of independents add, so ``k_r = sum_i W_ij^r k_r(relu_i)``
    — one ``n^2`` contraction per order.  ``gamma1``/``gamma2`` are invariant
    under the affine renormalisation and are therefore the model's entire claim
    about the shape of ``S_j``.
    """
    W = np.asarray(W, dtype=np.float64)
    _, c2, c3, c4 = relu_cumulants(m, s)
    W2 = W * W
    k2 = W2.T @ c2
    k3 = (W2 * W).T @ c3
    k4 = (W2 * W2).T @ c4
    return k3 / k2**1.5, k4 / k2**2, k2


def indep_relu_mean_fft(W, m, s, c_exact, v_exact, *, log2m: int = 18,
                        span_sd: float = 14.0, jobs=None) -> np.ndarray:
    """``E[relu(S_j)]`` under the affinely renormalised independence model.

    Exact to grid resolution: each summand ``lam W_ij relu(z_i)`` is discretised
    by *exact CDF differences* on a common grid (which places the atom at zero
    exactly), the laws are convolved by FFT, and ``E[relu]`` is read off.  The
    only error is the O(dx^2) rounding bias, ~1e-8 at the default resolution.
    """
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[1]
    M = 1 << log2m
    _, c2, _, _ = relu_cumulants(m, s)
    k2i = (W * W).T @ c2
    lam = np.sqrt(v_exact / k2i)
    mu_i = relu_raw_moments(m, s, 1)[1]
    a = c_exact - lam * (W.T @ mu_i)

    out = np.empty(n)
    idx = range(n) if jobs is None else jobs
    for j in idx:
        cvec = lam[j] * W[:, j]
        sigma = np.sqrt(v_exact[j])
        half = span_sd * sigma + np.max(np.abs(cvec) * (np.abs(m) + 6.0 * s))
        dx = 2.0 * half / M
        k = np.arange(M) - M // 2
        edge_lo = (k - 0.5) * dx
        edge_hi = (k + 0.5) * dx
        acc = None
        for i in range(len(cvec)):
            ci = cvec[i]
            if ci == 0.0:
                continue
            if ci > 0.0:
                cdf_lo = np.where(edge_lo >= 0.0, Phi((edge_lo / ci - m[i]) / s[i]), 0.0)
                cdf_hi = np.where(edge_hi >= 0.0, Phi((edge_hi / ci - m[i]) / s[i]), 0.0)
            else:
                cdf_lo = np.where(edge_lo >= 0.0, 1.0,
                                  1.0 - Phi((edge_lo / ci - m[i]) / s[i]))
                cdf_hi = np.where(edge_hi >= 0.0, 1.0,
                                  1.0 - Phi((edge_hi / ci - m[i]) / s[i]))
            p = np.maximum(cdf_hi - cdf_lo, 0.0)
            p /= p.sum()
            F = np.fft.rfft(np.fft.ifftshift(p))
            acc = F if acc is None else acc * F
        v = np.fft.fftshift(np.fft.irfft(acc, M))
        v = np.maximum(v, 0.0)
        v /= v.sum()
        x = k * dx
        out[j] = float(np.maximum(a[j] + x, 0.0) @ v)
    return out
