"""Analytic third cumulant of a pre-activation, for a given weight matrix.

The problem.  With ``z ~ N(m, Sigma)`` and ``x = relu(z)``, the next layer's
pre-activation is ``S_j = sum_i W_ij x_i`` and its third cumulant is

    kappa_3(S_j) = sum_{i1,i2,i3} W_{i1 j} W_{i2 j} W_{i3 j}
                                  kappa(x_{i1}, x_{i2}, x_{i3})

— an ``n^3`` contraction for each of ``n`` outputs, i.e. ``n^4`` if taken
literally.  At n = 256 that is 4.3e9 per layer, 5x over the whole free budget.

The diagram expansion.  Expand each ``relu`` in Hermite polynomials of its own
standardised pre-activation, ``x_i = sum_k (a_k^i / k!) He_k(t_i)``, with the
``a_k`` in closed form (see :mod:`whestfloor.relu_moments`).  The classical
Gaussian three-point formula

    E[He_a(t1) He_b(t2) He_c(t3)]
        = a! b! c! * sum_{p+r=a, p+q=b, q+r=c} rho_12^p rho_23^q rho_31^r
                                               / (p! q! r!)

turns the joint cumulant into a sum over triangles with edge multiplicities
``(p, q, r)``, giving

    kappa_3(S_j) = sum_{p,q,r} (1/(p! q! r!))
                   tr[ D_{p+r} R^(p) D_{p+q} R^(q) D_{q+r} R^(r) ]

where ``D_m = diag(W_{:,j} * a_m)`` and ``R^(p)`` is the elementwise p-th power
of the correlation matrix (``R^(0)`` being the all-ones matrix).

The useful structure.  **Whenever one edge multiplicity is zero the trace
factorises**, because ``R^(0)`` is rank one:

    T(u,v) = colsum_j [ G^(u+v) * (R^(u) G^(u)) * (R^(v) G^(v)) ],
    G^(m)  = diag(a_m) W

which is two matmuls and some elementwise work — ``O(n^3)`` for *all* j at
once, not per j.  The three orientations ``(p,q,0)``, ``(p,0,r)``, ``(0,q,r)``
each contribute the same form, so the star total is

    kappa_3^star(S_j) = 3 * sum_{u,v >= 1} T(u,v) / (u! v!)

Triangles with all three multiplicities non-zero do *not* factorise and cost
``O(n^4)``; they are higher order in the correlations and are omitted here.
:func:`validate` measures what fraction of the true kappa_3 the star terms
capture, so the omission is quantified rather than assumed.
"""

from __future__ import annotations

import numpy as np

from .relu_moments import relu_hermite_coeffs


def kappa3_star(W: np.ndarray, m: np.ndarray, s: np.ndarray, R: np.ndarray,
                umax: int = 4) -> np.ndarray:
    """Star-diagram third cumulant of ``S_j = sum_i W_ij relu(z_i)``.

    ``W`` is ``(n, n)`` with the forward convention ``x @ W``; ``m``/``s`` are
    the pre-activation mean and sd; ``R`` the pre-activation correlation.
    Returns ``(n,)``.  Cost: ``umax`` matmuls of ``n^3`` plus ``umax^2`` n^2
    elementwise passes.
    """
    a = relu_hermite_coeffs(m, s, 2 * umax)
    G = [None] + [a[u][:, None] * W for u in range(1, 2 * umax + 1)]
    Rp = [None, R]
    for u in range(2, umax + 1):
        Rp.append(Rp[u - 1] * R)
    RG = [None] + [Rp[u] @ G[u] for u in range(1, umax + 1)]

    fact = [1.0]
    for i in range(1, 2 * umax + 2):
        fact.append(fact[-1] * i)

    out = np.zeros(W.shape[1], dtype=np.float64)
    for u in range(1, umax + 1):
        for v in range(1, umax + 1):
            if u + v > 2 * umax:
                continue
            out += np.sum(G[u + v] * RG[u] * RG[v], axis=0) / (fact[u] * fact[v])
    return 3.0 * out


def kappa3_brute(W: np.ndarray, m: np.ndarray, cov: np.ndarray,
                 n_samples: int, seed: int, chunk: int = 4096) -> np.ndarray:
    """Monte-Carlo kappa_3 of ``S_j``, for validating :func:`kappa3_star`."""
    n = W.shape[0]
    ev, U = np.linalg.eigh(cov)
    L = U * np.sqrt(np.maximum(ev, 0.0))
    rng = np.random.default_rng(seed)
    s1 = np.zeros(W.shape[1])
    s2 = np.zeros(W.shape[1])
    s3 = np.zeros(W.shape[1])
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        z = rng.standard_normal((nb, n)) @ L.T + m
        S = np.maximum(z, 0.0) @ W
        s1 += S.sum(axis=0)
        s2 += (S * S).sum(axis=0)
        s3 += (S * S * S).sum(axis=0)
        done += nb
    m1, m2, m3 = s1 / done, s2 / done, s3 / done
    return m3 - 3 * m1 * m2 + 2 * m1 ** 3
