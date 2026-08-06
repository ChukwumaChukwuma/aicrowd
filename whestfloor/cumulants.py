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

from .relu_moments import Phi, phi, relu_hermite_coeffs


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


def kappa34_brute(W: np.ndarray, m: np.ndarray, cov: np.ndarray,
                  n_samples: int, seed: int, chunk: int = 4096):
    """Monte-Carlo ``(kappa_3, kappa_4)`` of ``S_j``, for validating the
    diagram catalogue.  ``z`` is drawn Gaussian, which is exactly the
    assumption the catalogue makes, so any gap is truncation, not transport."""
    n = W.shape[0]
    ev, U = np.linalg.eigh(cov)
    L = U * np.sqrt(np.maximum(ev, 0.0))
    rng = np.random.default_rng(seed)
    s1 = np.zeros(W.shape[1])
    s2 = np.zeros(W.shape[1])
    s3 = np.zeros(W.shape[1])
    s4 = np.zeros(W.shape[1])
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        z = rng.standard_normal((nb, n)) @ L.T + m
        S = np.maximum(z, 0.0) @ W
        S2 = S * S
        s1 += S.sum(axis=0)
        s2 += S2.sum(axis=0)
        s3 += (S2 * S).sum(axis=0)
        s4 += (S2 * S2).sum(axis=0)
        done += nb
    m1, m2, m3, m4 = s1 / done, s2 / done, s3 / done, s4 / done
    k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    k4 = (m4 - 4 * m1 * m3 - 3 * m2 * m2 + 12 * m1 * m1 * m2
          - 6 * m1 ** 4)
    return k3, k4


# ---------------------------------------------------------------------------
# The COMPLETE tree catalogue.
#
# ``kappa3_star`` above is the incomplete form: it contracts with ``R`` instead
# of ``rhohat = R - I``, keeps only the ``(1,1,1)`` slot partition, and drops
# the injectivity correction that forces the two leaves onto distinct neurons.
# The functions below implement the full catalogue of
# ``docs/cumulant_expansion.md`` sec 5.2-5.3 -- every slot partition, every
# index-coincidence correction, and (for ``kappa_4``) both Ursell terms paired
# with their graph partners in the same expression.  Cyclic diagrams are the
# only omission; they are 0.5-1.6% and are the only ones that do not factorise.
#
# Both are pinned to the generic diagram engine in
# ``scripts/03_theory_orders.py`` at 1e-16 relative (``scripts/16``).
#
# Truncation: ``K2`` caps the two-block bundles, which are edge-kernel folded
# and therefore cost ONE matmul at any order; ``T3``/``T4`` cap the TOTAL edge
# multiplicity of the three- and four-block diagrams, which cost a matmul per
# order.  ``kappa4_tree`` is exact against the engine for ``T4 <= 3`` (the
# recommended setting); at ``T4 = 4`` it deviates by 4e-5 relative, so a
# high-order injectivity term is still missing there.
# ---------------------------------------------------------------------------

FACT = [1.0]
for _i in range(1, 60):
    FACT.append(FACT[-1] * _i)


def coeffs(m, s, K):
    """a_k (k=0..K), c_k = beta_{2,k}, g_k = beta_{3,k}, and kappa_2..4 of x."""
    al = m / s
    Pa, pa = Phi(al), phi(al)
    a = np.empty((K + 1,) + m.shape)
    a[0] = m * Pa + s * pa
    if K >= 1:
        a[1] = s * Pa
    hp, h = np.ones_like(al), al.copy()          # He_0, He_1
    for k in range(2, K + 1):
        j = k - 2
        if j == 0:
            hj = np.ones_like(al)
        elif j == 1:
            hj = al
        else:
            hp, h = h, al * h - (j - 1) * hp
            hj = h
        a[k] = ((-1.0) ** k) * s * hj * pa
    I = [Pa, al * Pa + pa]
    for p in range(2, 5):
        I.append(al * I[p - 1] + (p - 1) * I[p - 2])
    A2 = np.empty_like(a)
    A3 = np.empty_like(a)
    A2[0] = s ** 2 * I[2]
    A3[0] = s ** 3 * I[3]
    if K >= 1:
        A3[1] = 3 * s ** 3 * I[2]
    for k in range(1, K + 1):
        A2[k] = 2.0 * s * a[k - 1]
    for k in range(2, K + 1):
        A3[k] = 6.0 * s ** 2 * a[k - 2]
    mu = a[0]
    m2, m3, m4 = A2[0], A3[0], s ** 4 * I[4]
    k2 = m2 - mu ** 2
    k3 = m3 - 3 * mu * m2 + 2 * mu ** 3
    k4 = m4 - 4 * mu * m3 - 3 * m2 ** 2 + 12 * mu ** 2 * m2 - 6 * mu ** 4
    c = A2 - 2.0 * mu * a
    g = A3 - 3.0 * mu * A2 + (6.0 * mu ** 2 - 3.0 * m2) * a
    return a, c, g, mu, k2, k3, k4


def epow(R, K):
    n = R.shape[0]
    Rh = R - np.eye(n)
    E = [np.ones((n, n)), Rh]
    for e in range(2, K + 1):
        E.append(E[-1] * Rh)
    return E


def _cs(X):
    return X.sum(axis=0)


def kappa3_tree(W, m, s, R, K2=16, T3=4, *, ctx=None):
    """Complete kappa_3 tree catalogue: (3) + (2,1) + (1,1,1) paths, with the
    exact injectivity (leaf-coincidence) correction.  Cycles dropped."""
    n = W.shape[0]
    if ctx is None:
        K = max(K2, T3) + 1
        a, c, g, mu, k2x, k3x, k4x = coeffs(m, s, K)
        E = epow(R, K)
    else:
        a, c, g, mu, k2x, k3x, k4x, E = ctx
    W1, W2, W3 = W, W * W, W ** 3

    out = _cs(W3 * k3x[:, None])                                   # (3)

    Xi = np.zeros((n, n))                                          # (2,1)
    for e in range(1, K2 + 1):
        Xi += (E[e] / FACT[e]) * np.outer(c[e], a[e])
    out = out + 3.0 * _cs(W2 * (Xi @ W1))

    M = [None] + [E[e] @ (a[e][:, None] * W1) for e in range(1, T3)]
    acc = np.zeros(n)                                              # (1,1,1) free
    for p in range(1, T3):
        for q in range(1, T3 - p + 1):
            acc += _cs((a[p + q][:, None] * W1) * M[p] * M[q]) / (FACT[p] * FACT[q])
    out = out + 3.0 * acc

    Psi = np.zeros((n, n))                                         # leaf coincidence
    for D in range(2, T3 + 1):
        eta = np.zeros(n)
        for p in range(1, D):
            eta += a[p] * a[D - p] / (FACT[p] * FACT[D - p])
        Psi += E[D] * np.outer(a[D], eta)
    out = out - 3.0 * _cs(W1 * (Psi @ W2))
    return out


def kappa4_tree(W, m, s, R, K2=8, T4=3, *, ctx=None):
    """Complete kappa_4 TREE catalogue with exact injectivity + both Ursell
    terms.  ``T4`` caps the total edge multiplicity of the 3- and 4-block
    diagrams; ``K2`` the folded 2-block bundles.  Cycles dropped."""
    n = W.shape[0]
    if ctx is None:
        K = max(K2, T4) + 1
        a, c, g, mu, k2x, k3x, k4x = coeffs(m, s, K)
        E = epow(R, K)
    else:
        a, c, g, mu, k2x, k3x, k4x, E = ctx
    W1, W2, W3, W4 = W, W * W, W ** 3, W ** 4
    F = FACT

    out = _cs(W4 * k4x[:, None])                                       # (4)

    X31 = np.zeros((n, n))                                             # (3,1)
    X22 = np.zeros((n, n))                                             # (2,2)
    Chat = np.zeros((n, n))
    for e in range(1, K2 + 1):
        Ee = E[e] / F[e]
        X31 += Ee * np.outer(g[e], a[e])
        X22 += Ee * np.outer(c[e], c[e])
        Chat += Ee * np.outer(a[e], a[e])
    out = out + 4.0 * _cs(W3 * (X31 @ W1))
    CC = (Chat * Chat) @ W2
    out = out + 3.0 * _cs(W2 * (X22 @ W2)) - 6.0 * _cs(W2 * CC)        # (2,2)+ursell
    out = out - 12.0 * (_cs(W2 * ((Chat @ W1) ** 2)) - _cs(W2 * CC))   # (2,1,1) ursell

    M = [None] + [E[e] @ (a[e][:, None] * W1) for e in range(1, T4)]
    P = [None] + [E[e] @ (c[e][:, None] * W2) for e in range(1, T4)]

    # ---- (2,1,1) paths ------------------------------------------------
    A = np.zeros(n)   # centre = the size-2 block
    B = np.zeros(n)   # centre = a size-1 block  (x2 orientations)
    for p in range(1, T4):
        for q in range(1, T4 - p + 1):
            wpq = 1.0 / (F[p] * F[q])
            A += _cs((c[p + q][:, None] * W2) * M[p] * M[q]) * wpq
            B += _cs((a[p + q][:, None] * W1) * P[p] * M[q]) * wpq
    out = out + 6.0 * A + 12.0 * B
    # ... leaf coincidences, both folded into one matmul each
    Psi2 = np.zeros((n, n))
    V = np.zeros((n, n))
    for D in range(2, T4 + 1):
        eta = np.zeros(n)
        zeta = np.zeros(n)
        for p in range(1, D):
            eta += a[p] * a[D - p] / (F[p] * F[D - p])
            zeta += c[p] * a[D - p] / (F[p] * F[D - p])
        Psi2 += E[D] * np.outer(c[D], eta)
        V += E[D] * np.outer(a[D], zeta)
    out = out - 6.0 * _cs(W2 * (Psi2 @ W2)) - 12.0 * _cs(W1 * (V @ W3))

    # ---- (1,1,1,1) star ------------------------------------------------
    S = np.zeros(n)
    for p in range(1, T4 - 1):
        for q in range(1, T4 - p):
            for r in range(1, T4 - p - q + 1):
                S += _cs((a[p + q + r][:, None] * W1) * M[p] * M[q] * M[r]) / (
                    F[p] * F[q] * F[r])
    out = out + 4.0 * S
    S2 = np.zeros(n)                       # merge two leaves (3 ways, -1 each)
    for D in range(2, T4):
        eta = np.zeros(n)
        for p in range(1, D):
            eta += a[p] * a[D - p] / (F[p] * F[D - p])
        YD = E[D] @ (eta[:, None] * W2)
        for r in range(1, T4 - D + 1):
            S2 += _cs((a[D + r][:, None] * W1) * YD * M[r]) / F[r]
    out = out - 12.0 * S2
    Z = np.zeros((n, n))                   # merge all three leaves (+2)
    for T in range(3, T4 + 1):
        th = np.zeros(n)
        for p in range(1, T - 1):
            for q in range(1, T - p):
                r = T - p - q
                th += a[p] * a[q] * a[r] / (F[p] * F[q] * F[r])
        Z += E[T] * np.outer(a[T], th)
    out = out + 8.0 * _cs(W1 * (Z @ W3))

    # ---- (1,1,1,1) 4-vertex path ---------------------------------------
    Pth = np.zeros(n)
    for q in range(1, T4 - 1):
        Ub = np.zeros((n, n))
        for p in range(1, T4 - q):
            Ub += (a[p + q][:, None] * W1) * M[p] / F[p]
        Pth += _cs(Ub * (E[q] @ Ub)) / F[q]
    out = out + 12.0 * Pth
    inj13 = np.zeros(n)
    inj_both = np.zeros(n)
    for p in range(1, T4 - 1):
        for q in range(1, T4 - p):
            for r in range(1, T4 - p - q + 1):
                wt = 1.0 / (F[p] * F[q] * F[r])
                inj13 += _cs(((a[p] * a[q + r])[:, None] * W2) * M[p + q] * M[r]) * wt
                inj_both += _cs(((a[p] * a[q + r])[:, None] * W2) *
                                (E[p + q + r] @ ((a[p + q] * a[r])[:, None] * W2))) * wt
    out = out - 12.0 * 2.0 * inj13 + 12.0 * inj_both
    return out
