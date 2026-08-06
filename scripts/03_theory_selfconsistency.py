#!/usr/bin/env python
"""Self-consistency: what a NON-Gaussian z^l does to E[x^l] and Cov(x^l).

FIRST-ORDER MULTIVARIATE EDGEWORTH
==================================
Let z have mean m, covariance Sigma and joint cumulants kappa_{i1..ir} (r>=3),
and let G ~ N(m, Sigma).  From

    E[e^{th.z}] = exp( th.m + th'Sigma th/2 + sum_{r>=3} kappa_{i1..ir}
                       th_i1..th_ir / r! )

and the substitution th_i <-> d/dz_i acting on the Gaussian law,

    E[F(z)] = E[ (1 + sum_{r>=3} kappa_{i1..ir} d_i1..d_ir / r!) F(G) ]
              + O(kappa^2).                                            (*)

THE THREE CONSEQUENCES USED BY THE RECURSION
============================================
Write a_k^i = E[relu(m_i + sigma_i t) He_k(t)] (the ReLU Hermite coefficients),
alpha_i = m_i/sigma_i, rho_ik the correlation.  The single identity

    E[ relu^{(k)}(G_i) ] = a_k^i / sigma_i^k      for every k >= 0

(k=0 trivially, k=1 gives Phi, k>=2 gives the delta-derivative form) collapses
everything:

  (1) MEAN
        Delta E[x_i] = sum_{r>=3} kappa_r(z_i) / r! * a_r^i / sigma_i^r
      which is exactly the Edgeworth rectified mean (formula (B)).

  (2) VARIANCE   (uses (relu^2)' = 2 relu, so (relu^2)^{(r)} = 2 relu^{(r-1)})
        Delta E[x_i^2]  = sum_{r>=3} kappa_r(z_i) / r! * 2 a_{r-1}^i / sigma_i^{r-1}
        Delta Var(x_i)  = Delta E[x_i^2] - 2 mu_i Delta E[x_i]

  (3) COVARIANCE  (i != k).  With the *shifted Mehler* kernel

        D_{a,b}[i,k] = sigma_i^{-a} sigma_k^{-b}
                       sum_{p>=0} rhohat_ik^p / p! * a_{a+p}^i a_{b+p}^k

      (D_{0,0} is the ordinary Mehler post-ReLU second moment), one gets

        Delta Cov(x_i,x_k) =
            (1/6)  [ k_iii Dt_{3,0} + k_kkk Dt_{0,3}
                     + 3 k_iik D_{2,1} + 3 k_ikk D_{1,2} ]
          + (1/24) [ k_iiii Dt_{4,0} + k_kkkk Dt_{0,4}
                     + 4 k_iiik D_{3,1} + 6 k_iikk D_{2,2} + 4 k_ikkk D_{1,3} ]

      where Dt_{r,0} = D_{r,0} - a_r^i a_0^k / sigma_i^r drops the p=0 term
      (that term is cancelled by the -mu_k Delta E[x_i] of the covariance), and
      likewise Dt_{0,r}.

COST
====
Every D_{a,b} is O(K n^2) -- no matmul.  The only new matmuls are the mixed
cumulant matrices themselves; at leading (coincident-block) order

    kappa_{i^a k^b}(z) = ( W^{oa} )^T diag( kappa^x_{a+b} ) W^{ob} + O(rhohat),

one matmul for each of (a,b) = (2,1), (3,1), (2,2)  --  three per layer.

VERIFICATION
============
z = m + A y with y_a i.i.d. standardised shifted-Gamma(k):  every joint cumulant
of z is then known in closed form, and letting k -> infinity drives the
non-Gaussianity to zero at a known rate, so the O(kappa^2) residual of (*) can
be seen to scale correctly.

Run:  python scripts/03_theory_selfconsistency.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/home/user/aicrowd")
from whestfloor.relu_moments import relu_hermite_coeffs, relu_mean, relu_var  # noqa: E402

FAIL: list[str] = []


def shifted_mehler(a, s, rhohat, amax, K):
    """D[a,b] for 0 <= a,b <= amax as an (amax+1, amax+1) list of (n,n) arrays."""
    n = rhohat.shape[0]
    D = [[None] * (amax + 1) for _ in range(amax + 1)]
    powr = [np.ones((n, n))]
    for p in range(1, K + 1):
        powr.append(powr[-1] * rhohat)
    for ai in range(amax + 1):
        for bi in range(amax + 1):
            acc = np.zeros((n, n))
            for p in range(K + 1):
                acc += (powr[p] / math.factorial(p)) * np.outer(a[ai + p], a[bi + p])
            D[ai][bi] = acc / np.outer(s ** ai, s ** bi)
    return D


def nongauss_corrections(m, s, rhohat, K3, K4, kmax=12):
    """Returns (dmu, dvar, dcov).  K3/K4 are dicts of cumulant arrays:
    K3['iii'] (n,), K3['iik'] (n,n) = kappa(z_i,z_i,z_k);
    K4['iiii'] (n,), K4['iiik'] (n,n), K4['iikk'] (n,n)."""
    n = m.shape[0]
    a = relu_hermite_coeffs(m, s, kmax + 6)
    mu = a[0]
    dmu = K3["iii"] / 6.0 * a[3] / s ** 3 + K4["iiii"] / 24.0 * a[4] / s ** 4
    dm2 = K3["iii"] / 6.0 * 2 * a[2] / s ** 2 + K4["iiii"] / 24.0 * 2 * a[3] / s ** 3
    dvar = dm2 - 2 * mu * dmu
    D = shifted_mehler(a, s, rhohat, 4, kmax)
    o = np.outer
    Dt30 = D[3][0] - o(a[3] / s ** 3, a[0])
    Dt03 = D[0][3] - o(a[0], a[3] / s ** 3)
    Dt40 = D[4][0] - o(a[4] / s ** 4, a[0])
    Dt04 = D[0][4] - o(a[0], a[4] / s ** 4)
    k3i = K3["iii"][:, None] * np.ones((1, n))
    k3k = np.ones((n, 1)) * K3["iii"][None, :]
    k4i = K4["iiii"][:, None] * np.ones((1, n))
    k4k = np.ones((n, 1)) * K4["iiii"][None, :]
    dcov = (k3i * Dt30 + k3k * Dt03
            + 3 * K3["iik"] * D[2][1] + 3 * K3["iik"].T * D[1][2]) / 6.0
    dcov += (k4i * Dt40 + k4k * Dt04
             + 4 * K4["iiik"] * D[3][1] + 6 * K4["iikk"] * D[2][2]
             + 4 * K4["iiik"].T * D[1][3]) / 24.0
    np.fill_diagonal(dcov, dvar)
    return dmu, dvar, dcov


# ---------------------------------------------------------------------------
def run(n=24, kgam=6.0, nsamp=40_000_000, seed=4, kmax=14):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / math.sqrt(n) * 1.1
    m = 0.6 * rng.standard_normal(n)
    Sig = A @ A.T
    s = np.sqrt(np.diag(Sig))
    rho = Sig / np.outer(s, s)
    rhohat = rho - np.eye(n)

    k3y, k4y = 2.0 / math.sqrt(kgam), 6.0 / kgam
    P3 = A ** 3
    P4 = A ** 4
    K3 = {"iii": k3y * P3.sum(1), "iik": k3y * (P3 * 0 + (A * A) @ A.T * 0)}
    # kappa(z_i,z_i,z_k) = k3y sum_a A_ia^2 A_ka
    K3["iik"] = k3y * ((A * A) @ A.T)
    K4 = {"iiii": k4y * P4.sum(1),
          "iiik": k4y * ((A ** 3) @ A.T),
          "iikk": k4y * ((A * A) @ (A * A).T)}

    # ---- Monte Carlo on the true non-Gaussian z ----
    S1 = np.zeros(n); S2 = np.zeros((n, n))
    chunk, done = 400_000, 0
    while done < nsamp:
        nb = min(chunk, nsamp - done)
        y = (rng.gamma(kgam, 1.0, size=(nb, n)) - kgam) / math.sqrt(kgam)
        z = m + y @ A.T
        x = np.maximum(z, 0.0)
        S1 += x.sum(0)
        S2 += x.T @ x
        done += nb
    S1 /= done
    S2 /= done
    Ctrue = S2 - np.outer(S1, S1)

    mu_g = relu_mean(m, s)
    a = relu_hermite_coeffs(m, s, kmax + 6)
    Cg = np.zeros((n, n))
    powr = np.ones((n, n))
    for p in range(1, kmax + 1):
        powr = powr * rhohat
        Cg += (powr / math.factorial(p)) * np.outer(a[p], a[p])
    np.fill_diagonal(Cg, relu_var(m, s))

    dmu, dvar, dcov = nongauss_corrections(m, s, rhohat, K3, K4, kmax)
    off = ~np.eye(n, dtype=bool)
    se = math.sqrt(1.0 / done)
    print(f"  gamma shape k={kgam:5.1f}  (skew_y={k3y:.3f}, exkurt_y={k4y:.3f})   "
          f"N={done:,}")
    e0 = np.sqrt(((mu_g - S1) ** 2).mean())
    e1 = np.sqrt(((mu_g + dmu - S1) ** 2).mean())
    print(f"    E[x]      gaussian {e0:.4e}   +1st order {e1:.4e}   gain {e0/e1:5.1f}x")
    d0 = np.sqrt(((np.diag(Cg) - np.diag(Ctrue)) ** 2).mean())
    d1 = np.sqrt(((np.diag(Cg) + dvar - np.diag(Ctrue)) ** 2).mean())
    print(f"    Var(x)    gaussian {d0:.4e}   +1st order {d1:.4e}   gain {d0/d1:5.1f}x")
    c0 = np.sqrt(((Cg - Ctrue)[off] ** 2).mean())
    c1 = np.sqrt(((Cg + dcov - Ctrue)[off] ** 2).mean())
    print(f"    Cov(x)off gaussian {c0:.4e}   +1st order {c1:.4e}   gain {c0/c1:5.1f}x"
          f"   (mc se ~{se:.1e})")
    return e0, e1, c0, c1


if __name__ == "__main__":
    print("first-order Edgeworth corrections to E[x], Var(x), Cov(x)")
    print("(residual must fall like 1/k while the correction itself falls like 1/sqrt(k))")
    prev = None
    for kg in (3.0, 6.0, 12.0, 24.0):
        r = run(kgam=kg)
        if prev is not None:
            print(f"      -> uncorrected shrank {prev[0]/r[0]:.2f}x (expect ~1.41), "
                  f"corrected shrank {prev[1]/r[1]:.2f}x (expect ~2)")
        prev = r
    print("\nFAILURES:", FAIL if FAIL else "none")
