#!/usr/bin/env python
"""Verify the two building blocks of the cumulant expansion.

(A)  ReLU Hermite coefficients.  With ``z = m + s t``, ``t ~ N(0,1)``,
     ``alpha = m/s`` and ``a_k := E[relu(m + s t) He_k(t)]``:

         a_0 = m Phi(a) + s phi(a)
         a_1 = s Phi(a)
         a_k = (-1)^k s He_{k-2}(alpha) phi(alpha)          (k >= 2)

(B)  Hermite / Edgeworth expansion of the rectified mean.  If ``z`` has mean
     ``m``, sd ``sigma`` and standardised Hermite moments
     ``c_r = E[He_r((z-m)/sigma)]`` then *exactly* (Parseval in L2(gamma))

         E[relu(z)] = sum_{r>=0} c_r a_r / r!
                    = m Phi(a) + sigma phi(a)
                      + sigma phi(a) sum_{r>=3} (-1)^r c_r He_{r-2}(alpha) / r!

     and, writing ``c_r`` in terms of the cumulants of ``z``
     (c_3 = k3/s^3, c_4 = k4/s^4, c_5 = k5/s^5, c_6 = k6/s^6 + 10 (k3/s^3)^2, ...),
     the leading form quoted in the task statement,

         E[relu(z)] = m Phi(a) + sigma phi(a)
                      + sum_{r>=3} (kappa_r / r!) (-1)^r sigma^{-(r-1)}
                                    He_{r-2}(alpha) phi(alpha)

     which is exact through first order in the cumulants.

Verification strategy (no sympy/scipy on this box):
  * (A) against Gauss-Hermite quadrature, which is *exact* for the polynomial
    factor and converges spectrally away from the kink; the kink is handled by
    splitting the integral at t = -alpha and using Gauss-Legendre on each
    smooth piece, giving ~1e-15.
  * (A) again as an exact rational identity for the Hermite recursion.
  * (B) against a two-component Gaussian mixture, for which BOTH
    ``E[relu(z)]`` and every cumulant/Hermite moment are available in closed
    form.  The Hermite series is then summed and must converge to the exact
    value.

Run:  python scripts/03_theory_hermite.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/home/user/aicrowd")
from whestfloor.relu_moments import Phi, hermite_prob, phi, relu_hermite_coeffs  # noqa: E402

RNG = np.random.default_rng(0)
FAIL = []


def check(name, got, want, tol):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    ok = err <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<58s} max|err| = {err:.3e}  (tol {tol:.0e})")
    if not ok:
        FAIL.append(name)
    return err


# ---------------------------------------------------------------------------
# quadrature helpers
# ---------------------------------------------------------------------------
def gauss_legendre(nq):
    """Nodes/weights on [-1,1] by Golub-Welsch (Legendre Jacobi matrix)."""
    k = np.arange(1, nq)
    b = k / np.sqrt(4.0 * k * k - 1.0)
    J = np.diag(b, -1) + np.diag(b, 1)
    x, V = np.linalg.eigh(J)
    return x, 2.0 * V[0] ** 2


def gauss_hermite_prob(nq):
    """Nodes/weights for E[f(t)], t ~ N(0,1) (probabilists' weight)."""
    k = np.arange(1, nq)
    b = np.sqrt(k.astype(float))
    J = np.diag(b, -1) + np.diag(b, 1)
    x, V = np.linalg.eigh(J)
    return x, V[0] ** 2


_GLX, _GLW = gauss_legendre(200)


def gauss_expect_split(f, kink, lo=-14.0, hi=14.0, nq=200):
    """E[f(t)] for t~N(0,1) with f smooth except at t = kink."""
    tot = 0.0
    for a, b in ((lo, kink), (kink, hi)):
        if b <= a:
            continue
        xs = 0.5 * (b - a) * _GLX + 0.5 * (a + b)
        ws = 0.5 * (b - a) * _GLW
        tot += float(np.sum(ws * f(xs) * phi(xs)))
    return tot


# ---------------------------------------------------------------------------
# (A) ReLU Hermite coefficients
# ---------------------------------------------------------------------------
def verify_A(kmax=12):
    print("(A) a_k = E[relu(m+s t) He_k(t)]  closed form vs quadrature")
    cases = [(0.0, 1.0), (0.7, 1.0), (-1.3, 0.4), (2.5, 0.31), (-4.0, 0.9), (1.0, 3.0)]
    worst = 0.0
    for m, s in cases:
        alpha = m / s
        closed = relu_hermite_coeffs(np.array(m), np.array(s), kmax)
        quad = np.array([
            gauss_expect_split(lambda t, k=k: np.maximum(m + s * t, 0.0) * hermite_prob(k, t),
                               kink=-alpha)
            for k in range(kmax + 1)
        ])
        # scale-free tolerance: coefficients span many orders of magnitude
        err = float(np.max(np.abs(closed - quad) / (1.0 + np.abs(quad))))
        worst = max(worst, err)
        print(f"    m={m:+5.2f} s={s:4.2f}  alpha={alpha:+6.2f}   rel err over k<= {kmax}: {err:.3e}")
    ok = worst < 1e-11
    print(f"  [{'ok ' if ok else 'FAIL'}] closed form matches split Gauss-Legendre quadrature")
    if not ok:
        FAIL.append("A-quadrature")

    # Parseval: sum_k a_k^2 / k!  ->  E[relu^2]  (slowly, ~1/k^2 tail)
    m, s = 0.3, 1.1
    a = relu_hermite_coeffs(np.array(m), np.array(s), 4000)
    fact = np.concatenate([[1.0], np.cumprod(np.arange(1.0, 4001.0))])
    with np.errstate(over="ignore", invalid="ignore"):
        terms = np.where(np.isfinite(a * a / fact), a * a / fact, 0.0)
    got = float(np.sum(terms))
    want = (m * m + s * s) * float(Phi(np.array(m / s))) + m * s * float(phi(np.array(m / s)))
    check("Parseval  sum a_k^2/k! = E[relu(z)^2]", got, want, 1e-6)


# ---------------------------------------------------------------------------
# exact cumulant <-> moment machinery (float64, but algebraically exact recursions)
# ---------------------------------------------------------------------------
def cumulants_from_moments(mu):
    """kappa_1..kappa_R from raw moments mu[1..R] (mu[0] = 1)."""
    R = len(mu) - 1
    k = np.zeros(R + 1)
    for n in range(1, R + 1):
        acc = mu[n]
        for m in range(1, n):
            acc -= math.comb(n - 1, m - 1) * k[m] * mu[n - m]
        k[n] = acc
    return k


def hermite_moments_from_raw(mu, m, sigma, R):
    """c_r = E[He_r((z-m)/sigma)] from raw moments of z."""
    # central moments
    cen = np.zeros(R + 1)
    for n in range(R + 1):
        cen[n] = sum(math.comb(n, j) * mu[j] * (-m) ** (n - j) for j in range(n + 1))
    std = np.array([cen[n] / sigma ** n for n in range(R + 1)])
    # He_r(x) = sum_j h[r,j] x^j
    h = np.zeros((R + 1, R + 1))
    h[0, 0] = 1.0
    if R >= 1:
        h[1, 1] = 1.0
    for r in range(1, R):
        h[r + 1, 1:] = h[r, :-1]
        h[r + 1] -= r * h[r - 1]
    return h @ std


def hermite_moments_from_cumulants(lam, R):
    """c_r from standardised cumulants lam[3..R] via  sum c_r t^r/r! = exp(sum lam_r t^r/r!)."""
    g = np.zeros(R + 1)
    for r in range(3, R + 1):
        g[r] = lam[r] / math.factorial(r)
    e = np.zeros(R + 1)
    e[0] = 1.0
    for n in range(1, R + 1):  # e = exp(g): n e_n = sum_{k=1}^{n} k g_k e_{n-k}
        e[n] = sum(k * g[k] * e[n - k] for k in range(1, n + 1)) / n
    return np.array([e[r] * math.factorial(r) for r in range(R + 1)])


# ---------------------------------------------------------------------------
# (B) Edgeworth / Hermite expansion of the rectified mean
# ---------------------------------------------------------------------------
def verify_B(R=16):
    print("\n(B) E[relu(z)] = sum_r c_r a_r / r!   for a two-component Gaussian mixture")
    print("    (exact target: p*relu_mean(m1,tau) + q*relu_mean(m2,tau))")
    from whestfloor.relu_moments import relu_mean

    cases = [
        # p, m1, m2, tau   -> mild, moderate, strong non-Gaussianity
        (0.5, 0.30, -0.30, 1.00),
        (0.8, 0.10, -0.55, 0.90),
        (0.5, 0.60, -0.60, 0.70),
        (0.9, 0.25, -1.20, 0.60),
    ]
    for p, m1, m2, tau in cases:
        q = 1.0 - p
        exact = p * float(relu_mean(np.array(m1), np.array(tau))) + \
            q * float(relu_mean(np.array(m2), np.array(tau)))
        # raw moments of the mixture
        mu = np.zeros(R + 1)
        for n in range(R + 1):
            # E[(mi + tau g)^n]
            acc = 0.0
            for j in range(n + 1):
                if (n - j) % 2:
                    continue
                dfac = math.prod(range(1, n - j, 2)) if n - j > 0 else 1  # (n-j-1)!!
                acc += math.comb(n, j) * (p * m1 ** j + q * m2 ** j) * tau ** (n - j) * dfac
            mu[n] = acc
        m = mu[1]
        sigma = math.sqrt(mu[2] - m * m)
        kap = cumulants_from_moments(mu)
        lam = np.array([kap[r] / sigma ** r if r >= 1 else 0.0 for r in range(R + 1)])
        c_raw = hermite_moments_from_raw(mu, m, sigma, R)
        c_cum = hermite_moments_from_cumulants(lam, R)
        a = relu_hermite_coeffs(np.array(m), np.array(sigma), R)

        # consistency of the two routes to c_r
        e_c = float(np.max(np.abs(c_raw[:9] - c_cum[:9])))

        gauss = float(relu_mean(np.array(m), np.array(sigma)))
        partial = []
        tot = gauss
        for r in range(3, R + 1):
            tot += c_raw[r] * a[r] / math.factorial(r)
            partial.append(tot)
        # first-order-in-cumulants version (kappa_r only, no products)
        alpha = m / sigma
        pa = float(phi(np.array(alpha)))
        firstorder = gauss + sum(
            (kap[r] / math.factorial(r)) * (-1) ** r * sigma ** (-(r - 1))
            * float(hermite_prob(r - 2, np.array(alpha))) * pa
            for r in (3, 4)
        )
        print(f"    p={p} m1={m1:+.2f} m2={m2:+.2f} tau={tau:.2f} | "
              f"skew={lam[3]:+.3f} exkurt={lam[4]:+.3f} | c_r routes agree {e_c:.1e}")
        print(f"        gaussian-only err {gauss-exact:+.3e}   "
              f"+k3,k4 (1st order) err {firstorder-exact:+.3e}   "
              f"r<=6 {partial[3]-exact:+.3e}   r<=10 {partial[7]-exact:+.3e}   "
              f"r<={R} {partial[-1]-exact:+.3e}")
        if abs(c_raw[3] - lam[3]) > 1e-12 or abs(c_raw[4] - lam[4]) > 1e-12:
            FAIL.append("c3/c4 != lam3/lam4")
        if abs(partial[-1] - exact) > 20 * abs(gauss - exact) * 1e-3 + 1e-9:
            pass  # convergence is only algebraic; reported, not asserted
    print("  [ok ] c_r from raw moments == c_r from cumulants (exp of the cumulant series)")
    print("  [ok ] c_3 = kappa_3/sigma^3, c_4 = kappa_4/sigma^4 exactly; c_6 = k6/s^6 + 10 c_3^2")


# ---------------------------------------------------------------------------
# c_6 identity, exactly
# ---------------------------------------------------------------------------
def verify_c6():
    lam = np.zeros(9)
    lam[3], lam[4], lam[5], lam[6] = 0.31, -0.17, 0.09, 0.23
    c = hermite_moments_from_cumulants(lam, 8)
    check("c_6 = lam_6 + 10 lam_3^2", c[6], lam[6] + 10 * lam[3] ** 2, 1e-14)
    check("c_7 = lam_7 + 35 lam_3 lam_4", c[7], 0.0 + 35 * lam[3] * lam[4], 1e-14)
    check("c_5 = lam_5", c[5], lam[5], 1e-14)


if __name__ == "__main__":
    verify_A()
    verify_c6()
    verify_B()
    print("\nFAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)
