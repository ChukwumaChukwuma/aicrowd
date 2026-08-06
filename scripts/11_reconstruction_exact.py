#!/usr/bin/env python
"""Noise-free bake-off: can ANY reconstruction recover E[relu] from cumulants?

``scripts/10`` tried to answer this against Monte-Carlo cumulants and failed
methodologically: the prediction was built from both reference halves, so it
correlated with both, the cross-product estimator went negative, and a
``max(mse, 0)`` clamp turned that into a column of exact zeros.  A column of
exact zeros is not a result.  The lesson is that the question should not
involve Monte Carlo at all.

It doesn't have to.  The reconstruction step is a pure map

    (kappa_1, ..., kappa_r)  ->  E[relu(X)]

and it can be tested on any distribution whose ``E[relu]`` and cumulants are
both known exactly.  So this script builds test distributions as sums of
independent rectified Gaussians plus a Gaussian — the exact shape that arises
one layer into the network — computes their densities by FFT convolution on a
fine grid (accurate to ~1e-12), and reads off both the exact ``E[relu(X)]`` and
the exact cumulants.  No sampling, no noise, no clamping.

The test population is filtered to the regime the network actually occupies at
depth, measured in scripts/04: skewness ~0.44, excess kurtosis ~0.40, and
alpha = mean/sd spread over the range the neurons span.

Verdict rule fixed before the run: a reconstruction is viable if its RMS error
over the population is below the per-layer budget of 2.37e-6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.relu_moments import Phi, hermite_prob, phi  # noqa: E402

GRID_N = 1 << 21
PER_LAYER_BAR = 1.34e-5 / (32 ** 0.5)


def relu_gauss(m, s):
    a = m / s
    return m * Phi(a) + s * phi(a)


# --------------------------------------------------------------------------
# Exact target construction
# --------------------------------------------------------------------------
def rectified_gaussian_pdf(grid, m, s):
    """Density of relu(N(m, s^2)) on `grid`, with the atom at 0 returned apart."""
    p = np.where(grid > 0, phi((grid - m) / s) / s, 0.0)
    atom = float(Phi(-m / s))
    return p, atom


def build_exact(rng, K, span, hi):
    """One test distribution: sum_i c_i relu(N(m_i,s_i^2)) + N(mu0, tau^2).

    Returns (grid, pdf, exact_E_relu, cumulants[1..6]).
    """
    dx = span / GRID_N
    grid = (np.arange(GRID_N) - GRID_N // 2) * dx

    # start from the Gaussian part
    tau = float(rng.uniform(0.15, 0.6))
    mu0 = float(rng.uniform(-hi, hi))
    f = np.exp(-0.5 * ((grid - mu0) / tau) ** 2) / (tau * np.sqrt(2 * np.pi))
    f /= f.sum() * dx

    F = np.fft.rfft(np.fft.ifftshift(f))
    for _ in range(K):
        c = float(rng.uniform(0.25, 1.0)) * (1.0 if rng.random() < 0.75 else -1.0)
        m = float(rng.normal(0.0, 1.0))
        s = float(rng.uniform(0.4, 1.3))
        g = np.zeros_like(grid)
        pos = grid / c if c > 0 else grid / c
        # density of c*relu(Y): for c>0 supported on x>0, for c<0 on x<0
        valid = (pos > 0)
        g[valid] = phi((pos[valid] - m) / s) / (s * abs(c))
        atom = float(Phi(-m / s))
        G = np.fft.rfft(np.fft.ifftshift(g)) * dx + atom
        F = F * G
    f = np.fft.fftshift(np.fft.irfft(F, GRID_N))
    f = np.maximum(f, 0.0)
    f /= f.sum() * dx

    exact = float(np.sum(np.maximum(grid, 0.0) * f) * dx)
    mom = [float(np.sum(grid ** p * f) * dx) for p in range(1, 7)]
    k = {}
    k[1] = mom[0]
    c2 = mom[1] - mom[0] ** 2
    k[2] = c2
    k[3] = mom[2] - 3 * mom[0] * mom[1] + 2 * mom[0] ** 3
    k[4] = (mom[3] - 4 * mom[0] * mom[2] - 3 * mom[1] ** 2
            + 12 * mom[0] ** 2 * mom[1] - 6 * mom[0] ** 4)
    k[5] = (mom[4] - 5 * mom[0] * mom[3] - 10 * mom[1] * mom[2]
            + 20 * mom[0] ** 2 * mom[2] + 30 * mom[0] * mom[1] ** 2
            - 60 * mom[0] ** 3 * mom[1] + 24 * mom[0] ** 5)
    k[6] = (mom[5] - 6 * mom[0] * mom[4] - 15 * mom[1] * mom[3]
            - 10 * mom[2] ** 2 + 30 * mom[0] ** 2 * mom[3]
            + 120 * mom[0] * mom[1] * mom[2] + 30 * mom[1] ** 3
            - 270 * mom[0] ** 2 * mom[1] ** 2 - 120 * mom[0] ** 3 * mom[2]
            + 360 * mom[0] ** 4 * mom[1] - 120 * mom[0] ** 6)
    return exact, k


# --------------------------------------------------------------------------
# Reconstructions
# --------------------------------------------------------------------------
def rec_gauss(k):
    s = np.sqrt(k[2])
    return relu_gauss(k[1], s)


def rec_edgeworth(k, order):
    s = np.sqrt(k[2])
    a = k[1] / s
    g1 = k[3] / s ** 3
    corr = -g1 * a / 6.0
    if order >= 4:
        g2 = k[4] / s ** 4
        corr += g2 * hermite_prob(2, a) / 24.0 + g1 * g1 * hermite_prob(4, a) / 72.0
    if order >= 6:
        g3, g4 = k[5] / s ** 5, k[6] / s ** 6
        corr += (-g3 * hermite_prob(3, a) / 120.0
                 + g4 * hermite_prob(4, a) / 720.0
                 + g1 * g2 * hermite_prob(5, a) / 144.0
                 - g1 ** 3 * hermite_prob(7, a) / 1296.0)
    return relu_gauss(k[1], s) + s * phi(a) * corr


def rec_saddlepoint(k, order=4, nq=96):
    """E[relu(X)] = 1/2 (m + E|X-m|) + ... via numeric CF inversion of a
    *tilted* truncated CGF is unstable; instead integrate the tail probability
    from a Lugannani-Rice saddlepoint on the truncated CGF."""
    m, c2, c3 = k[1], k[2], k[3]
    c4 = k[4] if order >= 4 else 0.0
    s = np.sqrt(c2)

    def K(t):
        return m * t + c2 * t ** 2 / 2 + c3 * t ** 3 / 6 + c4 * t ** 4 / 24

    def K1(t):
        return m + c2 * t + c3 * t ** 2 / 2 + c4 * t ** 3 / 6

    def K2(t):
        return c2 + c3 * t + c4 * t ** 2 / 2

    def tail(x):
        t = np.zeros_like(x)
        for _ in range(60):
            f = K1(t) - x
            d = K2(t)
            d = np.where(np.abs(d) < 1e-12, 1e-12, d)
            t = t - f / d
            t = np.clip(t, -12.0 / s, 12.0 / s)
        kpp = np.maximum(K2(t), 1e-30)
        w = np.sign(t) * np.sqrt(np.maximum(2.0 * (t * x - K(t)), 0.0))
        u = t * np.sqrt(kpp)
        out = np.where(
            np.abs(t) < 1e-7,
            0.5 - (k[3] / (6 * np.sqrt(2 * np.pi) * c2 ** 1.5)),
            1.0 - Phi(w) + phi(w) * (1.0 / np.where(np.abs(u) < 1e-12, 1e-12, u)
                                     - 1.0 / np.where(np.abs(w) < 1e-12, 1e-12, w)),
        )
        return np.clip(out, 0.0, 1.0)

    lo, hi = 0.0, max(float(m) + 12 * float(s), 12 * float(s))
    xs, ws = np.polynomial.legendre.leggauss(nq)
    xs = 0.5 * (hi - lo) * (xs + 1) + lo
    ws = ws * 0.5 * (hi - lo)
    return float(np.sum(ws * tail(xs)))


def rec_gamma_gauss(k, nq=400):
    """Shifted Gamma (x) Gaussian, matched to kappa_1..kappa_4 -- a genuine
    density with the right tail, unlike a truncated series."""
    c2, c3, c4 = k[2], k[3], k[4]
    sgn = 1.0 if c3 >= 0 else -1.0
    a3, a4 = abs(c3), c4
    if a3 < 1e-14 or a4 <= 0:
        return rec_edgeworth(k, 4)
    theta = a4 / (3.0 * a3)
    shape = a3 / (2.0 * theta ** 3)
    tau2 = c2 - shape * theta * theta
    if not (np.isfinite(theta) and np.isfinite(shape)) or shape <= 1e-6 or tau2 <= 1e-12:
        return rec_edgeworth(k, 4)
    tau = np.sqrt(tau2)
    u = (np.arange(nq) + 0.5) / nq
    zq = np.sqrt(2.0) * _erfinv(2.0 * u - 1.0)
    wh = shape * np.maximum(1.0 - 1.0 / (9 * shape) + zq / np.sqrt(9 * shape), 0.0) ** 3
    centre = k[1] - sgn * shape * theta
    mu = sgn * (centre * sgn + theta * wh) if sgn > 0 else -(theta * wh) + centre
    mu = centre + sgn * theta * wh
    val = float(np.mean(mu * Phi(mu / tau) + tau * phi(mu / tau)))
    return val


def _erfinv(y):
    w = -np.log(np.maximum(1.0 - y * y, 1e-300))
    out = np.empty_like(y)
    lo = w < 5.0
    ww = w[lo] - 2.5
    p = 2.81022636e-08
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
              -0.00125372503, -0.00417768164, 0.246640727, 1.50140941):
        p = p * ww + c
    out[lo] = p * y[lo]
    hi = ~lo
    if hi.any():
        ww = np.sqrt(w[hi]) - 3.0
        p = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
                  -0.0076224613, 0.00943887047, 1.00167406, 2.83297682):
            p = p * ww + c
        out[hi] = p * y[hi]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    methods = {
        "gauss": rec_gauss,
        "edge3": lambda k: rec_edgeworth(k, 3),
        "edge4": lambda k: rec_edgeworth(k, 4),
        "edge6": lambda k: rec_edgeworth(k, 6),
        "saddle4": lambda k: rec_saddlepoint(k, 4),
        "gamma_g": rec_gamma_gauss,
    }
    errs = {m: [] for m in methods}
    kept = 0
    tried = 0
    stats = []
    while kept < args.n_cases and tried < args.n_cases * 40:
        tried += 1
        K = int(rng.integers(3, 9))
        exact, k = build_exact(rng, K, span=80.0, hi=2.0)
        s = np.sqrt(max(k[2], 1e-30))
        g1 = k[3] / s ** 3
        g2 = k[4] / s ** 4
        alpha = k[1] / s
        if not (0.15 <= abs(g1) <= 0.9 and -0.2 <= g2 <= 1.2 and 0.2 <= abs(alpha) <= 7.0):
            continue
        kept += 1
        stats.append((abs(g1), g2, abs(alpha)))
        for name, fn in methods.items():
            errs[name].append(fn(k) - exact)

    st = np.array(stats)
    print(f"# exact reconstruction bake-off  cases={kept} (from {tried} draws)")
    print(f"# population: |gamma1| {st[:, 0].mean():.3f}+-{st[:, 0].std():.3f}, "
          f"gamma2 {st[:, 1].mean():.3f}+-{st[:, 1].std():.3f}, "
          f"|alpha| {st[:, 2].mean():.2f}+-{st[:, 2].std():.2f}")
    print(f"# per-layer bar = {PER_LAYER_BAR:.3e}   (no Monte Carlo anywhere)")
    print()
    print("  method     rms_error     max|error|    vs gauss   verdict")
    print("  " + "-" * 60)
    base = float(np.sqrt(np.mean(np.array(errs["gauss"]) ** 2)))
    for name in methods:
        e = np.array(errs[name])
        r = float(np.sqrt(np.mean(e ** 2)))
        v = "PASS" if r <= PER_LAYER_BAR else "fail"
        print(f"  {name:9s}  {r:.4e}    {np.abs(e).max():.4e}   "
              f"{base / max(r, 1e-30):8.1f}x   {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
