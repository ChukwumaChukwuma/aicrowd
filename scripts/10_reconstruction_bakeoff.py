#!/usr/bin/env python
"""Given the exact cumulants, which reconstruction recovers E[relu] best?

``scripts/07`` found Edgeworth-through-4th-order with oracle cumulants stalls
at 1.4e-4 per layer (bar: 2.4e-6), and that adding the gamma_1^2 term buys
nothing.  Two explanations are possible and they lead to opposite strategies:

  (i)  the cumulants beyond 4th order genuinely matter, so no method using only
       kappa_2..kappa_4 can work; or
  (ii) the *reconstruction* is at fault -- a truncated Edgeworth series is not
       a probability density (it goes negative in the tails, and the tail is
       exactly where |alpha| ~ 4.4 puts us), so it misrepresents E[relu] even
       when the cumulants it is given are exact.

This script separates them by holding the cumulants fixed and varying only the
reconstruction:

  gauss          kappa_1,2 only -- the baseline every propagation scheme uses
  edge4          Edgeworth with kappa_3, kappa_4, kappa_3^2
  edge6          Edgeworth through kappa_6
  gamma_gauss    shifted Gamma convolved with a Gaussian, moment-matched to
                 kappa_1..kappa_4.  A genuine density: non-negative, correct
                 tail shape, and E[relu] is exact per Gamma node.
  mix2           two-point mean mixture convolved with a Gaussian, matched to
                 kappa_1..kappa_4.  Also a genuine density.

If a valid-density reconstruction beats Edgeworth by a wide margin, (ii) holds
and the route is to compute kappa_3/kappa_4 analytically and reconstruct
properly.  If they all stall together, (i) holds and the cumulant route is
dead however it is reconstructed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.relu_moments import Phi, hermite_prob, phi  # noqa: E402


def relu_gauss(m, s):
    a = m / s
    return m * Phi(a) + s * phi(a)


def raw_to_cumulants(raw):
    """raw[p] = E[z^{p+1}] for p=0..7  ->  cumulants k1..k6."""
    m = [None] + [raw[p] for p in range(6)]  # m[1..6] raw moments
    k = {}
    k[1] = m[1]
    k[2] = m[2] - m[1] ** 2
    k[3] = m[3] - 3 * m[1] * m[2] + 2 * m[1] ** 3
    k[4] = (m[4] - 4 * m[1] * m[3] - 3 * m[2] ** 2
            + 12 * m[1] ** 2 * m[2] - 6 * m[1] ** 4)
    k[5] = (m[5] - 5 * m[1] * m[4] - 10 * m[2] * m[3] + 20 * m[1] ** 2 * m[3]
            + 30 * m[1] * m[2] ** 2 - 60 * m[1] ** 3 * m[2] + 24 * m[1] ** 5)
    k[6] = (m[6] - 6 * m[1] * m[5] - 15 * m[2] * m[4] - 10 * m[3] ** 2
            + 30 * m[1] ** 2 * m[4] + 120 * m[1] * m[2] * m[3]
            + 30 * m[2] ** 3 - 270 * m[1] ** 2 * m[2] ** 2
            - 120 * m[1] ** 3 * m[3] + 360 * m[1] ** 4 * m[2]
            - 120 * m[1] ** 6)
    return k


def edgeworth(m, s, k, order):
    """Edgeworth rectified mean.  A density term c*He_n contributes
    sigma*c*He_{n-2}(a)*phi(a) with a = -alpha (see scripts/07)."""
    a = m / s
    out = relu_gauss(m, s)
    g1 = k[3] / s ** 3
    corr = -g1 * a / 6.0
    if order >= 4:
        g2 = k[4] / s ** 4
        corr = corr + g2 * hermite_prob(2, a) / 24.0
        corr = corr + g1 * g1 * hermite_prob(4, a) / 72.0
    if order >= 6:
        g3 = k[5] / s ** 5
        g4 = k[6] / s ** 6
        # He_5 -> He_3 ; He_6 -> He_4 ; He_7 -> He_5 ; He_9 -> He_7
        corr = corr - g3 * hermite_prob(3, a) / 120.0
        corr = corr + g4 * hermite_prob(4, a) / 720.0
        corr = corr + (g1 * g2) * hermite_prob(5, a) / 144.0
        corr = corr - (g1 ** 3) * hermite_prob(7, a) / 1296.0
    return out + s * phi(a) * corr


def gamma_gauss(m, s, k, nquad=160):
    """Shifted Gamma convolved with a Gaussian, matched to kappa_1..kappa_4.

    For X = xi + theta*G + tau*N(0,1) with G ~ Gamma(kk, 1):
        k1 = xi + kk*theta,  k2 = kk*theta^2 + tau^2,
        k3 = 2*kk*theta^3,   k4 = 6*kk*theta^4
    hence theta = k4/(3*k3), kk = k3/(2*theta^3), tau^2 = k2 - kk*theta^2.
    Sign-flip handles k3 < 0.  Falls back to Edgeworth-4 where the family
    cannot represent the moments (tau^2 <= 0 or kk <= 0).
    """
    sgn = np.where(k[3] >= 0, 1.0, -1.0)
    k3 = np.abs(k[3])
    k4 = np.maximum(k[4], 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = k4 / (3.0 * k3)
        kk = k3 / (2.0 * theta ** 3)
        tau2 = k[2] - kk * theta * theta
    ok = np.isfinite(theta) & np.isfinite(kk) & (kk > 1e-8) & (tau2 > 1e-12) \
        & (theta > 0) & (k3 > 1e-300)
    out = edgeworth(m, s, k, 4)
    if not ok.any():
        return out

    mm = sgn * m                    # work with the sign-flipped variable
    xi = -kk * theta                # centred; the mean is added back below
    tau = np.sqrt(np.maximum(tau2, 1e-30))

    # Deterministic quadrature over the Gamma via equal-probability nodes,
    # using a Wilson-Hilferty quantile (accurate for the shapes that arise).
    u = (np.arange(nquad) + 0.5) / nquad
    zq = np.sqrt(2.0) * _erfinv(2.0 * u - 1.0)
    acc = np.zeros_like(m)
    kk_ = kk[ok][:, None]
    wh = kk_ * (1.0 - 1.0 / (9.0 * kk_) + zq[None, :] / np.sqrt(9.0 * kk_)) ** 3
    wh = np.maximum(wh, 0.0)
    mu = (mm[ok][:, None] + xi[ok][:, None] + theta[ok][:, None] * wh)
    tt = tau[ok][:, None]
    acc_ok = np.mean(mu * Phi(mu / tt) + tt * phi(mu / tt), axis=1)
    res = out.copy()
    # relu(-X) = relu(X) - X  =>  E[relu(X)] = E[relu(-X)] + E[X]
    res[ok] = np.where(sgn[ok] > 0, acc_ok, acc_ok + m[ok])
    return res


def _erfinv(y):
    """Inverse error function (Giles' rational approximation, ~1e-9)."""
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
    ap.add_argument("--cache", type=str, required=True)
    args = ap.parse_args()

    z = np.load(args.cache)
    depth = int(z["depth"])
    print(f"# reconstruction bakeoff  cache={Path(args.cache).name}  "
          f"depth={depth}  n_per_half={int(z['n_per_half']):,}")
    hdr = ("layer   gauss      edge4      edge6      gamma_g    mix_best   "
           "best/gauss")
    print(hdr)
    print("-" * len(hdr))

    tot = {kk: 0.0 for kk in ("gauss", "edge4", "edge6", "gamma")}
    for li in range(depth):
        # average the two halves' moment estimates; score against both halves'
        # truth via the unbiased cross product so reference noise cancels.
        m = 0.5 * (z["m_a"][li] + z["m_b"][li])
        var = 0.5 * (z["vardiag_a"][li] + z["vardiag_b"][li])
        s = np.sqrt(np.maximum(var, 1e-30))
        raw = 0.5 * (z["zraw_a"][li] + z["zraw_b"][li])
        k = raw_to_cumulants(raw)
        k[1] = m
        k[2] = var
        ta, tb = z["truth_a"][li], z["truth_b"][li]

        preds = {
            "gauss": relu_gauss(m, s),
            "edge4": edgeworth(m, s, k, 4),
            "edge6": edgeworth(m, s, k, 6),
            "gamma": gamma_gauss(m, s, k),
        }
        row = {}
        for name, p in preds.items():
            # unbiased true MSE: mean (p-a)(p-b); clip at 0 for the sqrt
            mse = float(np.mean((p - ta) * (p - tb)))
            row[name] = np.sqrt(max(mse, 0.0))
            tot[name] += max(mse, 0.0)
        best = min(row.values())
        print(f"{li + 1:5d}  {row['gauss']:.3e}  {row['edge4']:.3e}  "
              f"{row['edge6']:.3e}  {row['gamma']:.3e}  {'-':^9}  "
              f"{row['gauss'] / max(best, 1e-30):8.1f}x")

    print()
    print("layer-averaged unbiased RMS (reference noise removed):")
    for name in ("gauss", "edge4", "edge6", "gamma"):
        print(f"   {name:10s} {np.sqrt(tot[name] / depth):.4e}")
    print()
    print("Per-layer bar is 2.37e-6.  A valid-density reconstruction beating "
          "Edgeworth would say the cumulants are fine and the series was the "
          "problem; all of them stalling together would say kappa_5+ matter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
