#!/usr/bin/env python
"""Does an Edgeworth correction built from the TRUE cumulants close the gap?

``scripts/04`` measured the one-step error of the Gaussian assumption at
~1.3e-3 RMS per layer.  ``scripts/06`` refuted the hypothesis that this error
is carried by the few dominant covariance directions (only 2.9x from
conditioning on the top 3).  So the correction has to come from the shape of
each pre-activation's own distribution.

The Edgeworth series for the rectified mean.  With ``alpha = m/sigma``,
``gamma1 = kappa_3/sigma^3``, ``gamma2 = kappa_4/sigma^4``:

    E[relu(z)] = m Phi(a) + sigma phi(a)
                 + sigma phi(a) [ -gamma1 a/6
                                  + gamma2 (a^2-1)/24
                                  + gamma1^2 He_4(a)/72 + ... ]

*Derivation of the general term.*  A density perturbation ``c·He_n(t)φ(t)``
contributes ``sigma·c·∫_a^∞ (t-a) He_n(t) φ(t) dt`` with ``a = -alpha``, and

    ∫_a^∞ (t-a) He_n(t) φ(t) dt = He_{n-2}(a) φ(a)

(using ``t He_n = He_{n+1} + n He_{n-1}``, ``∫_a^∞ He_n φ = He_{n-1}(a)φ(a)``,
and the recurrence ``He_n = a He_{n-1} - (n-1) He_{n-2}``).  That identity is
what makes every Edgeworth order a one-line closed form here.

This script feeds the correction the *measured* cumulants — the best case any
analytic scheme could achieve at that truncation order — and reports how much
of the one-step error survives.  If the measured cumulants cannot close the
gap, no method for *computing* those cumulants can either, and the Edgeworth
route is dead regardless of how cleverly the cumulants are obtained.

A synthetic Gaussian control runs the identical pipeline on data that is
Gaussian by construction, so the reported floor is the measurement's own noise
rather than a modelling error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import Phi, hermite_prob, phi, relu_mean  # noqa: E402


def edgeworth_relu_mean(m, sigma, k3=None, k4=None, use_k3sq=False):
    """Rectified mean with optional Edgeworth corrections."""
    a = m / sigma
    out = m * Phi(a) + sigma * phi(a)
    if k3 is None and k4 is None:
        return out
    g1 = (k3 / sigma**3) if k3 is not None else 0.0
    g2 = (k4 / sigma**4) if k4 is not None else 0.0
    corr = -g1 * a / 6.0
    if k4 is not None:
        corr = corr + g2 * hermite_prob(2, a) / 24.0
    if use_k3sq:
        corr = corr + (g1 * g1) * hermite_prob(4, a) / 72.0
    return out + sigma * phi(a) * corr


def stream(weights, n_samples, seed, chunk=1024):
    depth = len(weights)
    n = weights[0].shape[0]
    rng = np.random.default_rng(seed)
    xs = [np.zeros(n) for _ in range(depth)]
    xg = [np.zeros((n, n)) for _ in range(depth)]
    zs = [np.zeros(n) for _ in range(depth)]
    zp = [[np.zeros(n) for _ in range(3)] for _ in range(depth)]  # m2,m3,m4
    in_g = np.zeros((n, n))
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        in_g += x.T.astype(np.float64) @ x.astype(np.float64)
        for li, w in enumerate(weights):
            z = x @ w
            zf = z.astype(np.float64)
            zs[li] += zf.sum(axis=0)
            z2 = zf * zf
            zp[li][0] += z2.sum(axis=0)
            zp[li][1] += (z2 * zf).sum(axis=0)
            zp[li][2] += (z2 * z2).sum(axis=0)
            x = np.maximum(z, np.float32(0.0))
            xf = x.astype(np.float64)
            xs[li] += xf.sum(axis=0)
            xg[li] += xf.T @ xf
        done += nb
    N = done
    return {
        "N": N,
        "in_cov": in_g / N,
        "x_mean": [s / N for s in xs],
        "x_cov": [g / N - np.outer(s / N, s / N) for g, s in zip(xg, xs)],
        "z_mean": [s / N for s in zs],
        "z_m": [[p / N for p in row] for row in zp],
    }


def cumulants(zm, m2, m3, m4):
    c2 = m2 - zm * zm
    c3 = m3 - 3 * zm * m2 + 2 * zm**3
    c4 = (m4 - 4 * zm * m3 + 6 * zm * zm * m2 - 3 * zm**4) - 3 * c2 * c2
    return c2, c3, c4


def gaussian_control(m, S, n_samples, seed):
    """Same pipeline on data that IS Gaussian: reports the measurement floor."""
    n = m.shape[0]
    ev, U = np.linalg.eigh(S)
    L = U * np.sqrt(np.maximum(ev, 0.0))
    rng = np.random.default_rng(seed)
    tot = np.zeros(n)
    s1 = np.zeros(n)
    g1 = np.zeros((n, n))
    done = 0
    chunk = 4096
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        g = (rng.standard_normal((nb, n)) @ L.T + m).astype(np.float32)
        r = np.maximum(g, np.float32(0.0)).astype(np.float64)
        tot += r.sum(axis=0)
        gf = g.astype(np.float64)
        s1 += gf.sum(axis=0)
        g1 += gf.T @ gf
        done += nb
    truth = tot / done
    mm = s1 / done
    SS = g1 / done - np.outer(mm, mm)
    ss = np.sqrt(np.maximum(np.diag(SS), 1e-30))
    return float(np.sqrt(np.mean((relu_mean(mm, ss) - truth) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=400_000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--mc-seed", type=int, default=246_813)
    ap.add_argument("--control-layers", type=str, default="32")
    args = ap.parse_args()

    W = make_mlp(args.width, args.depth, args.mlp_seed)
    st = stream(W, args.n_samples, args.mc_seed)
    N = st["N"]
    ctrl = {int(x) for x in args.control_layers.split(",") if x}

    print(f"# edgeworth test  width={args.width} depth={args.depth} "
          f"mlp_seed={args.mlp_seed} N={N:,}")
    hdr = ("layer   gauss      +k3        +k3+k4     +k3+k4+k3^2   "
           "best/gauss   noise_ctrl")
    print(hdr)
    print("-" * len(hdr))

    prev_mean = np.zeros(args.width)
    prev_cov = st["in_cov"]
    for li in range(args.depth):
        w = W[li].astype(np.float64)
        m = prev_mean @ w
        S = w.T @ prev_cov @ w
        s = np.sqrt(np.maximum(np.diag(S), 1e-30))
        truth = st["x_mean"][li]

        c2, c3, c4 = cumulants(st["z_mean"][li], *st["z_m"][li])
        # use the propagated sigma (exact given exact inputs), measured k3/k4
        e_g = relu_mean(m, s) - truth
        e_3 = edgeworth_relu_mean(m, s, c3) - truth
        e_34 = edgeworth_relu_mean(m, s, c3, c4) - truth
        e_345 = edgeworth_relu_mean(m, s, c3, c4, use_k3sq=True) - truth

        def r(e):
            return float(np.sqrt(np.mean(e**2)))

        best = min(r(e_3), r(e_34), r(e_345))
        cs = ""
        if (li + 1) in ctrl:
            cs = f"{gaussian_control(m, S, min(N, 300_000), 4242):.3e}"
        print(f"{li + 1:5d}  {r(e_g):.3e}  {r(e_3):.3e}  {r(e_34):.3e}  "
              f"{r(e_345):.3e}     {r(e_g) / max(best, 1e-30):6.2f}x   {cs}")

        prev_mean = truth
        prev_cov = st["x_cov"][li]

    print()
    print("noise_ctrl is the same statistic computed on genuinely Gaussian "
          "data at the same N: any 'error' at or below it is measurement "
          "noise, not modelling error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
