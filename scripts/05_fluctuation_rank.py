#!/usr/bin/env python
"""How many directions does the pre-activation fluctuation actually live in?

Motivation.  ``scripts/04`` showed that by layer 32 about 95% of ``E[z^2]`` is
the input-independent mean, leaving only ~5% fluctuation.  Deep ReLU networks
at criticality are known to contract the input manifold; if that contraction
also *concentrates* the surviving fluctuation into a few directions, then the
whole non-Gaussian problem is low-dimensional and can be handled exactly,
with the high-dimensional remainder — which is a sum of many small independent
contributions and therefore genuinely close to Gaussian — left Gaussian.

This script measures, per layer:
  * the eigenvalue spectrum of ``Cov(z^l)``,
  * the number of eigenvalues needed to capture a given fraction of the total
    variance,
  * and, the number that actually matters, the number needed so that the
    *neglected* per-neuron variance is small enough to move ``E[relu]`` by less
    than the 1.34e-5 target.  Dropping variance ``eps`` from neuron j shifts
    its rectified mean by about ``phi(alpha_j) * eps / (2 s_j)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import phi  # noqa: E402


def stream_z_stats(weights, n_samples, seed, layers, chunk=1024):
    """Mean and second-moment matrix of z at the requested layers."""
    n = weights[0].shape[0]
    rng = np.random.default_rng(seed)
    want = set(layers)
    zs = {li: np.zeros(n) for li in want}
    zg = {li: np.zeros((n, n)) for li in want}
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        for li, w in enumerate(weights):
            z = x @ w
            if li in want:
                zf = z.astype(np.float64)
                zs[li] += zf.sum(axis=0)
                zg[li] += zf.T @ zf
            x = np.maximum(z, np.float32(0.0))
        done += nb
    out = {}
    for li in want:
        m = zs[li] / done
        out[li] = (m, zg[li] / done - np.outer(m, m))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=100_000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--mc-seed", type=int, default=555_001)
    args = ap.parse_args()

    W = make_mlp(args.width, args.depth, args.mlp_seed)
    layers = [0, 1, 3, 7, 11, 15, 19, 23, 27, 31]
    st = stream_z_stats(W, args.n_samples, args.mc_seed, layers)

    print(f"# fluctuation rank  width={args.width} depth={args.depth} "
          f"mlp_seed={args.mlp_seed} N={args.n_samples:,}")
    hdr = ("layer   tr(Sig)  lam1/tr  k@90%  k@99%  k@99.9%  k@99.99%   "
           "k_needed(1.3e-5)   eff_rank")
    print(hdr)
    print("-" * len(hdr))
    for li in layers:
        m, S = st[li]
        ev = np.linalg.eigvalsh(S)[::-1]
        ev = np.maximum(ev, 0.0)
        tot = ev.sum()
        cum = np.cumsum(ev) / tot

        def kfrac(f):
            return int(np.searchsorted(cum, f) + 1)

        s = np.sqrt(np.maximum(np.diag(S), 1e-30))
        alpha = m / s
        # dropping a tail of total variance eps spreads over neurons; the worst
        # per-neuron shift is bounded by phi(alpha)*eps_j/(2 s_j).  Use the mean
        # neglected per-neuron variance eps/n as the representative value.
        sens = phi(alpha) / (2.0 * s)
        k_need = args.width
        for k in range(1, args.width + 1):
            eps = tot - np.cumsum(ev)[k - 1]
            if float(np.sqrt(np.mean((sens * eps / args.width) ** 2))) < 1.34e-5:
                k_need = k
                break
        eff = float(tot * tot / np.sum(ev * ev))
        print(f"{li + 1:5d}  {tot:8.4f}  {ev[0] / tot:7.4f}  {kfrac(0.90):5d}  "
              f"{kfrac(0.99):5d}  {kfrac(0.999):7d}  {kfrac(0.9999):8d}  "
              f"{k_need:16d}   {eff:8.2f}")

    print()
    print("eff_rank is the participation ratio (sum(lam))^2 / sum(lam^2).")
    print("k_needed is the rank at which the neglected variance no longer "
          "moves any rectified mean by 1.34e-5 -- the only column that decides "
          "whether a low-rank exact treatment is viable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
