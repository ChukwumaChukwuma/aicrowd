#!/usr/bin/env python
"""Where does covariance propagation's error actually come from?

The Gaussian-propagation family makes exactly one approximation: it assumes the
pre-activation vector ``z^l`` is jointly Gaussian.  Everything else in it is
exact — the linear map ``m -> W^T m``, ``Sigma -> W^T Cov(x) W``, and the
rectified-Gaussian marginals.

So the interesting quantity is not the end-to-end error (which mixes a
per-layer modelling error with 32 layers of accumulation).  It is the
**one-step error**: feed the layer its *measured* exact mean and covariance,
apply the Gaussian rectifier formula once, and compare with the measured truth.

That isolates the modelling error per layer with zero accumulation:

    m_l  = mu_true^{l-1} W^l                       (exact, no assumption)
    S_l  = W^l' Cov_true(x^{l-1}) W^l              (exact, no assumption)
    mu_gauss[j] = m_j Phi(m_j/s_j) + s_j phi(m_j/s_j)     <- the only assumption
    one_step_error = mu_gauss - mu_true^l

Also reported, because they are what an Edgeworth correction would consume:
the measured skewness and excess kurtosis of each ``z^l_j``, and the measured
"order parameter" ratio ``mean(m^2)/mean(z^2)`` which controls how much of the
signal is input-independent by layer l.

Everything is measured on the *same* Monte-Carlo stream, so the comparison is
paired and the MC noise largely cancels.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import Phi, phi, relu_mean  # noqa: E402


def measure(weights, n_samples, seed, chunk=1024):
    """One streamed pass collecting, per layer: mean, second moment matrix,
    and the third/fourth central moments of the pre-activation z."""
    depth = len(weights)
    n = weights[0].shape[0]
    rng = np.random.default_rng(seed)

    x_sum = [np.zeros(n) for _ in range(depth)]
    x_gram = [np.zeros((n, n)) for _ in range(depth)]
    z_sum = [np.zeros(n) for _ in range(depth)]
    z_p2 = [np.zeros(n) for _ in range(depth)]
    z_p3 = [np.zeros(n) for _ in range(depth)]
    z_p4 = [np.zeros(n) for _ in range(depth)]
    in_sum = np.zeros(n)
    in_gram = np.zeros((n, n))

    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        in_sum += x.sum(axis=0, dtype=np.float64)
        in_gram += (x.T.astype(np.float64) @ x.astype(np.float64))
        for li, w in enumerate(weights):
            z = x @ w
            zf = z.astype(np.float64)
            z_sum[li] += zf.sum(axis=0)
            z2 = zf * zf
            z_p2[li] += z2.sum(axis=0)
            z_p3[li] += (z2 * zf).sum(axis=0)
            z_p4[li] += (z2 * z2).sum(axis=0)
            x = np.maximum(z, np.float32(0.0))
            xf = x.astype(np.float64)
            x_sum[li] += xf.sum(axis=0)
            x_gram[li] += xf.T @ xf
        done += nb

    N = done
    out = {
        "N": N,
        "in_mean": in_sum / N,
        "in_cov": in_gram / N - np.outer(in_sum / N, in_sum / N),
        "x_mean": [s / N for s in x_sum],
        "x_cov": [
            g / N - np.outer(s / N, s / N) for g, s in zip(x_gram, x_sum)
        ],
        "z_mean": [s / N for s in z_sum],
        "z_m2": [p / N for p in z_p2],
        "z_m3": [p / N for p in z_p3],
        "z_m4": [p / N for p in z_p4],
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=500_000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--mc-seed", type=int, default=987_654)
    args = ap.parse_args()

    W = make_mlp(args.width, args.depth, args.mlp_seed)
    st = measure(W, args.n_samples, args.mc_seed)
    N = st["N"]

    print(f"# layer anatomy  width={args.width} depth={args.depth} "
          f"mlp_seed={args.mlp_seed} N={N:,}")
    print(f"# MC noise on a per-neuron mean ~ sqrt(v/N); "
          f"one-step errors below must be read against that.")
    hdr = ("layer  onestep_rms  onestep_mean   |  skew_rms  exkurt_mean  "
           "|  r=m2/q   sigma_z   mu_rms   mcnoise")
    print(hdr)
    print("-" * len(hdr))

    prev_mean = st["in_mean"]
    prev_cov = st["in_cov"]
    for li in range(args.depth):
        w = W[li].astype(np.float64)
        m = prev_mean @ w
        S = w.T @ prev_cov @ w
        s = np.sqrt(np.maximum(np.diag(S), 1e-30))

        mu_gauss = relu_mean(m, s)
        mu_true = st["x_mean"][li]
        err = mu_gauss - mu_true

        zm = st["z_mean"][li]
        c2 = st["z_m2"][li] - zm * zm
        c3 = st["z_m3"][li] - 3 * zm * st["z_m2"][li] + 2 * zm**3
        c4 = (st["z_m4"][li] - 4 * zm * st["z_m3"][li]
              + 6 * zm * zm * st["z_m2"][li] - 3 * zm**4) - 3 * c2 * c2
        skew = c3 / c2**1.5
        exkurt = c4 / (c2 * c2)

        q = float(np.mean(st["z_m2"][li]))
        r = float(np.mean(zm * zm)) / q
        v_final = float(np.mean(st["x_cov"][li].diagonal()))
        mcnoise = np.sqrt(v_final / N)

        print(f"{li + 1:5d}  {np.sqrt(np.mean(err**2)):11.3e}  "
              f"{np.mean(err):+.3e}  |  {np.sqrt(np.mean(skew**2)):8.4f}  "
              f"{np.mean(exkurt):+11.4f}  |  {r:6.4f}  {np.sqrt(q):7.4f}  "
              f"{np.sqrt(np.mean(mu_true**2)):7.4f}  {mcnoise:.2e}")

        prev_mean = mu_true
        prev_cov = st["x_cov"][li]

    print()
    print("Reading guide: onestep_rms is the per-neuron RMS error introduced by "
          "the Gaussian assumption in ONE layer, given exact inputs.\n"
          "The target end-to-end RMS error is 1.34e-5.  If onestep_rms is far "
          "above that at every layer, closed-form Gaussian propagation cannot "
          "reach the floor no matter how the chain is arranged, and the "
          "non-Gaussian correction is mandatory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
