#!/usr/bin/env python
"""Is the pre-activation *conditionally* Gaussian along its dominant directions?

``scripts/04`` established that treating ``z^l`` as jointly Gaussian costs
~1.3e-3 RMS per layer, 100x more than the whole error budget.  ``scripts/05``
established that the fluctuation of ``z^l`` is dominated by very few
directions (participation ratio 2.2 at layer 32, top eigenvalue 67% of the
trace) but has a heavy tail, so simply *truncating* to a low rank fails.

This script tests the hypothesis that makes both facts useful at once:

    the non-Gaussianity of z^l is carried almost entirely by its few
    dominant directions, so that CONDITIONAL on the top-k coordinates the
    remainder is very close to Gaussian.

If true, then representing ``z^l`` as a *mixture* of Gaussians indexed by the
top-k coordinates — rather than as one Gaussian — removes most of the error,
and such a mixture is cheap to propagate: the components share one covariance
and differ only in their means.

The test.  Take MC samples of ``z^l``.  Let ``T`` be the coordinates along the
top-k eigenvectors of ``Cov(z^l)``.  Bin the samples by ``T``.  Predict

    E[relu(z_j)] ~= sum_cells P(cell) * g( E[z_j | cell], sd(z_j | cell) )

with ``g`` the exact rectified-Gaussian mean — i.e. assume Gaussianity only
*within* a cell.  Compare against the measured truth, and against the k = 0
case, which is exactly the unconditional Gaussian assumption that costs 1.3e-3.

Bin-count convergence is reported so that binning error is not mistaken for
signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import relu_mean  # noqa: E402


def collect_z(weights, layer, n_samples, seed, chunk=2048):
    """Return the (n_samples, width) matrix of pre-activations at `layer`."""
    n = weights[0].shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, n), dtype=np.float32)
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        for li, w in enumerate(weights):
            z = x @ w
            if li == layer:
                out[done:done + nb] = z
                break
            x = np.maximum(z, np.float32(0.0))
        done += nb
    return out


def conditional_prediction(Z, T, nbins):
    """Predict E[relu(z_j)] assuming Gaussianity only within cells of T.

    T is (N, k).  Cells are the product of per-axis equal-probability bins.
    Returns the (width,) prediction.
    """
    N, n = Z.shape
    k = T.shape[1]
    # equal-count bins per axis via quantiles -> cell index
    idx = np.zeros(N, dtype=np.int64)
    for a in range(k):
        q = np.quantile(T[:, a], np.linspace(0, 1, nbins + 1)[1:-1])
        b = np.searchsorted(q, T[:, a])
        idx = idx * nbins + b
    ncell = nbins ** k

    counts = np.bincount(idx, minlength=ncell).astype(np.float64)
    pred = np.zeros(n, dtype=np.float64)
    order = np.argsort(idx, kind="stable")
    Zs = Z[order]
    starts = np.concatenate(([0], np.cumsum(counts).astype(np.int64)))
    for c in range(ncell):
        lo, hi = starts[c], starts[c + 1]
        if hi - lo < 8:
            if hi > lo:  # too few points to estimate a variance: use them raw
                pred += np.maximum(Zs[lo:hi], 0.0).sum(axis=0, dtype=np.float64)
            continue
        blk = Zs[lo:hi].astype(np.float64)
        m = blk.mean(axis=0)
        v = blk.var(axis=0)
        s = np.sqrt(np.maximum(v, 1e-30))
        pred += (hi - lo) * relu_mean(m, s)
    return pred / N


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=400_000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--layer", type=int, default=32, help="1-based")
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--mc-seed", type=int, default=31_337)
    args = ap.parse_args()

    W = make_mlp(args.width, args.depth, args.mlp_seed)
    Z = collect_z(W, args.layer - 1, args.n_samples, args.mc_seed)
    N = Z.shape[0]

    truth = np.maximum(Z, 0.0).mean(axis=0, dtype=np.float64)
    Zf = Z.astype(np.float64)
    m = Zf.mean(axis=0)
    Zc = Zf - m
    Sig = (Zc.T @ Zc) / N
    s = np.sqrt(np.maximum(np.diag(Sig), 1e-30))

    v = float(np.mean(np.maximum(Z, 0.0).var(axis=0)))
    mcnoise = np.sqrt(v / N)

    base = relu_mean(m, s)
    print(f"# conditional gaussianity  layer={args.layer} N={N:,} "
          f"mlp_seed={args.mlp_seed}")
    print(f"# per-neuron MC noise on the truth = {mcnoise:.3e}  "
          f"(measured final-layer variance v = {v:.4f})")
    print(f"# target end-to-end RMS error = 1.34e-5")
    print()
    print("  k  nbins   cells    rms_error    mean_error    vs k=0")
    print("  " + "-" * 56)
    e0 = float(np.sqrt(np.mean((base - truth) ** 2)))
    print(f"  0      -       1    {e0:.4e}   {np.mean(base - truth):+.3e}"
          f"      1.0x")

    evals, evecs = np.linalg.eigh(Sig)
    order = np.argsort(evals)[::-1]
    U = evecs[:, order]
    T_all = Zc @ U

    for k, nbins_list in ((1, (16, 64, 256)), (2, (8, 16, 32)), (3, (6, 10))):
        for nbins in nbins_list:
            if nbins ** k > N / 40:
                continue
            pred = conditional_prediction(Z, T_all[:, :k], nbins)
            e = float(np.sqrt(np.mean((pred - truth) ** 2)))
            print(f"  {k}  {nbins:5d}  {nbins ** k:6d}    {e:.4e}   "
                  f"{np.mean(pred - truth):+.3e}    {e0 / e:6.1f}x")

    print()
    print("If rms_error falls by orders of magnitude with k and keeps falling "
          "as nbins grows (rather than flattening at the binning error), then "
          "the non-Gaussianity is carried by the dominant directions and a "
          "shared-covariance Gaussian mixture indexed by those directions is "
          "the right state to propagate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
