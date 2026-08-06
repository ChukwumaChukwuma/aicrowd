#!/usr/bin/env python
"""Cache per-layer ground-truth statistics once, so reconstruction methods can
be compared for free.

``scripts/07`` showed that Edgeworth-through-4th-order with *exact* cumulants
still leaves 1.4e-4 per layer against a 2.4e-6 budget, and that adding the
gamma_1^2 term buys nothing.  That points at the **reconstruction** — turning
cumulants into ``E[relu]`` — being the bottleneck, not the cumulants.  Edgeworth
is an asymptotic series whose effective expansion parameter here is
``gamma_1 = 0.44``, so it stalls.

Comparing alternative reconstructions (saddlepoint, Fourier inversion,
Cornish-Fisher, ...) needs the same measured inputs over and over.  Running
Monte Carlo each time is wasteful and, worse, changes the noise realisation
between methods so small differences become unreadable.  So this script runs
the sampling **once** and caches, per layer:

  * ``m``, ``S`` — the exact propagated mean and covariance of ``z``, formed
    from the *measured* previous-layer activation moments, so they carry no
    Gaussian assumption;
  * measured central moments of ``z`` up to order 8, from which cumulants up to
    order 6 are formed;
  * the measured truth ``E[relu(z)]``.

Two independent halves are cached so any reconstruction can be scored with the
same unbiased, noise-corrected estimator the main harness uses.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.mc import make_mlp  # noqa: E402

MAXP = 8


def stream(weights, n_samples, seed, chunk=1024):
    depth = len(weights)
    n = weights[0].shape[0]
    rng = np.random.default_rng(seed)
    xs = [np.zeros(n) for _ in range(depth)]
    xg = [np.zeros((n, n)) for _ in range(depth)]
    zp = [np.zeros((MAXP, n)) for _ in range(depth)]
    in_g = np.zeros((n, n))
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        in_g += x.T.astype(np.float64) @ x.astype(np.float64)
        for li, w in enumerate(weights):
            z = x @ w
            zf = z.astype(np.float64)
            acc = np.ones_like(zf)
            for p in range(MAXP):
                acc *= zf
                zp[li][p] += acc.sum(axis=0)
            x = np.maximum(z, np.float32(0.0))
            xf = x.astype(np.float64)
            xs[li] += xf.sum(axis=0)
            xg[li] += xf.T @ xf
        done += nb
    N = done
    return {
        "N": N,
        "in_cov": in_g / N,
        "x_mean": np.array([s / N for s in xs]),
        "x_cov": np.array([g / N - np.outer(s / N, s / N)
                           for g, s in zip(xg, xs)]),
        "z_raw": np.array([p / N for p in zp]),  # (depth, MAXP, n) raw moments
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=500_000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--seed-a", type=int, default=770_001)
    ap.add_argument("--seed-b", type=int, default=770_002)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    W = make_mlp(args.width, args.depth, args.mlp_seed)
    out = Path(args.out or (
        Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts"))
        / "layerstats" / f"mlp{args.mlp_seed}_N{args.n_samples}.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)

    halves = {}
    for tag, sd in (("a", args.seed_a), ("b", args.seed_b)):
        print(f"[cache] half {tag} ...", flush=True)
        halves[tag] = stream(W, args.n_samples, sd)

    # Exact propagated (m, S) per layer, built from measured previous-layer
    # activation moments -- no Gaussian assumption enters these.
    def propagate(st):
        ms, Ss = [], []
        prev_mean = np.zeros(args.width)
        prev_cov = st["in_cov"]
        for li in range(args.depth):
            w = W[li].astype(np.float64)
            ms.append(prev_mean @ w)
            Ss.append(np.diag(w.T @ prev_cov @ w).copy())
            prev_mean = st["x_mean"][li]
            prev_cov = st["x_cov"][li]
        return np.array(ms), np.array(Ss)

    payload = {
        "width": args.width, "depth": args.depth,
        "mlp_seed": args.mlp_seed, "n_per_half": args.n_samples,
    }
    for tag in ("a", "b"):
        m, Sd = propagate(halves[tag])
        payload[f"m_{tag}"] = m
        payload[f"vardiag_{tag}"] = Sd
        payload[f"truth_{tag}"] = halves[tag]["x_mean"]
        payload[f"zraw_{tag}"] = halves[tag]["z_raw"]
        payload[f"xvar_{tag}"] = np.array(
            [np.diag(c) for c in halves[tag]["x_cov"]])
    np.savez_compressed(out, **payload)
    print(f"[cache] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
