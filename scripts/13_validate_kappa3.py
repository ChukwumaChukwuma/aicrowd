#!/usr/bin/env python
"""Validate the star-diagram kappa_3 against brute-force Monte Carlo.

Bar fixed before the run: the star terms must capture at least 80% of the true
kappa_3 (measured as 1 - ||k3_star - k3_true|| / ||k3_true||) at the
competition's correlation scale.  Below that, the omitted triangle diagrams
dominate and the O(n^3) shortcut is not usable.

The test uses a *real* layer: an MLP is propagated with measured moments so
that ``R`` has the correlation structure the network actually produces, rather
than a synthetic one that would flatter the expansion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.cumulants import kappa3_brute, kappa3_star  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import relu_cov_mehler, relu_mean  # noqa: E402


def layer_state(W, upto, n_samples, seed, chunk=1024):
    """Measured (m, cov) of z at layer `upto` (0-based), by Monte Carlo."""
    n = W[0].shape[0]
    rng = np.random.default_rng(seed)
    s1 = np.zeros(n)
    g = np.zeros((n, n))
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        for li, w in enumerate(W):
            z = x @ w
            if li == upto:
                zf = z.astype(np.float64)
                s1 += zf.sum(axis=0)
                g += zf.T @ zf
                break
            x = np.maximum(z, np.float32(0.0))
        done += nb
    m = s1 / done
    return m, g / done - np.outer(m, m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--layers", type=str, default="1,3,5,7")
    ap.add_argument("--n-state", type=int, default=200_000)
    ap.add_argument("--n-brute", type=int, default=4_000_000)
    ap.add_argument("--umax", type=str, default="1,2,4,6")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    W = make_mlp(args.width, args.depth, args.seed)
    umaxes = [int(x) for x in args.umax.split(",")]
    print(f"# kappa_3 star-diagram validation  width={args.width} "
          f"depth={args.depth} n_brute={args.n_brute:,}")
    print("# bar: star terms capture >= 80% of true kappa_3")
    print()
    print("layer  |rho|_off  ||k3_true||   " +
          "  ".join(f"umax={u:<2d}" for u in umaxes))
    print("-" * (34 + 10 * len(umaxes)))

    ok_all = True
    for li in [int(x) for x in args.layers.split(",")]:
        m, cov = layer_state(W, li, args.n_state, 4242 + li)
        s = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        R = cov / np.outer(s, s)
        np.fill_diagonal(R, 1.0)
        Wn = W[li + 1].astype(np.float64) if li + 1 < args.depth else W[0].astype(np.float64)

        k3t = kappa3_brute(Wn, m, cov, args.n_brute, 909 + li)
        nrm = np.linalg.norm(k3t)
        off = np.abs(R - np.eye(len(R))).sum() / (len(R) * (len(R) - 1))

        caps = []
        for u in umaxes:
            k3s = kappa3_star(Wn, m, s, R, umax=u)
            cap = 1.0 - np.linalg.norm(k3s - k3t) / nrm
            caps.append(cap)
        best = max(caps)
        ok_all &= best >= 0.80
        print(f"{li + 1:5d}  {off:9.4f}  {nrm:11.4e}   " +
              "  ".join(f"{c:7.3f}" for c in caps))

    print()
    print("VERDICT:", "PASS" if ok_all else "FAIL",
          "- star diagrams capture >=80% of kappa_3" if ok_all else
          "- omitted triangle diagrams are not negligible; the O(n^3) "
          "shortcut does not reproduce kappa_3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
