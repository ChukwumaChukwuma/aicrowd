#!/usr/bin/env python
"""ADVERSARIAL: is the "star diagrams capture 84-88% of kappa_3" claim
measuring the right kappa_3, and is the gap Monte-Carlo noise?

``scripts/13`` samples its brute-force reference from a **Gaussian** with the
layer's measured ``(m, cov)``.  The star formula assumes the same Gaussian, so
the comparison isolates the diagram truncation and nothing else.  The Edgeworth
term, however, needs the third cumulant of the *actual* pre-activation, which
is not Gaussian -- that non-Gaussianity is the entire reason the correction
exists.

This script measures both references on the same draws, in two independent
halves, so the sampling error of each is visible next to the gap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.cumulants import kappa3_brute, kappa3_star  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402


def true_state_and_k3(W, li, Wn, n, seed, chunk=4096):
    """Two independent halves of (m, cov) and of the TRUE kappa_3 of S=relu(z)@Wn."""
    width = W[0].shape[0]
    out = []
    for h in range(2):
        rng = np.random.default_rng(seed + 1000 * h)
        s1 = np.zeros(width)
        g = np.zeros((width, width))
        S1 = np.zeros(Wn.shape[1])
        S2 = np.zeros(Wn.shape[1])
        S3 = np.zeros(Wn.shape[1])
        done = 0
        while done < n:
            nb = min(chunk, n - done)
            x = rng.standard_normal((nb, width), dtype=np.float32)
            for lj, w in enumerate(W):
                z = x @ w
                if lj == li:
                    break
                x = np.maximum(z, np.float32(0.0))
            zf = z.astype(np.float64)
            s1 += zf.sum(axis=0)
            g += zf.T @ zf
            Sv = np.maximum(zf, 0.0) @ Wn
            S1 += Sv.sum(axis=0)
            S2 += (Sv * Sv).sum(axis=0)
            S3 += (Sv ** 3).sum(axis=0)
            done += nb
        m = s1 / done
        cov = g / done - np.outer(m, m)
        m1, m2, m3 = S1 / done, S2 / done, S3 / done
        out.append((m, cov, m3 - 3 * m1 * m2 + 2 * m1 ** 3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--layers", type=str, default="1,3,5")
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    W = make_mlp(args.width, args.depth, args.seed)
    print(f"# width={args.width} depth={args.depth} "
          f"n per half={args.n:,} (two independent halves)")
    print()
    print("layer   ||k3_true||   ||k3_gauss||   ratio   MC noise on ratio   "
          "cap_vs_gauss  cap_vs_true")
    print("-" * 94)
    for li in [int(x) for x in args.layers.split(",")]:
        Wn = (W[li + 1] if li + 1 < args.depth else W[0]).astype(np.float64)
        halves = true_state_and_k3(W, li, Wn, args.n, 4242 + li)
        ratios, capg, capt, ngs, nts = [], [], [], [], []
        for (m, cov, k3t) in halves:
            sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
            R = cov / np.outer(sd, sd)
            np.fill_diagonal(R, 1.0)
            k3g = kappa3_brute(Wn, m, cov, args.n, 5150 + li)
            k3s = kappa3_star(Wn, m, sd, R, umax=1)
            ngs.append(np.linalg.norm(k3g))
            nts.append(np.linalg.norm(k3t))
            ratios.append(np.linalg.norm(k3t) / np.linalg.norm(k3g))
            capg.append(1 - np.linalg.norm(k3s - k3g) / np.linalg.norm(k3g))
            capt.append(1 - np.linalg.norm(k3s - k3t) / np.linalg.norm(k3t))
        print(f"{li + 1:5d}   {np.mean(nts):.4e}   {np.mean(ngs):.4e}   "
              f"{np.mean(ratios):5.3f}   half-to-half "
              f"{abs(ratios[0] - ratios[1]):.4f}      "
              f"{np.mean(capg):8.3f}      {np.mean(capt):8.3f}")
    print()
    print("If the half-to-half spread is far below (ratio - 1), the gap is "
          "real, not sampling noise.")
    print("ratio > 1 means the estimator's analytic kappa_3 is systematically "
          "TOO SMALL, which is")
    print("exactly what a damp optimum above 1 compensates for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
