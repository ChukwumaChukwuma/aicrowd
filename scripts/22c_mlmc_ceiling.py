#!/usr/bin/env python
"""Close the MLMC question: the per-layer compressibility profile and the
structural ceiling on any rank-based level-0.

Three parts.

1. **Additivity check.**  ``scripts/22b`` calibrates ``Var(f-f~)/V = K d^2`` by
   perturbing all 32 layers at once.  Extrapolating to a per-layer rank budget
   needs the injections to add in quadrature, so this perturbs a *subset* of
   layers and checks the predicted proportionality.

2. **Per-layer compressibility.**  ``d_l(r) = sqrt(1 - cap_l(r))`` where
   ``cap_l(r)`` is the fraction of ``E||h_l||^2`` inside the top-r eigenspace of
   the measured activation second moment.  This is the ORACLE input-adapted
   projector — no cheaper surrogate in the family can do better — and it decides
   how few directions each layer can be run in.

3. **The ceiling.**  For a factored layer ``x @ A_r @ B_r`` flopscope charges
   ``4nr`` against ``2n^2`` for the dense layer, so a level is only cheaper at
   ``r < n/2``.  With the per-layer rank profile fixed, this prints the actual
   MLMC cost-to-accuracy ``(sum_k sqrt(V_k C_k))^2`` and compares it to plain
   MC's ``V C_full``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.official_seeds import make_official_mlp  # noqa: E402

WIDTH = 256
DEPTH = 32


def matmul_flops(b, m, n):
    return b * n * (2 * m - 1)


def cost_dense_layer(n=WIDTH):
    return matmul_flops(1, n, n) + n


def cost_rank_layer(r, n=WIDTH):
    return matmul_flops(1, n, r) + matmul_flops(1, r, n) + n


def run_pair(W, Wt, n_samples, chunk, seed):
    n = W[0].shape[0]
    Sf = np.zeros(n); Qf = np.zeros(n)
    Sd = np.zeros(n); Qd = np.zeros(n)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        h = x
        for w in W:
            h = np.maximum(h @ w, 0.0, dtype=np.float32)
        g = x
        for w in Wt:
            g = np.maximum(g @ w, 0.0, dtype=np.float32)
        hd = h.astype(np.float64); gd = g.astype(np.float64)
        Sf += hd.sum(axis=0); Qf += (hd * hd).sum(axis=0)
        d = hd - gd
        Sd += d.sum(axis=0); Qd += (d * d).sum(axis=0)
        done += nb
    N = float(done)
    mf = Sf / N
    vf = (Qf / N - mf ** 2).mean()
    md = Sd / N
    vd = (Qd / N - md ** 2).mean()
    return float(vf), float(vd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mlps", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--decorr", type=str,
                    default=os.path.join(os.environ.get("WHEST_ARTIFACTS", "."),
                                         "mlmc", "decorr.json"))
    ap.add_argument("--suite", type=str,
                    default=os.path.join(os.environ.get("WHEST_ARTIFACTS", "."),
                                         "suites", "official_mini.npz"))
    ap.add_argument("--bar", type=float, default=20.0)
    args = ap.parse_args()

    D = json.load(open(args.decorr))
    K = float(D["K"])
    cap = np.array(D["act_capture"])          # (depth, n)
    z = np.load(args.suite, allow_pickle=False)
    seeds = [int(s) for s in z["mlp_seeds"][: args.n_mlps]]

    # ---------------- 1. additivity ----------------
    print("== 1. do per-layer perturbations add in quadrature? ==")
    print("   perturb only the LAST m layers at d=1e-2; prediction is "
          "(m/32) x the all-layer value")
    d = 1e-2
    allv = None
    for m in (32, 16, 8, 4):
        vals = []
        for i, ms in enumerate(seeds):
            W = make_official_mlp(WIDTH, DEPTH, ms)
            rng = np.random.default_rng(31337 + i)
            s = float(np.sqrt(2.0 / WIDTH))
            keep = float(np.sqrt(1 - d * d))
            Wt = []
            for li, w in enumerate(W):
                if li >= DEPTH - m:
                    Wt.append((keep * w + d * (rng.standard_normal(w.shape) * s)
                               ).astype(np.float32))
                else:
                    Wt.append(w)
            vf, vd = run_pair(W, Wt, args.n_samples, args.chunk, 777_000 + i)
            vals.append(vd / vf)
        v = float(np.mean(vals))
        if allv is None:
            allv = v
        print(f"   m={m:3d}  V/V_full = {v:.5e}   "
              f"predicted {allv * m / 32:.5e}   ratio {v / (allv * m / 32):.3f}")

    # ---------------- 2. per-layer compressibility ----------------
    print()
    print("== 2. oracle input-adapted rank needed per layer ==")
    print("   d_l(r) = sqrt(1 - cap_l(r));  cap from measured E[h_l h_l^T]")
    print()
    print("   layer " + "".join(f"{r:>10d}" for r in (4, 8, 16, 32, 64, 128)))
    for li in (0, 1, 2, 3, 5, 7, 11, 15, 23, 31):
        row = "".join(f"{np.sqrt(max(0.0, 1 - cap[li, r - 1])):10.5f}"
                      for r in (4, 8, 16, 32, 64, 128))
        print(f"   {li + 1:5d} " + row)

    # the per-layer rank needed to hit a target total V/V_full
    print()
    print("   minimum rank per layer for a TOTAL Var(f-f~)/V of:")
    for tgt in (5e-2, 1e-3):
        # split the budget evenly: per-layer d_l^2 <= tgt/(K/32 * 32) = tgt/K
        per = tgt / K
        ranks = []
        for li in range(DEPTH):
            need = np.searchsorted(cap[li], 1.0 - per) + 1
            ranks.append(min(need, WIDTH))
        cost = sum(cost_rank_layer(r) if r < WIDTH // 2 else cost_dense_layer()
                   for r in ranks)
        dense = DEPTH * cost_dense_layer()
        print(f"     target {tgt:.0e}:  ranks = {ranks}")
        print(f"       -> surrogate cost / dense cost = {cost / dense:.4f} "
              f"(a level is only cheaper at r < {WIDTH // 2})")

    # ---------------- 3. the ceiling ----------------
    print()
    print("== 3. MLMC cost-to-accuracy ceiling ==")
    print("   gain = 1 / (sum_k sqrt(v_k c_k))^2 ,  v = V_k/V , c = C_k/C_full")
    print("   two-level (rank r, then full).  v_0 ~ 1 for any surrogate that")
    print("   reproduces the output fluctuation at all; c_0 = 2r/n; the top")
    print("   level must run the full net, so c_1 = 1 + 2r/n.")
    print()
    print("     r    c_0      v_1 (predicted)   v_1 (needed for 20x)   gain")
    for r in (2, 4, 8, 16, 32, 64, 128):
        c0 = 2.0 * r / WIDTH
        c1 = 1.0 + c0
        # predicted v_1 from the oracle rank profile, saturating at 2
        per_layer = np.array([1.0 - cap[li, min(r, WIDTH) - 1]
                              for li in range(DEPTH)])
        v1 = min(2.0, K / DEPTH * per_layer.sum())
        # what v_1 would be needed for a 20x gain
        need = (1.0 / np.sqrt(args.bar)) - np.sqrt(1.0 * c0)
        need_v = (need ** 2 / c1) if need > 0 else float("nan")
        gain = 1.0 / (np.sqrt(1.0 * c0) + np.sqrt(v1 * c1)) ** 2
        print(f"   {r:5d}  {c0:6.4f}   {v1:15.4e}   "
              f"{need_v:20.4e}   {gain:8.4f}x")

    print()
    print(f"   the level-0 discount alone caps the gain at n/(2 r_0) = "
          f"{WIDTH // 2} / r_0, so a {args.bar:.0f}x bar needs r_0 <= "
          f"{WIDTH / (2 * args.bar):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
