#!/usr/bin/env python
"""Step 1 of the MLMC programme: measure the level variances BEFORE building.

Multilevel Monte Carlo over cheap surrogate networks rests on one quantitative
claim:

    Var( f_{r_k}(x) - f_{r_{k-1}}(x) )   falls fast in the surrogate rank r,

with the two networks driven by the SAME input draw.  If it does, the
telescoping identity

    E[f_full] = E[f_{r_0}] + sum_k E[f_{r_k} - f_{r_{k-1}}]

can be estimated with N_k ~ sqrt(V_k / C_k) samples per level at total cost-to-
accuracy ``(sum_k sqrt(V_k C_k))^2`` instead of plain MC's ``V * C_full``.

Four surrogate families are measured, because the obvious one fails for a
trivially fixable reason and killing the mechanism on that would be unfair:

``svd``
    ``W_r = U_r S_r V_r^T``, the plain SVD truncation named in the brief.
``svd_rescale``
    the same, with every output column rescaled to its original norm.  A random
    square matrix has a quarter-circle spectrum, so ``W_r`` keeps only a
    fraction of ``||W||_F``; without the rescale the surrogate's activations
    decay by ``keep^(depth/2)`` and the network outputs *zero*.
``covproj``
    ``W~ = P_k W`` with ``P_k`` the ORACLE projector onto the top-k eigenspace
    of the layer's measured activation second moment ``E[h h^T]``.  This is the
    right truncation for this problem: the thing that must survive is W's action
    on the activations actually reaching it, not W itself.  Uses measured
    second moments, i.e. an oracle a real estimator could only approximate, so
    it upper-bounds the family.
``sketch``
    ``W V V^T`` for a fixed random ``V`` (the cheap sketch alternative to an
    SVD), with the same column rescale.

Everything here is raw NumPy (research only, never shipped); FLOP costs are the
analytic flopscope prices, cross-checked against a real BudgetContext in
``scripts/23_mlmc_endtoend.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.official_seeds import make_official_mlp  # noqa: E402

WIDTH = 256
DEPTH = 32


# ---------------------------------------------------------------------------
# Analytic flopscope prices (float32; matmul and maximum both weight 1.0)
# ---------------------------------------------------------------------------
def matmul_flops(b: int, m: int, n: int) -> int:
    return b * n * (2 * m - 1)


def cost_full_per_sample(n: int = WIDTH, depth: int = DEPTH) -> int:
    return depth * (matmul_flops(1, n, n) + n)


def cost_rank_per_sample(r: int, n: int = WIDTH, depth: int = DEPTH) -> int:
    if r >= n:
        return cost_full_per_sample(n, depth)
    return depth * (matmul_flops(1, n, r) + matmul_flops(1, r, n) + n)


def cost_draw_per_sample(n: int = WIDTH) -> int:
    return 16 * n


def svd_topk_flops(k: int, m: int = WIDTH, n: int = WIDTH) -> int:
    """flopscope linalg.svd top-k: min(4mnk, economy = 6ab^2 + 20b^3)."""
    a, b = max(m, n), min(m, n)
    economy = 6 * a * b * b + 20 * b ** 3
    if k is None or k >= b:
        return economy
    return min(4 * m * n * k, economy)


# ---------------------------------------------------------------------------
# Surrogate construction
# ---------------------------------------------------------------------------
def _colnorm_rescale(A, B, w):
    """Scale each output column of ``A@B`` back to ``||w[:, j]||``."""
    approx_sq = np.einsum("ik,kj,il,lj->j", A, B, A, B, optimize=True)
    tgt = (w.astype(np.float64) ** 2).sum(axis=0)
    s = np.sqrt(np.maximum(tgt, 1e-30) / np.maximum(approx_sq, 1e-30))
    return A, (B * s[None, :])


def build_surrogates(weights, ranks, mode, second_moments=None, seed=0):
    """Return {r: [ (A_l, B_l) or None ]} with the surrogate ``W~ = A @ B``."""
    n = weights[0].shape[0]
    out = {}
    rng = np.random.default_rng(seed)
    svds = None
    if mode in ("svd", "svd_rescale"):
        svds = [np.linalg.svd(w.astype(np.float64), full_matrices=False)
                for w in weights]
    eigs = None
    if mode == "covproj":
        eigs = []
        for M in second_moments:
            ev, U = np.linalg.eigh(M)
            eigs.append(U[:, ::-1])  # descending
    for r in ranks:
        lst = []
        for li, w in enumerate(weights):
            if r >= n:
                lst.append(None)
                continue
            if mode == "svd":
                u, s, vt = svds[li]
                A = u[:, :r].astype(np.float32)
                B = (s[:r, None] * vt[:r]).astype(np.float32)
            elif mode == "svd_rescale":
                u, s, vt = svds[li]
                A = u[:, :r]
                B = s[:r, None] * vt[:r]
                A, B = _colnorm_rescale(A, B, w)
                A, B = A.astype(np.float32), B.astype(np.float32)
            elif mode == "covproj":
                # W~ = P_r W with P_r = Q Q^T the ORACLE projector onto the
                # top-r eigenspace of the measured E[h h^T].  Deliberately NOT
                # rescaled: the projector already preserves W's action on the
                # activations that actually reach it, and forcing the column
                # norms back up double-counts the discarded directions (it
                # amplifies by 1/sqrt(1-cap) per layer and overflows by layer
                # 32).
                Q = eigs[li][:, :r]                     # (n, r)
                A = Q.astype(np.float32)
                B = (Q.T @ w.astype(np.float64)).astype(np.float32)  # (r, n)
            elif mode == "sketch":
                V, _ = np.linalg.qr(rng.standard_normal((n, r)))
                A = V
                B = V.T @ w.astype(np.float64)
                A, B = _colnorm_rescale(A, B, w)
                A, B = A.astype(np.float32), B.astype(np.float32)
            else:
                raise ValueError(mode)
            lst.append((A, B))
        out[r] = lst
    spec = (np.array([s for (_, s, _) in svds]) if svds is not None
            else np.zeros((len(weights), n)))
    return out, spec


def forward(x, weights, fac):
    h = x
    for li, w in enumerate(weights):
        f = fac[li]
        if f is None:
            h = np.maximum(h @ w, 0.0, dtype=np.float32)
        else:
            A, B = f
            h = np.maximum((h @ A) @ B, 0.0, dtype=np.float32)
    return h


def measure_second_moments(weights, n_samples, chunk, seed):
    """Oracle ``E[h_l h_l^T]`` for the INPUT of every layer (h_0 = x)."""
    n = weights[0].shape[0]
    M = [np.zeros((n, n)) for _ in range(len(weights))]
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        h = rng.standard_normal((nb, n), dtype=np.float32)
        for li, w in enumerate(weights):
            hd = h.astype(np.float64)
            M[li] += hd.T @ hd
            h = np.maximum(h @ w, 0.0, dtype=np.float32)
        done += nb
    return [m / done for m in M]


# ---------------------------------------------------------------------------
def measure_one(mlp_seed, ranks, n_samples, chunk, sample_seed, mode,
                sm_samples=4096):
    weights = make_official_mlp(WIDTH, DEPTH, mlp_seed)
    sm = None
    if mode == "covproj":
        sm = measure_second_moments(weights, sm_samples, chunk,
                                    sample_seed + 555)
    fac, spec = build_surrogates(weights, ranks, mode, sm, seed=mlp_seed & 0xFFFF)

    L = len(ranks)
    n = WIDTH
    S = np.zeros((L, n))
    Q = np.zeros((L, n))
    DS = np.zeros((L, n))     # adjacent-level difference (k vs k-1)
    DQ = np.zeros((L, n))
    FS = np.zeros((L, n))     # difference against the FULL network
    FQ = np.zeros((L, n))

    rng = np.random.default_rng(sample_seed)
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        ys = []
        for r in ranks:
            ys.append(forward(x, weights, fac[r]).astype(np.float64))
        yfull = ys[-1] if ranks[-1] >= n else forward(
            x, weights, [None] * DEPTH).astype(np.float64)
        for k in range(L):
            S[k] += ys[k].sum(axis=0)
            Q[k] += (ys[k] ** 2).sum(axis=0)
            if k:
                d = ys[k] - ys[k - 1]
                DS[k] += d.sum(axis=0)
                DQ[k] += (d * d).sum(axis=0)
            g = ys[k] - yfull
            FS[k] += g.sum(axis=0)
            FQ[k] += (g * g).sum(axis=0)
        done += nb

    N = float(done)
    mean, dmean, fmean = S / N, DS / N, FS / N
    return {
        "mlp_seed": int(mlp_seed),
        "mode": mode,
        "ranks": list(ranks),
        "n_samples": int(done),
        "level_var": (Q / N - mean ** 2).mean(axis=1).tolist(),
        "level_meansq": (mean ** 2).mean(axis=1).tolist(),
        "diff_var": (DQ / N - dmean ** 2).mean(axis=1).tolist(),
        "vs_full_var": (FQ / N - fmean ** 2).mean(axis=1).tolist(),
        "vs_full_biassq": (fmean ** 2).mean(axis=1).tolist(),
        "spectrum": spec.mean(axis=0).tolist(),
    }


def ladder_scan(ranks, lv, dv, V_full, C_full):
    """Cost-to-accuracy of every contiguous sub-ladder ending at the full net."""
    inner = [r for r in ranks if r < WIDTH]
    scored = []
    for m in range(0, len(inner) + 1):
        for sub in itertools.combinations(inner, m):
            lad = list(sub) + [WIDTH]
            idx = [ranks.index(r) for r in lad]
            ok, tot = True, 0.0
            for j, r in enumerate(lad):
                if j == 0:
                    Vk = lv[idx[0]]
                    Ck = cost_rank_per_sample(r) + cost_draw_per_sample()
                else:
                    if idx[j] != idx[j - 1] + 1:
                        ok = False
                        break
                    Vk = dv[idx[j]]
                    Ck = (cost_rank_per_sample(r)
                          + cost_rank_per_sample(lad[j - 1])
                          + cost_draw_per_sample())
                tot += float(np.sqrt(max(Vk, 0.0) * Ck))
            if ok and tot > 0:
                scored.append(((V_full * C_full) / tot ** 2, lad))
    scored.sort(reverse=True)
    return scored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mlps", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--ranks", type=str, default="4,8,16,32,64,128,256")
    ap.add_argument("--modes", type=str,
                    default="svd,svd_rescale,covproj,sketch")
    ap.add_argument("--suite", type=str,
                    default=os.path.join(os.environ.get("WHEST_ARTIFACTS", "."),
                                         "suites", "official_mini.npz"))
    ap.add_argument("--bar", type=float, default=20.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",") if x]
    modes = [m for m in args.modes.split(",") if m]
    z = np.load(args.suite, allow_pickle=False)
    seeds = [int(s) for s in z["mlp_seeds"][: args.n_mlps]]

    C_full = cost_full_per_sample() + cost_draw_per_sample()
    print(f"# MLMC level-variance probe   ranks={ranks}")
    print(f"# n_samples={args.n_samples}  n_mlps={len(seeds)}  "
          f"C_full={C_full:,} FLOP/sample   bar={args.bar}x")

    all_res = {}
    summary = {}
    for mode in modes:
        results = []
        for i, ms in enumerate(seeds):
            t0 = time.time()
            r = measure_one(ms, ranks, args.n_samples, args.chunk,
                            sample_seed=987_654 + i, mode=mode)
            r["wall_s"] = time.time() - t0
            results.append(r)
        all_res[mode] = results

        lv = np.mean([r["level_var"] for r in results], axis=0)
        dv = np.mean([r["diff_var"] for r in results], axis=0)
        fv = np.mean([r["vs_full_var"] for r in results], axis=0)
        fb = np.mean([r["vs_full_biassq"] for r in results], axis=0)
        lm = np.mean([r["level_meansq"] for r in results], axis=0)
        V_full = lv[-1]

        print()
        print(f"=== mode = {mode} ===   V_full = {V_full:.6f}  "
              f"(mean^2 = {lm[-1]:.4f})")
        hdr = ("   r     Var(f_r)    E[f_r]^2   Var(f_r-f_prev)  "
               "Var(f_r-f_full)  /V      rho(f_r,f_full)")
        print(hdr)
        print("   " + "-" * (len(hdr) - 3))
        for k, r in enumerate(ranks):
            ratio = fv[k] / V_full
            rho = float(np.sqrt(max(0.0, 1.0 - ratio / 2.0))) if ratio < 2 else 0.0
            # rho from Var(a-b) = Va + Vb - 2 rho sqrt(Va Vb)
            va, vb = lv[k], V_full
            if va > 0:
                rho = (va + vb - fv[k]) / (2.0 * np.sqrt(va * vb))
            print(f"   {r:5d} {lv[k]:11.6f} {lm[k]:10.5f} {dv[k]:15.4e} "
                  f"{fv[k]:15.4e} {ratio:8.4f}  {rho:8.5f}")

        scored = ladder_scan(ranks, lv, dv, V_full, C_full)
        print(f"   best ladders (gain over plain MC):")
        for g, lad in scored[:5]:
            print(f"     {str(lad):34s} {g:8.3f}x")
        summary[mode] = {"best_gain": float(scored[0][0]),
                         "best_ladder": scored[0][1],
                         "V_full": float(V_full),
                         "level_var": lv.tolist(),
                         "diff_var": dv.tolist(),
                         "vs_full_var": fv.tolist(),
                         "vs_full_biassq": fb.tolist()}

    print()
    print("== VERDICT ==")
    best_mode = max(summary, key=lambda m: summary[m]["best_gain"])
    bg = summary[best_mode]["best_gain"]
    for m in modes:
        print(f"   {m:14s} best gain {summary[m]['best_gain']:8.3f}x  "
              f"ladder {summary[m]['best_ladder']}")
    print(f"   BAR {args.bar}x -> "
          f"{'PASS' if bg >= args.bar else 'FAIL'} "
          f"(best {bg:.3f}x, mode {best_mode})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "raw": all_res,
                       "C_full": int(C_full), "bar": args.bar,
                       "n_samples": args.n_samples,
                       "mlp_seeds": seeds}, fh)
        print(f"   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
