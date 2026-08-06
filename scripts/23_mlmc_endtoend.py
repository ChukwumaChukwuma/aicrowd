#!/usr/bin/env python
"""End-to-end MLMC on the official suite, with plain MC through the same path.

Three things happen here.

1. **Cost-model audit.**  ``scripts/22`` prices levels analytically
   (``4 n r`` for a factored layer, ``2 n^2`` dense, ``min(4 m n k, economy)``
   for the top-k SVD).  This runs the real kernel inside a real
   ``flops.BudgetContext`` and checks the analytic price against the billed one,
   because an allocation computed from a wrong cost model would be a wrong
   experiment rather than a wrong result.

2. **Parity.**  ``mlmc_kernel(levels=((256, N),))`` must be bitwise identical to
   ``mc_kernel(n_samples=N)`` -- that is what makes the plain-MC arm a genuine
   ablation through the identical code path rather than a separate program.

3. **The scored comparison.**  Every variant is sized to the same FLOP budget
   with the MLMC-optimal allocation ``N_k ~ sqrt(V_k / C_k)`` computed from the
   variances ``scripts/22`` measured, then scored on the official N=1e9 suite.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import kernels  # noqa: E402
from whestfloor.contract import FLOP_BUDGET  # noqa: E402
from whestfloor.harness import evaluate, run_billed  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402

WIDTH = 256
DEPTH = 32


def matmul_flops(b, m, n):
    return b * n * (2 * m - 1)


def cost_fwd(r, n=WIDTH, depth=DEPTH):
    if r >= n:
        return depth * (matmul_flops(1, n, n) + n)
    return depth * (matmul_flops(1, n, r) + matmul_flops(1, r, n) + n)


def svd_flops(k, m=WIDTH, n=WIDTH):
    a, b = max(m, n), min(m, n)
    return min(4 * m * n * k, 6 * a * b * b + 20 * b ** 3)


def level_costs(ranks, n=WIDTH):
    """(fixed setup FLOPs, per-sample FLOPs per level)."""
    fixed = sum(DEPTH * svd_flops(r) for r in ranks if r < n)
    per = []
    for k, r in enumerate(ranks):
        c = cost_fwd(r) + 16 * n           # forward + the float32 normal draw
        if k:
            c += cost_fwd(ranks[k - 1])    # the coupled partner
        per.append(c)
    return fixed, per


def allocate(ranks, V, target_flops):
    """MLMC-optimal ``N_k ~ sqrt(V_k / C_k)`` filling ``target_flops``."""
    fixed, per = level_costs(ranks)
    avail = max(target_flops - fixed, 1.0)
    s = sum(np.sqrt(max(V[k], 0.0) * per[k]) for k in range(len(ranks)))
    if s <= 0:
        return [0] * len(ranks), fixed, per
    N = [max(1, int(avail * np.sqrt(max(V[k], 0.0) / per[k]) / s))
         for k in range(len(ranks))]
    return N, fixed, per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--levels", type=str,
                    default=os.path.join(os.environ.get("WHEST_ARTIFACTS", "."),
                                         "mlmc", "levels.json"))
    ap.add_argument("--target-flops", type=float, default=2.2e10)
    ap.add_argument("--n-mlps", type=int, default=0, help="0 = whole suite")
    ap.add_argument("--mode", type=str, default="svd",
                    help="which measured surrogate family to size against")
    ap.add_argument("--ladders", type=str,
                    default="256,8-256,16-256,128-256,224-256,192-224-256")
    args = ap.parse_args()

    suite = Suite.load(args.suite)
    if args.n_mlps:
        suite.mlp_seeds = suite.mlp_seeds[: args.n_mlps]
        suite.gt_a = suite.gt_a[: args.n_mlps]
        suite.gt_b = suite.gt_b[: args.n_mlps]
        suite.final_var = suite.final_var[: args.n_mlps]

    L = json.load(open(args.levels))
    probe_ranks = [8, 16, 32, 64, 128, 192, 224, 256]
    S = L["summary"][args.mode]
    lv = dict(zip(probe_ranks, S["level_var"]))         # Var(f_r)
    dv = dict(zip(probe_ranks, S["diff_var"]))          # Var(f_r - f_prev_probe)
    fv = dict(zip(probe_ranks, S["vs_full_var"]))       # Var(f_r - f_full)

    def level_variances(ranks):
        """``V_k`` for one ladder, picking the right measured difference.

        ``diff_var`` is only the coupled difference between rungs that are
        ADJACENT in the probe ladder; a ladder that skips rungs needs the
        difference actually taken.  For the top level (against the full
        network) that is ``vs_full_var``, which is measured at every rank, so
        two-level ladders -- the ones most likely to win -- are always priced
        exactly.  An intermediate skip has no measurement and is refused.
        """
        V = []
        for k, r in enumerate(ranks):
            if k == 0:
                V.append(lv[r])
            elif r >= WIDTH:
                V.append(fv[ranks[k - 1]])
            else:
                i, j = probe_ranks.index(ranks[k - 1]), probe_ranks.index(r)
                if j != i + 1:
                    raise ValueError(
                        f"no measured coupled variance for {ranks[k-1]}->{r}")
                V.append(dv[r])
        return V

    w0 = suite.weights(0)

    # ---------------- 1. cost-model audit ----------------
    print("== 1. analytic price vs flopscope bill ==")
    print("   ranks                     N        analytic F        billed F"
          "     ratio")
    audits = [([256], [512]), ([32, 256], [512, 128]), ([224, 256], [256, 64])]
    for ranks, N in audits:
        fixed, per = level_costs(ranks)
        pred = fixed + sum(n * c for n, c in zip(N, per))
        fn = functools.partial(kernels.mlmc_kernel,
                               levels=tuple(zip(ranks, N)), seed=0)
        _, fl, _ = run_billed(fn, w0)
        print(f"   {str(ranks):16s} {str(N):12s} {pred:15,d} {fl:15,d}"
              f"   {fl / pred:8.4f}")

    # ---------------- 2. parity with plain MC ----------------
    print()
    print("== 2. mlmc(single dense level) vs mc_kernel: identical path? ==")
    a, fa, _ = run_billed(functools.partial(kernels.mlmc_kernel,
                                            levels=((256, 777),), seed=0), w0)
    b, fb, _ = run_billed(functools.partial(kernels.mc_kernel,
                                            n_samples=777, seed=0), w0)
    print(f"   max|diff| = {np.abs(a - b).max():.3e}   "
          f"flops {fa:,} vs {fb:,}   {'IDENTICAL' if (np.array_equal(a, b) and fa == fb) else 'DIFFER'}")
    del w0

    # ---------------- 3. the scored comparison ----------------
    print()
    print(f"== 3. official suite, {len(suite.mlp_seeds)} MLPs, "
          f"target {args.target_flops:.2e} FLOPs ==")
    variants = {}
    ladders = [[int(t) for t in g.split("-")] for g in args.ladders.split(",")]
    for ranks in ladders:
        V = level_variances(ranks)
        N, fixed, per = allocate(ranks, V, args.target_flops)
        name = "mc_plain" if ranks == [WIDTH] else "mlmc_" + "_".join(
            str(r) for r in ranks)
        variants[name] = (tuple(zip(ranks, N)), V, per, fixed)

    # The suite's own mean per-neuron variance, to rescale the probe's V's.
    v_suite = float(np.mean(suite.final_var))
    v_probe = float(S["V_full"])
    print(f"   probe V_full = {v_probe:.6f}; suite avg_variance = "
          f"{v_suite:.6f}; scale {v_suite / v_probe:.4f}")
    print()
    hdr = ("   variant            levels (rank,N)                      "
           "flops        C/B    pred_mse     raw_mse      adjusted   "
           " x_plain  x_pred")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    base = base_pred = None
    rows = []
    for name, (levels, V, per, fixed) in variants.items():
        fn = functools.partial(kernels.mlmc_kernel, levels=levels, seed=0)
        r = evaluate(fn, suite, name=name)
        adj = r.final_layer_mse * r.mean_multiplier
        # predicted MSE = sum_k V_k/N_k, rescaled to this suite, + the 1e9
        # reference's own variance (negligible at 4.9e-11 but it is real).
        pred = sum(max(V[k], 0.0) / max(n, 1) for k, (_, n) in
                   enumerate(levels)) * (v_suite / v_probe) + v_suite / 1e9
        if base is None:
            base, base_pred = r.final_layer_mse, pred
        rows.append((name, levels, r, adj, pred))
        print(f"   {name:18s} {str(levels):36s} {r.flops_used:10.3e} "
              f"{r.compute_ratio:7.4f} {pred:.4e}  {r.final_layer_mse:.4e}  "
              f"{adj:.4e}  {base / r.final_layer_mse:7.3f} "
              f"{base_pred / pred:7.3f}")

    print()
    print("   x_plain > 1 would mean MLMC beats plain MC at matched compute.")
    print("   x_pred is the same ratio from the measured level variances; the")
    print("   two agree except within the ~15% Monte-Carlo noise on a raw_mse")
    print("   estimated over 100 MLPs whose neurons are strongly correlated")
    print("   (participation ratio ~2), so rows near 1.0 are ties with plain MC.")
    best = max(rows[1:], key=lambda t: base / t[2].final_layer_mse) if len(rows) > 1 else None
    if best:
        g = base / best[2].final_layer_mse
        gp = base_pred / best[4]
        print(f"   best MLMC ladder: {best[0]}  ->  measured {g:.3f}x / "
              f"predicted {gp:.3f}x plain MC   (bar 20x) -> "
              f"{'PASS' if g >= 20 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
