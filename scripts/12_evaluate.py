#!/usr/bin/env python
"""End-to-end evaluation of estimator kernels against a suite.

Every kernel is run inside a real flopscope BudgetContext, per MLP, so the
FLOP count and the residual wall time are measured rather than assumed.  The
report always carries, together:

  * ``adjusted_final_layer_score`` -- the leaderboard formula applied locally;
  * ``raw_final_layer_mse``        -- unadjusted, against the local reference;
  * ``unbiased_true_mse (+- se)``  -- the local reference's own noise removed;
  * ``leaderboard_equivalent``     -- ``unbiased_true_mse + v/1e9``, the only
    number comparable to a real leaderboard score, because a local suite's raw
    MSE is dominated by its own much noisier reference;
  * the compute ratio and the seed.

Reporting an adjusted score without the raw MSE, the compute ratio and the
seed is refused by the ledger, so all of them appear here.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import kernels  # noqa: E402
from whestfloor.contract import (  # noqa: E402
    ADJUSTED_FLOOR,
    MEASURED_AVG_VARIANCE,
    RAW_MSE_FLOOR,
)
from whestfloor.harness import evaluate, leaderboard_equivalent  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402


def build_kernels(kmax_list):
    ks = {
        "mean_prop": kernels.mean_prop,
        "cov_prop_gain": kernels.cov_prop_gain,
    }
    for k in kmax_list:
        ks[f"mehler_k{k}"] = functools.partial(kernels.cov_prop_mehler, kmax=k)
    for u in (1, 2, 3):
        ks[f"edgeworth_u{u}"] = functools.partial(
            kernels.cov_prop_edgeworth, kmax=4, umax=u)
    # ablation: identical code path, correction switched off
    ks["edgeworth_u2_ablated"] = functools.partial(
        kernels.cov_prop_edgeworth, kmax=4, umax=2, damp=0.0)
    # complete tree catalogue for kappa_3 (scripts/16), and through kappa_4
    ks["edge3"] = functools.partial(
        kernels.cov_prop_edge3, kmax=4, K2=4, T3=3)
    ks["edge3_ablated"] = functools.partial(
        kernels.cov_prop_edge3, kmax=4, K2=4, T3=3, damp=0.0)
    ks["edge4"] = functools.partial(
        kernels.cov_prop_edgeworth4, kmax=4, K2=4, T3=3, T4=3)
    ks["edge4_g1sq"] = functools.partial(
        kernels.cov_prop_edgeworth4, kmax=4, K2=4, T3=3, T4=3, g1sq=True)
    ks["edge4_ablated"] = functools.partial(
        kernels.cov_prop_edgeworth4, kmax=4, K2=4, T3=3, T4=3,
        damp=0.0, damp4=0.0)
    return ks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--kmax", type=str, default="2,4,8,16,24")
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    suite = Suite.load(args.suite)
    kl = [int(x) for x in args.kmax.split(",") if x]
    ks = build_kernels(kl)
    if args.only:
        keep = set(args.only.split(","))
        ks = {k: v for k, v in ks.items() if k in keep}

    print(f"# suite={suite.name} n_mlps={suite.n_mlps} "
          f"gt_samples={suite.gt_samples:,} "
          f"final_cov={'yes' if suite.final_cov is not None else 'NO (se is a lower bound)'}")
    print(f"# floor: raw {RAW_MSE_FLOOR:.3e}  adjusted {ADJUSTED_FLOOR:.3e}  "
          f"(v = {MEASURED_AVG_VARIANCE})")
    print()
    hdr = ("kernel            flops        C/B      raw_mse     "
           "unbiased_true_mse       lb_equiv_adj   x_floor")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, fn in ks.items():
        r = evaluate(fn, suite, name=name)
        raw_eq, adj_eq = leaderboard_equivalent(r)
        rows.append((name, r, adj_eq))
        se = r.unbiased_true_mse_stderr or 0.0
        print(f"{name:16s} {r.flops_used:11.4g}  {r.compute_ratio:7.4f}  "
              f"{r.final_layer_mse:.3e}  {r.unbiased_true_mse:+.4e} "
              f"+-{se:.1e}  {adj_eq:.4e}  {adj_eq / ADJUSTED_FLOOR:8.1f}")

    print()
    best = min(rows, key=lambda t: t[2])
    print(f"best: {best[0]}  leaderboard-equivalent adjusted "
          f"{best[2]:.4e}  = {best[2] / ADJUSTED_FLOOR:.1f}x the floor")
    print(f"note: {best[1].notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
