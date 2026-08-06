#!/usr/bin/env python
"""ADVERSARIAL: how much slower can the grader's box be before the score gets
WORSE?

``C = F + lambda * R`` with ``lambda = 1e11`` and ``R`` = residual wall time,
i.e. participant Python that is neither backend nor flopscope overhead.  ``F``
is machine independent; ``R`` is not.  The multiplier is
``max(0.1, C/B)``, so every second of ``R`` above

    R_crit = (0.1 * B - F) / lambda

multiplies the score by more than the 0.1 floor and makes it strictly worse.

Measured here, on many repeats rather than one draw:

  * the distribution of ``R`` (mean, sd, min, max, p95) for the shipped
    estimator and for the two baselines;
  * ``R_crit`` and the slowdown factor ``R_crit / R`` -- the safety margin;
  * the same margin against the p95 of ``R`` rather than its mean, because a
    grader sees one draw, not an average;
  * total ``predict`` wall clock against ``predict_timeout_s = 30`` and
    ``wall_time_limit_s = 60``;
  * the wall/backend/overhead/residual split, so it is visible how much of the
    total is the part that scales with machine speed.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

from whestfloor import kernels  # noqa: E402
from whestfloor.contract import (  # noqa: E402
    FLOP_BUDGET,
    LAMBDA_FLOPS_PER_SECOND,
    MULTIPLIER_FLOOR,
    PREDICT_TIMEOUT_S,
    WALL_TIME_LIMIT_S,
)
from whestfloor.mc import make_mlp  # noqa: E402


def timed_run(kernel, W):
    """Bill exactly as whestfloor.harness.run_billed does: the weight
    conversion is OUTSIDE the context, as it is for the grader."""
    fw = [fnp.asarray(w) for w in W]
    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as ctx:
        out = kernel(fw)
    wall = time.perf_counter() - t0
    _ = np.asarray(out)
    return (int(ctx.flops_used), float(ctx.residual_wall_time_s), wall,
            float(getattr(ctx, "wall_time_s", np.nan) or np.nan),
            float(getattr(ctx, "backend_time_s", np.nan) or np.nan),
            float(getattr(ctx, "flopscope_overhead_time_s", np.nan) or np.nan))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    args = ap.parse_args()

    W = make_mlp(args.width, args.depth, 777)
    ks = {
        "cov_prop_gain": kernels.cov_prop_gain,
        "mehler_k4": functools.partial(kernels.cov_prop_mehler, kmax=4),
        "edgeworth_u1": functools.partial(kernels.cov_prop_edgeworth, kmax=4,
                                          umax=1),
        "edgeworth_u2": functools.partial(kernels.cov_prop_edgeworth, kmax=4,
                                          umax=2),
    }

    B = FLOP_BUDGET
    lam = LAMBDA_FLOPS_PER_SECOND
    print(f"# {args.width}x{args.depth}  B={B:.4g}  lambda={lam:.3g}  "
          f"floor={MULTIPLIER_FLOOR}  repeats={args.repeats}")
    print(f"# R_crit = (0.1*B - F)/lambda   -- residual above this makes the "
          f"score strictly worse")
    print()
    hdr = ("kernel          F(flops)      R mean    R sd     R min    R p95"
           "    R max    C/B mean  C/B p95   R_crit   margin(mean) "
           "margin(p95)")
    print(hdr)
    print("-" * len(hdr))

    detail = {}
    for nm, fn in ks.items():
        rs, walls, splits = [], [], []
        F = None
        for _ in range(args.repeats):
            F, r, wall, w2, b2, o2 = timed_run(fn, W)
            rs.append(r)
            walls.append(wall)
            splits.append((w2, b2, o2, r))
        rs = np.array(rs)
        walls = np.array(walls)
        r_crit = (MULTIPLIER_FLOOR * B - F) / lam
        cb_mean = (F + lam * rs.mean()) / B
        p95 = float(np.percentile(rs, 95))
        cb_p95 = (F + lam * p95) / B
        print(f"{nm:14s} {F:11,d}  {rs.mean():8.4f} {rs.std():8.4f} "
              f"{rs.min():8.4f} {p95:8.4f} {rs.max():8.4f} "
              f"{cb_mean:9.4f} {cb_p95:8.4f} {r_crit:8.4f} "
              f"{r_crit / rs.mean():11.2f}x {r_crit / p95:11.2f}x")
        detail[nm] = (F, rs, walls, np.array(splits), r_crit)

    print()
    print("Wall-clock of one predict (the 30 s predict_timeout_s and 60 s "
          "wall_time_limit_s):")
    for nm, (F, rs, walls, splits, _rc) in detail.items():
        w, b, o, r = np.nanmean(splits, axis=0)
        print(f"  {nm:14s} total {walls.mean():6.3f}s "
              f"(max {walls.max():6.3f}s)   ctx wall {w:6.3f}  backend {b:6.3f}"
              f"  overhead {o:6.3f}  residual {r:6.3f}   "
              f"headroom to 30 s: {PREDICT_TIMEOUT_S / max(walls.max(), 1e-9):.1f}x"
              f"   to 60 s: {WALL_TIME_LIMIT_S / max(walls.max(), 1e-9):.1f}x")

    F, rs, walls, splits, r_crit = detail["edgeworth_u1"]
    print()
    print("SHIPPED ESTIMATOR — safety margin on a slower grader")
    print(f"  F               = {F:,}  ({F / B:.4f} of B, machine independent)")
    print(f"  R here          = {rs.mean():.4f} s  +- {rs.std():.4f} "
          f"({rs.std() / rs.mean():.1%} run-to-run)   "
          f"[{rs.min():.4f} .. {rs.max():.4f}]")
    print(f"  R_crit          = {r_crit:.4f} s")
    print(f"  slowdown to cross the floor: {r_crit / rs.mean():.2f}x on the "
          f"mean, {r_crit / np.percentile(rs, 95):.2f}x on the p95, "
          f"{r_crit / rs.max():.2f}x on the worst draw seen")
    print(f"  at 2x slower the multiplier is "
          f"{max(MULTIPLIER_FLOOR, (F + lam * 2 * rs.mean()) / B):.4f}; "
          f"at 4x {max(MULTIPLIER_FLOOR, (F + lam * 4 * rs.mean()) / B):.4f}; "
          f"at 8x {max(MULTIPLIER_FLOOR, (F + lam * 8 * rs.mean()) / B):.4f}")
    Fg, rg, *_ = detail["cov_prop_gain"]
    rcg = (MULTIPLIER_FLOOR * B - Fg) / lam
    print(f"  by comparison the baseline cov_prop_gain tolerates "
          f"{rcg / rg.mean():.2f}x")
    print()

    # at what slowdown does the ADJUSTED score stop beating the baseline?
    print("  Slowdown at which the shipped estimator's ADJUSTED score stops "
          "beating cov_prop_gain")
    print("  (adjusted = mse * max(0.1, C/B); mse is machine independent):")
    for label, mse_g, mse_e in (("quick suite", 6.8218e-5, 3.2338e-5),
                                ("adv suite  ", 6.6794e-5, 3.8526e-5),
                                ("all shapes ", 1.0, 1.0 / 1.6)):
        lo, hi = 1.0, 1e4
        for _ in range(200):
            k = 0.5 * (lo + hi)
            me = max(MULTIPLIER_FLOOR, (F + lam * k * rs.mean()) / B)
            mg = max(MULTIPLIER_FLOOR, (Fg + lam * k * rg.mean()) / B)
            if mse_e * me < mse_g * mg:
                lo = k
            else:
                hi = k
        print(f"    {label}: mse ratio {mse_g / mse_e:.3f}x  ->  break-even "
              f"at {lo:.2f}x slower")
    print()
    print("  NOTE: R is measured on a box with other work on it if anything "
          "else is running;")
    print("  run this on an idle machine for the number that should be "
          "quoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
