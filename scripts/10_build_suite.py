#!/usr/bin/env python
"""Generate an evaluation suite: MLP seeds + two independent ground-truth halves.

Artifacts are large and regenerable, so they are written outside the repository
(``$WHEST_ARTIFACTS``, default ``./_artifacts``) and are not committed.  This
script is the sole producer; running it with the same arguments reproduces the
suite bit-for-bit, because every random stream is seeded from the arguments.

Usage
-----
    python scripts/10_build_suite.py --name dev --n-mlps 20 --n-per-half 2000000
    python scripts/10_build_suite.py --name precision --n-mlps 4 --n-per-half 50000000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import DEPTH, WIDTH  # noqa: E402
from whestfloor.suite import build_suite  # noqa: E402


def artifacts_dir() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-mlps", type=int, required=True)
    ap.add_argument("--n-per-half", type=int, required=True)
    ap.add_argument("--mlp-seed-base", type=int, default=0)
    ap.add_argument("--gt-seed-base", type=int, default=1_000_000)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    args = ap.parse_args()

    seeds = [args.mlp_seed_base + i for i in range(args.n_mlps)]
    out = artifacts_dir() / "suites" / f"{args.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    fwd_flops = 2.0 * args.width * args.width * args.depth
    total = fwd_flops * 2 * args.n_per_half * args.n_mlps
    print(
        f"[suite:{args.name}] {args.n_mlps} MLPs x 2 x {args.n_per_half:,} samples "
        f"= {total:.3e} FLOPs of sampling",
        flush=True,
    )

    t0 = time.time()
    suite = build_suite(
        args.name,
        seeds,
        args.n_per_half,
        width=args.width,
        depth=args.depth,
        gt_seed_base=args.gt_seed_base,
        progress=True,
    )
    suite.save(out)
    dt = time.time() - t0
    tau2 = float(suite.final_var.mean()) / args.n_per_half
    print(
        f"[suite:{args.name}] done in {dt / 60:.1f} min -> {out}\n"
        f"  avg final-layer variance v = {float(suite.final_var.mean()):.5f}\n"
        f"  per-half noise variance    tau^2 = v/n_half = {tau2:.3e}\n"
        f"  unbiased-MSE resolution    ~tau^2/sqrt(n*M) = "
        f"{tau2 / (args.width * args.n_mlps) ** 0.5:.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
