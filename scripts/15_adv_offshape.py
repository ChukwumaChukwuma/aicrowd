#!/usr/bin/env python
"""ADVERSARIAL: does the correction survive off the (256, 32) shape?

The whole claim was measured at one width and one depth.  A term whose
coefficient was chosen (or whose UMAX/KMAX were chosen) at that shape is a
tuning artefact if it stops helping elsewhere.  This scores, per suite shape:

    cov_prop_gain, mehler_k4, edgeworth(u1, damp=1), edgeworth(u1, damp=2.25)

with the unbiased two-half MSE, and reports the improvement factor.  It also
re-finds the optimal ``damp`` per shape: if the argmin moves with shape, the
coefficient is not a derived constant, it is a per-shape fit.
"""

from __future__ import annotations

import argparse
import functools
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import kernels  # noqa: E402
from whestfloor.harness import run_billed  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402


def T_per_mlp(kernel, s):
    out = []
    for i in range(s.n_mlps):
        w = s.weights(i)
        pred, _f, _r = run_billed(kernel, w)
        p = pred[-1]
        out.append(float(np.mean((p - s.gt_a[i][-1]) * (p - s.gt_b[i][-1]))))
        del w
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--damps", type=str, default="0,0.5,1.0,1.5,2.0,2.5,3.0")
    args = ap.parse_args()

    paths = []
    for pat in args.suites:
        paths.extend(sorted(glob.glob(pat)))

    damps = [float(x) for x in args.damps.split(",")]
    hdr = ("shape      n   gain        mehler_k4    edge d=1     "
           "edge d=2.25   gain/edge1  meh/edge1  best_damp")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for p in paths:
        s = Suite.load(p)
        tg = T_per_mlp(kernels.cov_prop_gain, s).mean()
        tm = T_per_mlp(functools.partial(kernels.cov_prop_mehler, kmax=4),
                       s).mean()
        sweep = {}
        for d in damps:
            sweep[d] = T_per_mlp(functools.partial(
                kernels.cov_prop_edgeworth, kmax=4, umax=1, damp=d), s).mean()
        t1 = sweep.get(1.0)
        if t1 is None:
            t1 = T_per_mlp(functools.partial(kernels.cov_prop_edgeworth,
                                             kmax=4, umax=1, damp=1.0), s).mean()
        t225 = T_per_mlp(functools.partial(kernels.cov_prop_edgeworth, kmax=4,
                                           umax=1, damp=2.25), s).mean()
        ds = np.array(sorted(sweep))
        vs = np.array([sweep[d] for d in ds])
        c = np.polyfit(ds, vs, 2)
        argmin = -c[1] / (2 * c[0]) if c[0] > 0 else float("nan")
        gmin = ds[int(np.argmin(vs))]
        print(f"{s.width:4d}x{s.depth:<3d} {s.n_mlps:3d}  {tg:.4e}  "
              f"{tm:.4e}  {t1:.4e}  {t225:.4e}   {tg / t1:7.3f}x  "
              f"{tm / t1:7.3f}x   grid {gmin:.2f} / fit {argmin:.2f}")
        rows.append((s.width, s.depth, tg, tm, t1, sweep))

    print()
    print("damp sweep per shape (unbiased true MSE):")
    print("shape     " + "  ".join(f"d={d:<5g}" for d in damps))
    for wdt, dep, _tg, _tm, _t1, sweep in rows:
        print(f"{wdt:4d}x{dep:<3d}  " +
              "  ".join(f"{sweep[d]:.2e}" for d in damps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
