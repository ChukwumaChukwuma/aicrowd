#!/usr/bin/env python
"""ADVERSARIAL: per-MLP anatomy of the claimed 2.11x.

The headline is a suite *mean* over 8 MLPs.  A mean is not a mechanism.  This
script reports, per MLP:

  * unbiased two-half MSE for the baseline, mehler_k4 and edgeworth;
  * the improvement ratio, so a gain carried by one outlier is visible;
  * whether the correction ever makes an individual MLP WORSE;
  * the suite mean recomputed with the largest contributor dropped
    (leave-one-out), which is the cheap test for outlier dependence.

Also reports the mean over MLPs of the *ratio* alongside the ratio of the
means; when those disagree the headline number is being carried by whichever
MLP has the largest absolute error, not by a uniform effect.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import kernels  # noqa: E402
from whestfloor.harness import run_billed  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402


def collect_T(kernel, suite):
    T, P = [], []
    for i in range(suite.n_mlps):
        w = suite.weights(i)
        pred, _f, _r = run_billed(kernel, w)
        p = pred[-1]
        P.append(p.copy())
        T.append(float(np.mean((p - suite.gt_a[i][-1]) *
                               (p - suite.gt_b[i][-1]))))
        del w
    return np.array(T), np.asarray(P)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--damp-extra", type=float, default=2.25,
                    help="also evaluate edgeworth at this damp")
    args = ap.parse_args()

    s = Suite.load(args.suite)
    ks = {
        "cov_prop_gain": kernels.cov_prop_gain,
        "mehler_k4": functools.partial(kernels.cov_prop_mehler, kmax=4),
        "edge_u1_d1": functools.partial(kernels.cov_prop_edgeworth, kmax=4,
                                        umax=1, damp=1.0),
        "edge_u2_d1": functools.partial(kernels.cov_prop_edgeworth, kmax=4,
                                        umax=2, damp=1.0),
        f"edge_u1_d{args.damp_extra:g}": functools.partial(
            kernels.cov_prop_edgeworth, kmax=4, umax=1, damp=args.damp_extra),
    }

    print(f"# suite={s.name} n_mlps={s.n_mlps} n_per_half={s.n_per_half:,} "
          f"width={s.width} depth={s.depth}")
    print(f"# mlp_seeds={s.mlp_seeds}")
    print()

    res = {}
    for nm, fn in ks.items():
        res[nm] = collect_T(fn, s)

    names = list(ks)
    print("mlp_seed  " + "  ".join(f"{n:>13s}" for n in names))
    print("-" * (10 + 15 * len(names)))
    for i in range(s.n_mlps):
        print(f"{s.mlp_seeds[i]:8d}  " +
              "  ".join(f"{res[n][0][i]:13.4e}" for n in names))
    print("-" * (10 + 15 * len(names)))
    print("    MEAN  " + "  ".join(f"{res[n][0].mean():13.4e}" for n in names))
    print()

    base = res["cov_prop_gain"][0]
    meh = res["mehler_k4"][0]
    edg = res["edge_u1_d1"][0]

    print("Improvement of edgeworth(u1,damp=1) over each reference:")
    for refn, ref in (("cov_prop_gain", base), ("mehler_k4", meh)):
        ratio_of_means = ref.mean() / edg.mean()
        mean_of_ratios = float(np.mean(ref / edg))
        worse = int(np.sum(edg > ref))
        per = ref / edg
        print(f"  vs {refn:15s} ratio-of-means {ratio_of_means:6.3f}x   "
              f"mean-of-ratios {mean_of_ratios:6.3f}x   "
              f"per-MLP {per.min():.3f} .. {per.max():.3f}   "
              f"WORSE on {worse}/{s.n_mlps}")
    print()

    # leave-one-out: how much of the headline rides on a single MLP?
    print("Leave-one-out on the vs-cov_prop_gain headline "
          "(ratio of suite means with MLP i removed):")
    full = base.mean() / edg.mean()
    loo = []
    for i in range(s.n_mlps):
        m = np.ones(s.n_mlps, dtype=bool)
        m[i] = False
        loo.append(base[m].mean() / edg[m].mean())
    loo = np.array(loo)
    j = int(np.argmin(loo))
    print(f"  full {full:.3f}x   LOO range {loo.min():.3f} .. {loo.max():.3f}"
          f"   (dropping seed {s.mlp_seeds[j]} gives the minimum)")
    print()

    # is the headline stable to the choice of denominator kernel?
    print("Same table, at the empirically optimal damp:")
    de = res[f"edge_u1_d{args.damp_extra:g}"][0]
    print(f"  vs cov_prop_gain  {base.mean() / de.mean():.3f}x   "
          f"vs mehler_k4 {meh.mean() / de.mean():.3f}x   "
          f"WORSE than damp=1 on {int(np.sum(de > edg))}/{s.n_mlps} MLPs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
