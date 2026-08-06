#!/usr/bin/env python
"""Fit the per-layer shrink on one suite, score it on a disjoint one.

A calibrated constant is only legitimate if it transfers.  This script never
reports an in-sample number: `g` is fitted on suite A and scored on suite B,
and then the other way round.  It also reports the argmin on each suite so the
two can be compared directly -- if they coincide, `g` is a property of the
problem rather than of a particular draw of MLPs.
"""
from __future__ import annotations
import argparse, functools, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor import kernels                                    # noqa: E402
from whestfloor.harness import evaluate                           # noqa: E402
from whestfloor.suite import Suite                                # noqa: E402


def curve(suite, gs, kmax=4):
    out = {}
    for g in gs:
        r = evaluate(functools.partial(kernels.cov_prop_shrink, kmax=kmax, g=g),
                     suite, name=f"shrink{g}")
        out[g] = r.unbiased_true_mse
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--gmin", type=float, default=0.9990)
    ap.add_argument("--gmax", type=float, default=1.0002)
    ap.add_argument("--steps", type=int, default=13)
    args = ap.parse_args()
    gs = list(np.linspace(args.gmin, args.gmax, args.steps))
    suites = [Suite.load(p) for p in args.suites]
    curves = {}
    for p, s in zip(args.suites, suites):
        c = curve(s, gs)
        curves[Path(p).stem] = c
        best = min(c, key=c.get)
        print(f"[{Path(p).stem}] argmin g = {best:.6f}   mse {c[best]:.4e}   "
              f"(g=1: {c[min(gs, key=lambda x: abs(x-1.0))]:.4e})", flush=True)
    print()
    names = list(curves)
    print("out-of-sample transfer (fit on one, score on the other):")
    for i, a in enumerate(names):
        for b in names:
            if a == b:
                continue
            g_a = min(curves[a], key=curves[a].get)
            print(f"  g fitted on {a} = {g_a:.6f} -> scored on {b}: "
                  f"{curves[b][g_a]:.4e}   (that suite's own best "
                  f"{min(curves[b].values()):.4e}, g=1 "
                  f"{curves[b][min(gs, key=lambda x: abs(x-1.0))]:.4e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
