#!/usr/bin/env python
"""The lattice inside the SHIPPED estimator: cost, redundancy, score, and N.

``scripts/50_rqmc_rate.py`` measures the sampler in isolation.  This one runs
the lattice through ``kernels.corrected_sparse_kernel`` — the exact shipped
code path, with the layer-1 Hermite control variates and the offline head —
because that is where the number that ranks us comes from, and because the
control variates are the thing most likely to make the lattice worthless.

``--mode cost``
    The A8 trap, checked rather than assumed.  ``flopscope.stats.norm.ppf``
    promotes float32 to float64 and a promoted array reprices the entire
    downstream chain at 2x; this asserts the 32 scored matmuls still bill at
    the float32 rate in a real ``BudgetContext``, and prices the draw.

``--mode redundancy``
    The 2x2 that decides whether the lattice ships: {iid, lattice} x {head off,
    head on}, same MLPs, same randomisations, paired.  radiant-allomancer
    (18085 §2.1) reports the two levers are partly redundant — covariance
    shrinkage already removes most of the variance RQMC targets — and that is
    what turned 5-7x into 1.40x for them.  Our k=1 Hermite block is the
    *optimal input-linear* control variate and a rank-1 lattice annihilates
    exactly the first-order ANOVA terms, so the overlap is not a risk here, it
    is a theorem; the only question is how much is left at order >= 2.

``--mode score``
    End-to-end on the official 100-MLP suite against the 1e9 reference, so
    ``raw_mse`` is leaderboard-comparable, with the billed ``C/B``.

``--mode nsweep``
    ``N`` re-optimised from scratch.  With ``raw = v/N^p`` and ``p > 1`` the
    old argument inverts: ``adjusted = raw * max(0.1, C/B)`` keeps FALLING
    with ``N`` above the clamp instead of being flat in it.

Credit: evaaaz (18053) for the construction, radiant-allomancer (18085) for
the redundancy warning and the antithetic refutation, jamesrahenry (18097
erratum) and mohanty (18125) for the float64 inverse-CDF trap.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import rqmc as RQ  # noqa: E402
from whestfloor.official_seeds import make_official_mlp  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402

WIDTH, DEPTH = 256, 32
DENSE_FLOPS = 4_198_656.0


def art() -> Path:
    p = Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve() / "rqmc"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_suite(path: str | None) -> Suite:
    p = (Path(path) if path else
         Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()
         / "suites" / "official_mini.npz")
    return Suite.load(p)


def load_beta():
    f = Path(__file__).resolve().parent.parent / "submission" / "corrector.npz"
    if not f.is_file():
        return None
    d = np.load(f)
    return np.asarray(d["beta"], dtype=np.float32)


# ---------------------------------------------------------------------------
# mode: cost -- the float32 audit and the FLOP bill of the draw
# ---------------------------------------------------------------------------
def mode_cost(ns, tau, n_pilot):
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels as K
    from whestfloor.contract import FLOP_BUDGET

    beta = load_beta()
    W = [fnp.asarray(w) for w in make_official_mlp(WIDTH, DEPTH, 700_001)]
    print(f"{'N':>8} {'iid F':>16} {'lattice F':>16} {'delta':>14} "
          f"{'/elem':>8} {'x iid':>7} {'F/B lat':>8}")
    rows = []
    for n in ns:
        z = RQ.get_z(n, WIDTH, "cbc")
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c0:
            base = RQ.billed_lattice_base(n, z)
        f_base = int(c0.flops_used)
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c1:
            K.corrected_sparse_kernel(W, tau=tau, n_samples=n,
                                      n_pilot=n_pilot, beta=beta, safe=False)
        f_iid = int(c1.flops_used)
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c2:
            base2 = RQ.billed_lattice_base(n, z)   # setup builds it: free
            K.corrected_sparse_kernel(W, tau=tau, n_samples=n,
                                      n_pilot=n_pilot, beta=beta, safe=False,
                                      x0_fn=RQ.lattice_x0_fn(base2))
        f_lat = int(c2.flops_used) - f_base        # base is a setup cost
        d = f_lat - f_iid
        print(f"{n:8d} {f_iid:16,} {f_lat:16,} {d:14,} "
              f"{d / (n * WIDTH):8.1f} {f_lat / f_iid:7.4f} "
              f"{f_lat / FLOP_BUDGET:8.4f}")
        rows.append(dict(n=n, f_iid=f_iid, f_lat=f_lat, f_base=f_base,
                         per_elem=d / (n * WIDTH)))
        _ = base

    print("\n# float32 audit: does the promoted ppf reprice the scored pass?")
    n = ns[-1]
    z = RQ.get_z(n, WIDTH, "cbc")
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
        base = RQ.billed_lattice_base(n, z)
        rng = fnp.random.default_rng(0)
        x = RQ.billed_lattice_normals(base, rng)
    print(f"  draw dtype                    : {x.dtype}")
    assert str(x.dtype) == "float32", "the ppf promotion survived the cast"
    # the marginal FLOPs per sample must match the iid kernel's exactly once
    # the draw is subtracted; a float64 chain would double them
    lo, hi = ns[0], ns[-1]
    per_iid = (rows[-1]["f_iid"] - rows[0]["f_iid"]) / (hi - lo)
    per_lat = (rows[-1]["f_lat"] - rows[0]["f_lat"]) / (hi - lo)
    print(f"  marginal FLOPs/sample, iid    : {per_iid:,.0f}")
    print(f"  marginal FLOPs/sample, lattice: {per_lat:,.0f}")
    print(f"  ratio                         : {per_lat / per_iid:.4f}"
          "   (2.00 would mean the chain got repriced at float64)")
    print(f"  draw overhead                 : "
          f"{per_lat - per_iid:,.0f} FLOPs/sample = "
          f"{100 * (per_lat / per_iid - 1):.2f}% of the pass")
    (art() / "cost.json").write_text(json.dumps(
        dict(rows=rows, per_iid=per_iid, per_lat=per_lat), indent=1))
    return rows


# ---------------------------------------------------------------------------
# mode: redundancy -- lattice x control variates, 2x2
# ---------------------------------------------------------------------------
def mode_redundancy(suite, n_mlps, n, reps, tau, n_pilot, seed):
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels as K

    beta = load_beta()
    z = RQ.get_z(n, WIDTH, "cbc")
    gt = suite.gt[:, -1, :]
    arms = {
        "iid,      head off": dict(x0=False, damp=0.0),
        "iid,      head ON ": dict(x0=False, damp=1.0),
        "lattice,  head off": dict(x0=True, damp=0.0),
        "lattice,  head ON ": dict(x0=True, damp=1.0),
    }
    P = {k: np.zeros((n_mlps, reps, WIDTH)) for k in arms}
    t0 = time.time()
    for mi in range(n_mlps):
        W = [fnp.asarray(w) for w in make_official_mlp(WIDTH, DEPTH,
                                                       suite.mlp_seeds[mi])]
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
            base = RQ.billed_lattice_base(n, z)
        for r in range(reps):
            sd = 20_000_003 * (seed + 1) + 7919 * r + 104_729 * mi
            for name, cfg in arms.items():
                with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
                    out = K.corrected_sparse_kernel(
                        W, tau=tau, n_samples=n, n_pilot=n_pilot, seed=sd,
                        beta=beta, damp=cfg["damp"], safe=False,
                        x0_fn=RQ.lattice_x0_fn(base) if cfg["x0"] else None)
                P[name][mi, r] = np.asarray(out, dtype=np.float64)[-1]
        print(f"  mlp {mi + 1}/{n_mlps} [{time.time() - t0:.0f}s]", flush=True)

    print(f"\n=== lattice x control variates, N={n}, tau={tau}, "
          f"P={n_pilot}, {n_mlps} MLPs x {reps} randomisations ===")
    hdr = (f"{'arm':<20} {'mse':>11} {'variance':>11} {'bias^2':>11} "
           f"{'x over iid/off':>15} {'+- se':>7}")
    print(hdr)
    print("-" * len(hdr))
    ref = None
    res = {}
    for name in arms:
        A = P[name]
        mse = np.array([np.mean((A[mi] - gt[mi]) ** 2) for mi in range(n_mlps)])
        var = np.array([A[mi].var(0, ddof=1).mean() for mi in range(n_mlps)])
        b2 = np.array([np.mean((A[mi].mean(0) - gt[mi]) ** 2)
                       - A[mi].var(0, ddof=1).mean() / reps
                       for mi in range(n_mlps)])
        if ref is None:
            ref = mse
        g = ref.mean() / mse.mean()
        rng = np.random.default_rng(0)
        bs = [ref[i].mean() / mse[i].mean()
              for i in rng.integers(0, n_mlps, (2000, n_mlps))]
        res[name] = dict(mse=float(mse.mean()), var=float(var.mean()),
                         b2=float(b2.mean()), gain=float(g),
                         se=float(np.std(bs)))
        print(f"{name:<20} {mse.mean():11.4e} {var.mean():11.4e} "
              f"{b2.mean():11.4e} {g:15.3f} {np.std(bs):7.3f}")

    a = res["iid,      head off"]["mse"]
    b = res["iid,      head ON "]["mse"]
    c = res["lattice,  head off"]["mse"]
    d = res["lattice,  head ON "]["mse"]
    print(f"\n  lattice alone         : {a / c:6.3f}x")
    print(f"  head alone            : {a / b:6.3f}x")
    print(f"  both                  : {a / d:6.3f}x")
    print(f"  product if independent: {(a / b) * (a / c):6.3f}x")
    print(f"  REDUNDANCY            : "
          f"{(a / d) / ((a / b) * (a / c)):6.3f}   "
          "(1.00 = no overlap, <1 = the two levers overlap)")
    print(f"  head on top of lattice: {c / d:6.3f}x   "
          f"(vs {a / b:.3f}x on top of iid)")
    res["_summary"] = dict(lattice=a / c, head=a / b, both=a / d,
                           redundancy=(a / d) / ((a / b) * (a / c)))
    (art() / f"redundancy_n{n}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(art() / f"redundancy_n{n}.npz",
                        **{k.replace(" ", "").replace(",", "_"): v
                           for k, v in P.items()})
    return res


# ---------------------------------------------------------------------------
# mode: score / nsweep -- the official suite, billed
# ---------------------------------------------------------------------------
def _variants(grid, tau, n_pilot, beta):
    import flopscope as flops

    from whestfloor import kernels as K

    out = []
    for spec in grid.split(";"):
        if not spec:
            continue
        kw = dict(kv.split("=") for kv in spec.split(","))
        n = int(kw.get("n", 25000))
        draw = kw.get("draw", "lat")
        damp = float(kw.get("damp", 1.0))
        sd = int(kw.get("seed", 0))
        P = int(kw.get("P", n_pilot))
        tt = float(kw.get("tau", tau))
        x0 = None
        if draw == "lat":
            z = RQ.get_z(n, WIDTH, kw.get("z", "cbc"))
            with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
                base = RQ.billed_lattice_base(n, z)
            x0 = RQ.lattice_x0_fn(base)
        name = (f"{draw:<4} n={n:<6} P={P:<4} tau={tt:<4} damp={damp:<3} "
                f"s={sd}")
        out.append((name, functools.partial(
            K.corrected_sparse_kernel, tau=tt, n_samples=n, n_pilot=P,
            seed=sd, beta=beta, damp=damp, x0_fn=x0)))
    return out


def mode_score(suite, n_mlps, grid, tau, n_pilot, seed):
    from whestfloor.contract import (
        FLOP_BUDGET, LAMBDA_FLOPS_PER_SECOND, MULTIPLIER_FLOOR,
        effective_compute,
    )
    from whestfloor.harness import run_billed

    beta = load_beta()
    gt = suite.gt[:, -1, :]
    print(f"# official suite, {n_mlps} MLPs, 1e9 reference -> raw_mse is "
          "leaderboard-comparable")
    hdr = (f"{'variant':<44} {'raw_mse':>11} {'F/B':>7} {'C/B':>7} "
           f"{'adj@1x':>11} {'adj@2x':>11} {'raise':>5} {'worst':>10}")
    print(hdr)
    print("-" * len(hdr))
    # Written after EVERY variant, not at the end: a 100-MLP pass is ~15 minutes
    # and this box has restarted mid-run once, taking a finished variant with it.
    outp = art() / "score.json"
    res = json.loads(outp.read_text()) if outp.is_file() else {}
    for name, fn in _variants(grid, tau, n_pilot, beta):
        if name in res:
            v = res[name]
            print(f"{name:<44} {v['raw']:11.4e} {v['F'] / FLOP_BUDGET:7.4f} "
                  f"{v['CB']:7.4f} {v['adj']:11.4e} {v['adj2']:11.4e} "
                  f"{v['nfail']:5d} {v['worst']:10.3e}   (cached)")
            continue
        mses, fls, rss, nfail, worst = [], [], [], 0, 0.0
        for i in range(n_mlps):
            Wn = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])
            try:
                pred, fl, rs = run_billed(fn, Wn)
                assert pred.shape == (DEPTH, WIDTH) and np.isfinite(pred).all()
            except Exception as e:  # noqa: BLE001
                print(f"  !! RAISED on mlp {i}: {type(e).__name__}: {e}")
                nfail += 1
                pred, fl, rs = np.zeros((DEPTH, WIDTH)), FLOP_BUDGET, 0.0
            m = float(np.mean((pred[-1] - gt[i]) ** 2))
            worst = max(worst, m)
            mses.append(m)
            fls.append(fl)
            rss.append(rs)
        raw = float(np.mean(mses))
        F, Rr = float(np.mean(fls)), float(np.mean(rss))
        adj = [raw * max(MULTIPLIER_FLOOR,
                         effective_compute(F, k * Rr, LAMBDA_FLOPS_PER_SECOND)
                         / FLOP_BUDGET) for k in (1, 2)]
        C1 = effective_compute(F, Rr, LAMBDA_FLOPS_PER_SECOND)
        print(f"{name:<44} {raw:11.4e} {F / FLOP_BUDGET:7.4f} "
              f"{C1 / FLOP_BUDGET:7.4f} {adj[0]:11.4e} {adj[1]:11.4e} "
              f"{nfail:5d} {worst:10.3e}", flush=True)
        res[name] = dict(raw=raw, F=F, R=Rr, CB=C1 / FLOP_BUDGET,
                         adj=adj[0], adj2=adj[1], nfail=nfail, worst=worst)
        outp.write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("cost", "redundancy", "score"))
    ap.add_argument("--suite", default=None)
    ap.add_argument("--n-mlps", type=int, default=8)
    ap.add_argument("--N", type=int, default=32749)
    ap.add_argument("--ns", default="8191,16381,32749")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--n-pilot", type=int, default=225)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", default="draw=iid,n=25000;draw=lat,n=32749")
    args = ap.parse_args()
    if args.mode == "cost":
        mode_cost([int(v) for v in args.ns.split(",")], args.tau,
                  args.n_pilot)
        return 0
    suite = load_suite(args.suite)
    if args.mode == "redundancy":
        mode_redundancy(suite, args.n_mlps, args.N, args.reps, args.tau,
                        args.n_pilot, args.seed)
    elif args.mode == "score":
        mode_score(suite, args.n_mlps, args.grid, args.tau, args.n_pilot,
                   args.seed)
    _ = math
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
