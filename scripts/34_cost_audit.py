#!/usr/bin/env python
"""Charged-cost audit of the shipped estimator.

``adjusted = raw_mse x max(0.1, C/B)`` with ``C = F + 1e11 R``, so below
``C/B = 0.1`` the multiplier is clamped and every FLOP is free.  We sit above
it, so this script attributes both halves of ``C``:

``--mode ops``
    One ``predict`` under a real ``BudgetContext``, with the whole ``op_log``
    aggregated per operation name: dispatch count, charged FLOPs, resolved
    dtype, flopscope backend time and flopscope dispatch overhead.  Residual
    is ``wall - backend - overhead`` -- i.e. OUR Python -- so the table also
    prints what fraction of the wall clock is unattributed to any op.

``--mode resid``
    Repeat-measured residual for a set of variants.  Residual has 5-8%
    run-to-run spread on this box, so a single draw is not a measurement:
    every variant is run ``--repeats`` times, interleaved, and the median and
    the inter-quartile range are reported.

``--mode dtype``
    Every op whose ``resolved_dtype`` is not float32, with its FLOP cost, so
    the float64 promotions are visible individually rather than in aggregate.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import (  # noqa: E402
    DEPTH,
    FLOP_BUDGET,
    LAMBDA_FLOPS_PER_SECOND,
    MULTIPLIER_FLOOR,
    WIDTH,
    effective_compute,
)
from whestfloor.official_seeds import (  # noqa: E402
    derive_estimator_seed,
    make_official_mlp,
)
from whestfloor.suite import Suite  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def artifacts() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


def load_submission(path: Path):
    """Import ``submission/estimator.py`` standalone (no whestbench needed)."""
    if "whestbench" not in sys.modules:
        wb = types.ModuleType("whestbench")

        class BaseEstimator:  # noqa: D401
            pass

        wb.BaseEstimator = BaseEstimator
        sys.modules["whestbench"] = wb
    spec = importlib.util.spec_from_file_location(f"shipped_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _MLP:
    def __init__(self, w, seed):
        self.width = w[0].shape[0]
        self.depth = len(w)
        self.weights = w
        self.seed = seed


class _Ctx:
    width, depth = WIDTH, DEPTH
    flop_budget = FLOP_BUDGET
    api_version = "1.0"
    scratch_dir = None
    submission_dir = str(ROOT / "submission")
    seed = 0


def make_estimator(path: Path):
    mod = load_submission(path)
    est = mod.Estimator()
    est.setup(_Ctx())
    return est


def official_weights(suite: Suite, i: int):
    import flopscope.numpy as fnp  # noqa: PLC0415

    fw = [fnp.asarray(w)
          for w in make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])]
    return _MLP(fw, derive_estimator_seed(suite.mlp_seeds[i]))


# ---------------------------------------------------------------------------
def mode_ops(suite: Suite, path: Path, mlp_index: int) -> None:
    import flopscope as flops  # noqa: PLC0415

    est = make_estimator(path)
    mlp = official_weights(suite, mlp_index)
    est.predict(mlp, FLOP_BUDGET)          # warm every code path first

    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as ctx:
        est.predict(mlp, FLOP_BUDGET)

    log = ctx.op_log
    agg: dict[tuple[str, str], dict] = {}
    for op in log:
        key = (op.op_name, op.resolved_dtype or "?")
        b = agg.setdefault(key, {"n": 0, "f": 0, "back": 0.0, "over": 0.0})
        b["n"] += 1
        b["f"] += int(op.flop_cost)
        b["back"] += float(op.flopscope_backend_duration_s or 0.0)
        b["over"] += float(op.flopscope_overhead_duration_s or 0.0)

    wall = float(ctx.wall_time_s or 0.0)
    back = float(ctx.flopscope_backend_time_s)
    over = float(ctx.flopscope_overhead_time_s)
    resid = float(ctx.residual_wall_time_s or 0.0)
    F = int(ctx.flops_used)
    C = effective_compute(F, resid, LAMBDA_FLOPS_PER_SECOND)

    print(f"# {path}  mlp index {mlp_index}")
    print(f"# dispatches {len(log)}   F {F:,}  F/B {F/FLOP_BUDGET:.4f}")
    print(f"# wall {wall*1e3:8.2f} ms = backend {back*1e3:7.2f} + overhead "
          f"{over*1e3:7.2f} + RESIDUAL {resid*1e3:7.2f}")
    print(f"# C/B {C/FLOP_BUDGET:.4f}   (residual is {resid/max(wall,1e-9):.1%} "
          f"of wall, and 1e11*R/B = {LAMBDA_FLOPS_PER_SECOND*resid/FLOP_BUDGET:.4f})")
    print()
    hdr = (f"{'op':<28} {'dtype':<9} {'calls':>6} {'flops':>14} {'F/B':>7} "
           f"{'backend_ms':>11} {'over_ms':>9}")
    print(hdr)
    print("-" * len(hdr))
    for (name, dt), b in sorted(agg.items(), key=lambda kv: -kv[1]["f"]):
        print(f"{name:<28} {dt:<9} {b['n']:6d} {b['f']:14,d} "
              f"{b['f']/FLOP_BUDGET:7.4f} {b['back']*1e3:11.3f} "
              f"{b['over']*1e3:9.3f}")
    tn = sum(b["n"] for b in agg.values())
    print("-" * len(hdr))
    print(f"{'TOTAL':<28} {'':<9} {tn:6d} {F:14,d} {F/FLOP_BUDGET:7.4f} "
          f"{back*1e3:11.3f} {over*1e3:9.3f}")

    # ---- residual attribution -------------------------------------------
    # Each op carries its start offset from the context start and its own
    # backend+overhead duration, so the GAP before an op is Python that
    # flopscope did not account for -- i.e. billed residual.  Attribute each
    # gap to the op that follows it.
    gaps: dict[str, list] = {}
    end = 0.0
    for op in log:
        st = op.flopscope_context_start_offset_s
        if st is None:
            continue
        d = (float(op.flopscope_backend_duration_s or 0.0)
             + float(op.flopscope_overhead_duration_s or 0.0))
        gaps.setdefault(op.op_name, []).append(max(0.0, st - end))
        end = st + d
    print()
    print("# residual attributed to the gap BEFORE each op (ms)")
    print(f"{'before op':<32} {'calls':>6} {'total_ms':>9} {'mean_us':>9}")
    tot = 0.0
    for name, g in sorted(gaps.items(), key=lambda kv: -sum(kv[1])):
        s = sum(g)
        tot += s
        if s * 1e3 < 0.05:
            continue
        print(f"{name:<32} {len(g):6d} {s*1e3:9.3f} {s/len(g)*1e6:9.1f}")
    print(f"{'accounted':<32} {'':>6} {tot*1e3:9.3f}   of residual "
          f"{resid*1e3:.3f} ms (tail after the last op is the remainder)")


# ---------------------------------------------------------------------------
def mode_dtype(suite: Suite, path: Path, mlp_index: int) -> None:
    import flopscope as flops  # noqa: PLC0415

    est = make_estimator(path)
    mlp = official_weights(suite, mlp_index)
    est.predict(mlp, FLOP_BUDGET)
    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as ctx:
        est.predict(mlp, FLOP_BUDGET)

    bad = [op for op in ctx.op_log
           if (op.resolved_dtype or "float32") not in ("float32", "bool",
                                                       "int64", "int32",
                                                       "uint32", "uint64")]
    tot = sum(int(op.flop_cost) for op in bad)
    print(f"# ops whose resolved_dtype is not float32: {len(bad)} of "
          f"{len(ctx.op_log)}, {tot:,} FLOPs = {tot/FLOP_BUDGET:.4f} of B")
    agg: dict[tuple[str, str], list] = {}
    for op in bad:
        k = (op.op_name, op.resolved_dtype or "?")
        v = agg.setdefault(k, [0, 0])
        v[0] += 1
        v[1] += int(op.flop_cost)
    for (name, dt), (n, f) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:<28} {dt:<10} x{n:<5d} {f:14,d}  "
              f"{f/FLOP_BUDGET:.5f} of B")


# ---------------------------------------------------------------------------
def _measure(est, mlps, repeats: int):
    """Interleaved repeats; returns (F, [residual per repeat])."""
    import flopscope as flops  # noqa: PLC0415

    F = None
    out = []
    for _ in range(repeats):
        r = 0.0
        f = 0
        for mlp in mlps:
            with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as c:
                est.predict(mlp, FLOP_BUDGET)
            r += float(c.residual_wall_time_s or 0.0)
            f += int(c.flops_used)
        out.append(r / len(mlps))
        F = f / len(mlps)
    return F, out


def mode_resid(suite: Suite, paths: list[Path], repeats: int,
               n_mlps: int) -> None:
    mlps = [official_weights(suite, i) for i in range(n_mlps)]
    ests = [(p, make_estimator(p)) for p in paths]
    for _, e in ests:                      # warm
        e.predict(mlps[0], FLOP_BUDGET)

    res: dict[str, list] = {str(p): [] for p in paths}
    Fs: dict[str, float] = {}
    for _ in range(repeats):               # interleave to share load
        for p, e in ests:
            F, r = _measure(e, mlps, 1)
            res[str(p)].extend(r)
            Fs[str(p)] = F
    hdr = (f"{'variant':<44} {'F/B':>7} {'R_med_ms':>9} {'R_iqr':>7} "
           f"{'C/B':>7} {'C/B@2x':>7} {'C/B@3x':>7}")
    print(f"# {n_mlps} MLPs x {repeats} interleaved repeats")
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        rs = sorted(res[str(p)])
        med = statistics.median(rs)
        q1 = rs[len(rs) // 4]
        q3 = rs[(3 * len(rs)) // 4]
        F = Fs[str(p)]
        cb = [effective_compute(F, k * med, LAMBDA_FLOPS_PER_SECOND)
              / FLOP_BUDGET for k in (1, 2, 3)]
        print(f"{p.name:<44} {F/FLOP_BUDGET:7.4f} {med*1e3:9.3f} "
              f"{(q3-q1)/max(med,1e-12):7.1%} " + " ".join(f"{v:7.4f}"
                                                           for v in cb))


# ---------------------------------------------------------------------------
def mode_score(suite: Suite, paths: list[Path], n_mlps: int,
               repeats: int) -> None:
    """raw_mse + F/B + C/B + adjusted at 1x/2x/3x + raises, per variant."""
    import flopscope as flops  # noqa: PLC0415

    gt = suite.gt[:, -1, :]
    mlps = [official_weights(suite, i) for i in range(n_mlps)]
    ests = [(p, make_estimator(p)) for p in paths]
    for _, e in ests:
        e.predict(mlps[0], FLOP_BUDGET)

    raws: dict[str, float] = {}
    worsts: dict[str, tuple] = {}
    raises: dict[str, int] = {}
    Fs: dict[str, float] = {}
    Rs: dict[str, list] = {str(p): [] for p in paths}
    preds: dict[str, np.ndarray] = {}

    for rep in range(repeats):
        for p, e in ests:
            key = str(p)
            mses, fls, rss, nf = [], [], [], 0
            worst = (0.0, -1)
            allp = []
            for i, mlp in enumerate(mlps):
                try:
                    with flops.BudgetContext(flop_budget=FLOP_BUDGET,
                                             quiet=True) as c:
                        out = e.predict(mlp, FLOP_BUDGET)
                    pred = np.asarray(out, dtype=np.float64)
                    fl = int(c.flops_used)
                    rs = float(c.residual_wall_time_s or 0.0)
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! RAISED {p.name} mlp {i}: "
                          f"{type(exc).__name__}: {exc}")
                    nf += 1
                    pred = np.zeros((DEPTH, WIDTH))
                    fl, rs = FLOP_BUDGET, 0.0
                assert pred.shape == (DEPTH, WIDTH)
                assert np.isfinite(pred).all()
                allp.append(pred[-1])
                m = float(np.mean((pred[-1] - gt[i]) ** 2))
                if m > worst[0]:
                    worst = (m, i)
                mses.append(m)
                fls.append(fl)
                rss.append(rs)
            raws[key] = float(np.mean(mses))
            worsts[key] = worst
            raises[key] = nf
            Fs[key] = float(np.mean(fls))
            Rs[key].append(float(np.mean(rss)))
            if rep == 0:
                preds[key] = np.asarray(allp)

    hdr = (f"{'variant':<40} {'raw_mse':>11} {'F/B':>7} {'C/B':>7} "
           f"{'adj@1x':>10} {'adj@2x':>10} {'adj@3x':>10} {'raise':>6}")
    print(f"# official suite, {n_mlps} MLPs, N=1e9 reference; "
          f"residual = median of {repeats} repeats")
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        key = str(p)
        R = statistics.median(Rs[key])
        F, raw = Fs[key], raws[key]
        adj = [raw * max(MULTIPLIER_FLOOR,
                         effective_compute(F, k * R, LAMBDA_FLOPS_PER_SECOND)
                         / FLOP_BUDGET) for k in (1, 2, 3)]
        C1 = effective_compute(F, R, LAMBDA_FLOPS_PER_SECOND)
        print(f"{p.name:<40} {raw:11.4e} {F/FLOP_BUDGET:7.4f} "
              f"{C1/FLOP_BUDGET:7.4f} " + " ".join(f"{v:10.4e}" for v in adj)
              + f" {raises[key]:6d}")
        print(f"{'':<40} worst MLP {worsts[key][0]:.4e} "
              f"(index {worsts[key][1]})   R {R*1e3:.2f} ms")
    if len(paths) > 1:
        a = preds[str(paths[0])]
        for p in paths[1:]:
            b = preds[str(p)]
            same = np.array_equal(a, b)
            print(f"# bitwise identical to {paths[0].name}: {p.name} -> "
                  f"{same}" + ("" if same else
                               f"  (max |d| {np.max(np.abs(a-b)):.3e})"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("ops", "dtype", "resid", "score"))
    ap.add_argument("--suite", default=None)
    ap.add_argument("--paths", default="submission/estimator.py",
                    help="comma-separated estimator files")
    ap.add_argument("--n-mlps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--mlp-index", type=int, default=0)
    args = ap.parse_args()

    p = (Path(args.suite) if args.suite
         else artifacts() / "suites" / "official_mini.npz")
    suite = Suite.load(p)
    paths = [Path(x) if os.path.isabs(x) else ROOT / x
             for x in args.paths.split(",") if x]

    if args.mode == "ops":
        mode_ops(suite, paths[0], args.mlp_index)
    elif args.mode == "dtype":
        mode_dtype(suite, paths[0], args.mlp_index)
    elif args.mode == "resid":
        mode_resid(suite, paths, args.repeats, args.n_mlps)
    else:
        mode_score(suite, paths, args.n_mlps, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
