#!/usr/bin/env python
"""Sign-stable sparse Monte Carlo: measure the premise, price it, score it.

Four modes, in the order the decision was actually taken.

``--mode alpha``
    The premise.  Per-layer distribution of ``alpha = m/s`` on real official
    MLPs, the expected number of sign flips per sample, the kink set
    ``{|alpha| < tau}`` that must be EVALUATED to find those flips, and the
    dead set ``{alpha < -tau}`` that can be dropped.  This is the measurement
    that decides which sparse scheme is even worth writing.

``--mode price``
    The cost, billed rather than modelled.  Per-sample marginal FLOPs of the
    dense pass, of dead-neuron pruning at each threshold, and of the
    modal-fusion scheme (fuse the modal sign pattern into one matrix A and
    correct only the kink neurons), obtained by differencing two batch sizes
    inside a real ``flops.BudgetContext``.

``--mode bias``
    The accuracy cost, measured PAIRED: the pruned pass and the dense pass are
    run on the identical sample stream, so the difference is the sign error
    alone with the Monte-Carlo noise cancelled.  An unpaired comparison at
    this suite size is useless -- the seed-to-seed spread of the raw MSE is
    ~50% on 4 MLPs, which is larger than every effect being measured.

``--mode score``
    End-to-end on the official 100-MLP suite (N=1e9 reference, so ``raw_mse``
    is leaderboard-comparable), with ``tau=None`` reverting to dense through
    the identical code path as the ablation.  Reports ``F/B`` separately from
    ``C/B``: FLOPs are machine-independent, residual wall time is not, and the
    grader runs participant code on one physical core.  The adjusted score is
    also recomputed at 2x and 3x this machine's residual.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import kernels  # noqa: E402
from whestfloor.contract import (  # noqa: E402
    FLOP_BUDGET,
    LAMBDA_FLOPS_PER_SECOND,
    MULTIPLIER_FLOOR,
    effective_compute,
)
from whestfloor.official_seeds import make_official_mlp  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402

WIDTH, DEPTH = 256, 32
TAUS = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)


def artifacts() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


def load_suite(path: str | None) -> Suite:
    p = Path(path) if path else artifacts() / "suites" / "official_mini.npz"
    return Suite.load(p)


# ---------------------------------------------------------------------------
# mode: alpha
# ---------------------------------------------------------------------------
def mode_alpha(suite: Suite, n_mlps: int, n_samples: int) -> None:
    chunk = 4000
    n, L = WIDTH, DEPTH
    A, F = [], []
    for i in range(n_mlps):
        W = make_official_mlp(n, L, suite.mlp_seeds[i])
        s1 = np.zeros((L, n))
        s2 = np.zeros((L, n))
        rng = np.random.default_rng(12345)
        for _ in range(n_samples // chunk):
            x = rng.standard_normal((chunk, n), dtype=np.float32)
            for l, w in enumerate(W):
                z = x @ w
                s1[l] += z.sum(0, dtype=np.float64)
                s2[l] += (z.astype(np.float64) ** 2).sum(0)
                x = np.maximum(z, 0.0)
        N = (n_samples // chunk) * chunk
        m = s1 / N
        s = np.sqrt(np.maximum(s2 / N - m * m, 1e-30))
        d = m > 0
        # second, independent stream: empirical flip rate against the mode
        nf = np.zeros((L, n))
        rng = np.random.default_rng(54321)
        for _ in range(n_samples // chunk):
            x = rng.standard_normal((chunk, n), dtype=np.float32)
            for l, w in enumerate(W):
                z = x @ w
                nf[l] += ((z > 0) != d[l]).sum(0)
                x = np.maximum(z, 0.0)
        A.append(m / s)
        F.append(nf / N)
        print(f"  mlp {i + 1}/{n_mlps}", flush=True)
    alpha = np.stack(A)
    pflip = np.stack(F)
    aa = np.abs(alpha)

    print(f"\n=== |alpha| by layer, {n_mlps} official MLPs x {n_samples:,} "
          "samples ===")
    print(f"{'l':>3} {'rms|a|':>8} {'med|a|':>8} "
          + " ".join(("|a|<%g" % t).rjust(8) for t in TAUS) + f" {'E[flips]':>9}")
    for l in range(L):
        row = aa[:, l, :].ravel()
        print(f"{l + 1:3d} {np.sqrt((row ** 2).mean()):8.3f} "
              f"{np.median(row):8.3f} "
              + " ".join(f"{100 * np.mean(row < t):8.1f}" for t in TAUS)
              + f" {pflip[:, l, :].sum(1).mean():9.2f}")

    tot = pflip.sum(2).sum(1)
    print(f"\nE[sign flips / sample] = {tot.mean():.1f} +- {tot.std():.1f} of "
          f"{L * n} neurons ({100 * tot.mean() / (L * n):.2f}%)")
    print("\n=== what a flip-detecting sparse pass must actually evaluate ===")
    print(f"{'tau':>5} {'kink%':>7} {'dead%':>7} {'flips kept%':>12} "
          f"{'missed/sample':>14} {'kink/flip':>10}")
    for t in TAUS:
        kink = aa < t
        cap = (pflip * kink).sum() / pflip.sum()
        print(f"{t:5.1f} {100 * kink.mean():7.1f} "
              f"{100 * np.mean(alpha < -t):7.1f} {100 * cap:12.2f} "
              f"{(pflip * ~kink).sum(2).sum(1).mean():14.3f} "
              f"{kink.sum() / (pflip.sum() * n_mlps) * n_mlps:10.1f}")
    out = artifacts() / "sparse" / "alpha.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, alpha=alpha, pflip=pflip)
    print(f"\nsaved {out}")


# ---------------------------------------------------------------------------
# mode: price
# ---------------------------------------------------------------------------
def _modal_fusion_kernel(weights, ctx=None, tau=2.0, n_samples=600, seed=0,
                         n_pilot=150):
    """The scheme this work set out to build, kept only so it can be priced.

    ``z^32 = x A + sum_l eps^l R^l`` with ``A`` the fused modal chain and
    ``eps`` the ReLU deviation restricted to the kink neurons.  Exact, and
    catastrophically expensive: the kink-to-kink coupling is O(depth^2) in
    sets that are half the width.
    """
    import flopscope.numpy as fnp  # noqa: PLC0415

    n, L = weights[0].shape[0], len(weights)
    rng = fnp.random.default_rng(seed)
    x = rng.standard_normal((n_pilot, n), dtype=fnp.float32)
    ms, ss = [], []
    for w in weights:
        z = x @ w
        m = fnp.mean(z, axis=0)
        ms.append(m)
        ss.append(fnp.sqrt(fnp.maximum(fnp.mean(z * z, axis=0) - m * m, 1e-12)))
        x = fnp.maximum(z, 0.0)
    d = [(ms[l] > 0.0).astype(fnp.float32) for l in range(L)]
    a = [ms[l] / ss[l] for l in range(L)]
    kink = [fnp.maximum(a[l], -a[l]) < float(tau) for l in range(L)]

    A = weights[0]
    for l in range(1, L):
        A = (A * d[l - 1]) @ weights[l]
    R = [None] * L
    cur = weights[L - 1]
    R[L - 2] = cur[kink[L - 2], :]
    for l in range(L - 3, -1, -1):
        cur = (weights[l + 1] * d[l + 1]) @ cur
        R[l] = cur[kink[l], :]
    C = [None] * L
    cur = weights[0]
    C[0] = cur[:, kink[0]]
    for l in range(1, L - 1):
        cur = (cur * d[l - 1]) @ weights[l]
        C[l] = cur[:, kink[l]]
    G = {}
    for lp in range(L - 1):
        cur = weights[lp + 1][kink[lp], :]
        for l in range(lp + 1, L - 1):
            G[(lp, l)] = cur[:, kink[l]]
            cur = (cur * d[l]) @ weights[l + 1]
    dk = [d[l][kink[l]] for l in range(L - 1)]

    x = rng.standard_normal((n_samples, n), dtype=fnp.float32)
    z32 = x @ A
    eps = {}
    for l in range(L - 1):
        zk = x @ C[l]
        for lp in range(l):
            zk = zk + eps[lp] @ G[(lp, l)]
        e = fnp.maximum(zk, 0.0) - dk[l] * zk
        eps[l] = e
        z32 = z32 + e @ R[l]
    return fnp.mean(fnp.maximum(z32, 0.0), axis=0)


def _bill(fn, W, **kw):
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415
    fw = [fnp.asarray(w) for w in W]
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as ctx:
        fn(fw, **kw)
    return int(ctx.flops_used)


def _split(fn, W, b1=400, b2=1600, **kw):
    c1 = _bill(fn, W, n_samples=b1, **kw)
    c2 = _bill(fn, W, n_samples=b2, **kw)
    per = (c2 - c1) / (b2 - b1)
    return per, c1 - per * b1


def mode_price(suite: Suite) -> None:
    W = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[0])
    free = MULTIPLIER_FLOOR * FLOP_BUDGET
    print("per-sample marginal cost, billed in a real BudgetContext "
          "(two batch sizes, differenced)\n")
    print(f"{'scheme':<44} {'flops/sample':>13} {'x cheaper':>10} "
          f"{'setup':>13} {'N @ free':>10}")
    dense, _ = _split(kernels._dense_rows, W, seed=0)
    print(f"{'dense MC (all layers)':<44} {dense:13,.0f} {1.0:10.2f} "
          f"{0:13,} {int(free / dense):10,}")
    for tau in (None, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5, 0.0):
        per, st = _split(kernels.sparse_mc_kernel, W, tau=tau, n_pilot=150,
                         seed=0)
        lbl = f"dead-prune tau={tau}" + (" [= dense, ablation]" if tau is None
                                         else "")
        print(f"{lbl:<44} {per:13,.0f} {dense / per:10.2f} {st:13,.0f} "
              f"{int(max(free - st, 0) / per):10,}")
    for tau in (2.0, 3.0):
        per, st = _split(_modal_fusion_kernel, W, tau=tau, b1=200, b2=800)
        print(f"{'modal fusion + kink correction tau=%g' % tau:<44} "
              f"{per:13,.0f} {dense / per:10.2f} {st:13,.0f} "
              f"{int(max(free - st, 0) / per):10,}")
    print(f"\n(free budget = 0.1 * B = {free:,.0f}; setup for the pruned path "
          "is the pilot pass plus the weight slicing)")


# ---------------------------------------------------------------------------
# mode: bias
# ---------------------------------------------------------------------------
def mode_bias(suite: Suite, n_mlps: int, n_samples: int) -> None:
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    gt = suite.gt[:, -1, :]
    taus = (4.0, 3.0, 2.5, 2.0, 1.0)
    pilots = (150, 20000)
    # One generator feeds the pilot and then the scored draw, so the scored
    # STREAM depends on n_pilot.  Pairing therefore only holds at fixed P:
    # every column below is differenced against its OWN tau=None run at the
    # same P, never against a single shared baseline.  Getting this wrong
    # injects ~8e-7 of stream noise into a ~1e-7 effect.
    acc = {(t, p): [] for t in taus for p in pilots}
    base = {p: [] for p in pilots}
    for i in range(n_mlps):
        fw = [fnp.asarray(w)
              for w in make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])]
        for p in pilots:
            with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
                base[p].append(np.asarray(kernels.sparse_mc_kernel(
                    fw, tau=None, n_samples=n_samples, n_pilot=p, seed=0),
                    dtype=np.float64)[-1])
            for t in taus:
                with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
                    acc[(t, p)].append(np.asarray(kernels.sparse_mc_kernel(
                        fw, tau=t, n_samples=n_samples, n_pilot=p, seed=0),
                        dtype=np.float64)[-1])
        print(f"  mlp {i + 1}/{n_mlps}", flush=True)
    g = gt[:n_mlps]
    b_mse = {p: float(np.mean((np.array(base[p]) - g) ** 2)) for p in pilots}
    print(f"\n=== sign error, PAIRED against the dense pass on the identical "
          f"stream ({n_mlps} MLPs, N={n_samples}) ===")
    for p in pilots:
        print(f"  dense (tau=None) raw MSE at P={p}: {b_mse[p]:.4e}")
    print()
    print(f"{'tau':>5} " + " ".join(f"dMSE P={p}".rjust(14) for p in pilots))
    for t in taus:
        row = [float(np.mean((np.array(acc[(t, p)]) - g) ** 2)) - b_mse[p]
               for p in pilots]
        print(f"{t:5.1f} " + " ".join(f"{v:14.4e}" for v in row))
    print("\nP=20000 is an oracle-statistics control, not an affordable "
          "setting: it separates the frozen-constant closure error (which "
          "survives as P grows) from the pilot's own sampling noise (which "
          "enters the frozen constants as a fixed offset and so does NOT "
          "average away over the scored samples).")


# ---------------------------------------------------------------------------
# mode: score
# ---------------------------------------------------------------------------
def mode_score(suite: Suite, n_mlps: int, grid: str, seed: int = 0) -> None:
    from whestfloor.harness import run_billed  # noqa: PLC0415

    gt = suite.gt[:, -1, :]
    variants = []
    for spec in grid.split(";"):
        if not spec:
            continue
        kw = dict(kv.split("=") for kv in spec.split(","))
        tau = None if kw.get("tau", "None") == "None" else float(kw["tau"])
        ns = int(kw.get("n", 8500))
        npi = int(kw.get("P", 150))
        sd = int(kw.get("seed", seed))
        variants.append((
            f"tau={kw.get('tau','None'):>4} n={ns:<6} P={npi:<4} s={sd}",
            functools.partial(kernels.sparse_mc_kernel, tau=tau,
                              n_samples=ns, n_pilot=npi, seed=sd)))

    print(f"# official suite, {n_mlps} MLPs, N=1e9 reference -> raw_mse is "
          "leaderboard-comparable")
    print("# adj@kx = adjusted score if the grader's residual wall time is k "
          "times this machine's\n")
    hdr = (f"{'variant':<34} {'raw_mse':>11} {'F/B':>7} {'C/B':>7} "
           f"{'adj@1x':>10} {'adj@2x':>10} {'adj@3x':>10} {'raise':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, fn in variants:
        mses, fls, rss, nfail = [], [], [], 0
        for i in range(n_mlps):
            Wn = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])
            try:
                pred, fl, rs = run_billed(fn, Wn)
            except Exception as e:  # noqa: BLE001
                print(f"  !! RAISED on mlp {i}: {type(e).__name__}: {e}")
                nfail += 1
                pred, fl, rs = np.zeros((DEPTH, WIDTH)), FLOP_BUDGET, 0.0
            mses.append(float(np.mean((pred[-1] - gt[i]) ** 2)))
            fls.append(fl)
            rss.append(rs)
        raw = float(np.mean(mses))
        F = float(np.mean(fls))
        R = float(np.mean(rss))
        adj = []
        for k in (1, 2, 3):
            C = effective_compute(F, k * R, LAMBDA_FLOPS_PER_SECOND)
            adj.append(raw * max(MULTIPLIER_FLOOR, C / FLOP_BUDGET))
        C1 = effective_compute(F, R, LAMBDA_FLOPS_PER_SECOND)
        print(f"{name:<34} {raw:11.4e} {F / FLOP_BUDGET:7.4f} "
              f"{C1 / FLOP_BUDGET:7.4f} " + " ".join(f"{v:10.4e}" for v in adj)
              + f" {nfail:6d}")


# ---------------------------------------------------------------------------
# mode: ship  -- run the ACTUAL submission file, not a research copy
# ---------------------------------------------------------------------------
def mode_ship(suite: Suite, n_mlps: int) -> None:
    import importlib.util  # noqa: PLC0415
    import types  # noqa: PLC0415

    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    if "whestbench" not in sys.modules:
        wb = types.ModuleType("whestbench")

        class BaseEstimator:
            pass

        wb.BaseEstimator = BaseEstimator
        sys.modules["whestbench"] = wb
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "shipped", root / "submission" / "estimator.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)

    class _MLP:
        def __init__(self, w, seed):
            self.width = w[0].shape[0]
            self.depth = len(w)
            self.weights = w
            self.seed = seed

    class _Ctx:
        seed = 0

    est = sub.Estimator()
    est.setup(_Ctx())
    gt = suite.gt[:, -1, :]
    mses, fls, rss, worst = [], [], [], (0.0, -1)
    nfail = 0
    for i in range(n_mlps):
        fw = [fnp.asarray(w)
              for w in make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])]
        # the grader hands the estimator the protocol-3.0 derived seed
        from whestfloor.official_seeds import derive_estimator_seed  # noqa: PLC0415
        mlp = _MLP(fw, derive_estimator_seed(suite.mlp_seeds[i]))
        try:
            with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as c:
                out = est.predict(mlp, FLOP_BUDGET)
            pred = np.asarray(out, dtype=np.float64)
            fl, rs = int(c.flops_used), float(c.residual_wall_time_s)
        except Exception as e:  # noqa: BLE001
            print(f"  !! RAISED on mlp {i}: {type(e).__name__}: {e}")
            nfail += 1
            pred, fl, rs = np.zeros((DEPTH, WIDTH)), FLOP_BUDGET, 0.0
        assert pred.shape == (DEPTH, WIDTH) and np.isfinite(pred).all()
        mse = float(np.mean((pred[-1] - gt[i]) ** 2))
        if mse > worst[0]:
            worst = (mse, i)
        mses.append(mse)
        fls.append(fl)
        rss.append(rs)
    raw = float(np.mean(mses))
    F, R = float(np.mean(fls)), float(np.mean(rss))
    print(f"\n=== SHIPPED submission/estimator.py, {n_mlps} official MLPs "
          "(N=1e9 reference) ===")
    print(f"  raw final-layer MSE   {raw:.4e}")
    print(f"  F/B                   {F / FLOP_BUDGET:.4f}   "
          f"(machine-independent)")
    for k in (1, 2, 3):
        C = effective_compute(F, k * R, LAMBDA_FLOPS_PER_SECOND)
        print(f"  adjusted @ {k}x residual {raw * max(MULTIPLIER_FLOOR, C / FLOP_BUDGET):.4e}"
              f"   (C/B {C / FLOP_BUDGET:.4f})")
    print(f"  raises                {nfail} / {n_mlps}")
    print(f"  worst single MLP      {worst[0]:.4e} (index {worst[1]})")
    print(f"  max C over the suite  "
          f"{max(effective_compute(f, r, LAMBDA_FLOPS_PER_SECOND) for f, r in zip(fls, rss)) / FLOP_BUDGET:.4f} of B")

    # The defensive path must actually fire and must produce a usable answer.
    # Break only the sparse-specific stage: `_dense` shares `mlp.width` and
    # `mlp.weights` with it, so a fault in those is not recoverable by any
    # fallback and is left to the grader's own zeroing.
    good = _sparse_of = est._sparse
    est._sparse = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe"))
    try:
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as c:
            out = np.asarray(est.predict(mlp, FLOP_BUDGET), dtype=np.float64)
    finally:
        est._sparse = good
    fb_mse = float(np.mean((out[-1] - gt[n_mlps - 1]) ** 2))
    print(f"  fallback fires: shape {out.shape}, finite "
          f"{bool(np.isfinite(out).all())}, raw MSE {fb_mse:.4e}, "
          f"C/B {effective_compute(c.flops_used, c.residual_wall_time_s, LAMBDA_FLOPS_PER_SECOND) / FLOP_BUDGET:.4f}")
    _ = _sparse_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("alpha", "price", "bias", "score", "ship"))
    ap.add_argument("--suite", default=None)
    ap.add_argument("--n-mlps", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=str,
                    default="tau=None,n=6200;tau=2.5,n=8500,P=150")
    args = ap.parse_args()
    suite = load_suite(args.suite)
    if args.mode == "alpha":
        mode_alpha(suite, min(args.n_mlps, 8), args.n_samples)
    elif args.mode == "price":
        mode_price(suite)
    elif args.mode == "ship":
        mode_ship(suite, args.n_mlps)
    elif args.mode == "bias":
        mode_bias(suite, args.n_mlps, min(args.n_samples, 6000))
    else:
        mode_score(suite, args.n_mlps, args.grid, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
