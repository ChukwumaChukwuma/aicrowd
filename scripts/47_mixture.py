#!/usr/bin/env python
"""Is the top of the leaderboard a Gaussian mixture?  Measure the ceiling first.

Telemetry (public per-MLP pages, no auth) puts the top two entries at 6.30e9 and
2.43e10 billed FLOPs with raw MSE 9.0e-9 and 3.6e-9.  `docs/cost_floor.md` rules
out a sampler at that price and `docs/traj_closure.md` rules out a moment
closure at that accuracy, so what is left is a deterministic method costing a
handful of covariance propagations.  A `K`-component Gaussian mixture is exactly
that shape: `K` propagations, each component rectified in closed form.

**Every mixture method's answer has the same form**

    E[relu(z^L_j)]  ~=  sum_k w_k relu_mean(m_kj, s_kj),

so before building a propagator this script measures the best that form can
possibly do: take the *exact* law of `z^32` from Monte Carlo, cut it into `K`
cells along a chosen `r`-dimensional frame, and evaluate the formula with the
*true* per-cell conditional mean and standard deviation.  No propagation error,
no closure error, no fitting — only the representation error of a `K`-component
mixture.  That is a ceiling on the whole family, and it is cheap.

Modes
-----
``ceiling``   the ``(r, K)`` ceiling table, three frames, unbiased error.
``anatomy``   where the closure error lives in the eigenbasis, and the
              per-direction non-Gaussianity — the two premises the mixture
              hypothesis rests on, measured rather than assumed.
``propagate`` the self-contained mixture propagator: raw MSE and billed ``F``
              in a real ``flopscope.BudgetContext``.

Errors are reported as ``mean_i (p_i - a_i)(p_i - b_i)`` against two independent
reference halves, which removes the reference's own Monte-Carlo variance
exactly (``whestfloor/harness.py``).  The cell statistics come from a *third*
independent stream, so the predictor is independent of both halves and the
estimate is unbiased for the finite-sample predictor's true MSE; the run is
repeated at half the cell-stream size so the predictor's own noise is visible
rather than assumed away.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import forward_pass_flops  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.mixture import (  # noqa: E402
    CellAccumulator,
    MixtureState,
    frame_eig,
    frame_input,
    frame_kurt,
    lloyd,
    pooled_moments,
    predict_state,
)
from whestfloor.relu_moments import relu_mean  # noqa: E402
from whestfloor.trajclosure import run as chain_run  # noqa: E402

ART = Path(os.environ.get("WHEST_ARTIFACTS", "/tmp/whest")) / "mixture"

#: Seeds disjoint from every other page in this repository (docs/traj_closure.md
#: uses 960000+, the official suite uses the whestbench protocol).
MLP_SEED_BASE = 970_000


# ---------------------------------------------------------------------------
def stream_z(W, n_samples, seed, chunk=8192, want_x=False):
    """Yield ``(X, Z)`` chunks of input and final-layer *pre-activation*."""
    n = W[0].shape[0]
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        h = x
        for li, w in enumerate(W):
            z = h @ w
            if li == len(W) - 1:
                break
            h = np.maximum(z, np.float32(0.0))
        yield (x if want_x else None), z
        done += nb


def collect(W, n_samples, seed, chunk=8192, want_x=False):
    n = W[0].shape[0]
    Z = np.empty((n_samples, n), dtype=np.float32)
    X = np.empty((n_samples, n), dtype=np.float32) if want_x else None
    off = 0
    for x, z in stream_z(W, n_samples, seed, chunk, want_x):
        nb = z.shape[0]
        Z[off:off + nb] = z
        if want_x:
            X[off:off + nb] = x
        off += nb
    return X, Z


def truth_mean(W, n_samples, seed, chunk=8192):
    n = W[0].shape[0]
    acc = np.zeros(n, dtype=np.float64)
    tot = 0
    for _, z in stream_z(W, n_samples, seed, chunk):
        acc += np.maximum(z, 0.0).sum(axis=0, dtype=np.float64)
        tot += z.shape[0]
    return acc / tot


def umse(p, a, b):
    """Unbiased true MSE with the reference's own sampling variance removed."""
    return float(np.mean((p - a) * (p - b)))


# ---------------------------------------------------------------------------
#: (frame, r, [K, ...]).  ``r = 0`` is the K = 1 Gaussian oracle.
PLAN_FULL = [
    ("eig", 1, [2, 3, 4, 6, 8, 12, 16, 32, 64, 256, 1024]),
    ("eig", 2, [4, 6, 8, 16, 64, 256]),
    ("eig", 4, [6, 16, 64, 256]),
    ("eig", 8, [6, 64, 256]),
    ("eig", 16, [6, 64, 256]),
    ("eig", 32, [6, 64, 256]),
    ("kurt", 1, [6, 64]), ("kurt", 2, [6, 64]), ("kurt", 4, [6, 64]),
    ("input", 1, [6, 64]), ("input", 2, [6, 64]), ("input", 4, [6, 64]),
]
PLAN_QUICK = [("eig", 1, [6, 64]), ("eig", 4, [6, 64]), ("input", 1, [6])]


def _build_specs(plan, Zp, Xp):
    """Frames and Lloyd centroids from the pilot.  Returns specs + m0."""
    m0, U, ev = frame_eig(Zp)
    Zc = Zp.astype(np.float64) - m0
    rmax_k = max([r for (f, r, _) in plan if f == "kurt"], default=0)
    rmax_x = max([r for (f, r, _) in plan if f == "input"], default=0)
    Uk = frame_kurt(Zp, rmax_k) if rmax_k else None
    Ux = frame_input(Xp, Zp, rmax_x) if rmax_x else None
    Tp = {"eig": Zc @ U, "kurt": (Zc @ Uk) if Uk is not None else None,
          "input": (Xp.astype(np.float64) @ Ux) if Ux is not None else None}
    Uf = {"eig": U / np.sqrt(ev), "kurt": Uk, "input": Ux}
    Tp["eig"] = Tp["eig"] / np.sqrt(ev)
    specs = []
    for fname, r, Ks in plan:
        for K in Ks:
            C = lloyd(np.ascontiguousarray(Tp[fname][:, :r]), K, seed=1234 + K)
            specs.append((fname, r, K, "x" if fname == "input" else "z",
                          np.ascontiguousarray(Uf[fname][:, :r],
                                               dtype=np.float32), C))
    return specs, m0, ev


def _run_stream(W, specs, m0, n_samples, seed, chunk, marks):
    """One independent stream.  Returns ``{size: (preds, truth)}``.

    ``preds[c]`` is configuration ``c``'s answer and ``truth`` the plain sample
    mean of ``relu(z^32)`` **from the very same samples**.  Pairing them is the
    whole point: both are averages over one empirical distribution, so their
    difference cancels the leading Monte-Carlo fluctuation, and the product of
    two such differences from independent streams is an unbiased estimate of
    the squared model error with a variance far below what an independent
    reference could give at any affordable sample count.
    """
    n = 256
    accs = [CellAccumulator(C, n) for (*_, C) in specs]
    tsum = np.zeros(n)
    done = 0
    out = {}
    nxt = sorted(marks)
    m0f = np.asarray(m0, dtype=np.float32)
    for x, z in stream_z(W, n_samples, seed, chunk, want_x=True):
        B = z.shape[0]
        ZZ = np.empty((B, 2 * n), dtype=np.float32)
        ZZ[:, :n] = z
        np.multiply(z, z, out=ZZ[:, n:])
        tsum += np.maximum(z, np.float32(0.0)).sum(axis=0, dtype=np.float64)
        zc = z - m0f
        for acc, (_, _, _, kind, U, _) in zip(accs, specs):
            acc.add(ZZ, (zc if kind == "z" else x) @ U)
        done += B
        while nxt and done >= nxt[0]:
            sz = nxt.pop(0)
            out[sz] = ([predict_state(*a.state(), n) for a in accs],
                       tsum / done, pooled_moments(*accs[0].state(), n), done)
    return out


def mode_ceiling(args) -> int:
    ART.mkdir(parents=True, exist_ok=True)
    plan = PLAN_QUICK if args.quick else PLAN_FULL
    N = args.n_cells
    marks = [N // 4, N // 2, N]
    acc_rows: dict[tuple, dict[int, list]] = {}
    base: dict[str, dict[int, list]] = {"gauss": {}, "chain": {}, "pt": {}}

    print("# mixture ceiling: the best that  sum_k w_k relu_mean(m_k, s_k)  can do")
    print(f"# {args.mlps} MLPs, two independent streams of {N:,} samples each,")
    print("# paired same-stream residuals, Richardson-extrapolated in 1/N")
    print()

    for i in range(args.mlps):
        t0 = time.time()
        seed = MLP_SEED_BASE + i
        W = make_mlp(256, 32, seed)
        Xp, Zp = collect(W, args.n_pilot, seed=5_000_000 + i, want_x=True)
        specs, m0, ev = _build_specs(plan, Zp, Xp)
        del Xp, Zp
        t1 = time.time()
        s1 = _run_stream(W, specs, m0, N, 6_000_000 + 2 * i, args.chunk, marks)
        s2 = _run_stream(W, specs, m0, N, 6_000_000 + 2 * i + 1, args.chunk, marks)
        mu_chain = chain_run(W)[-1]
        for sz in marks:
            p1, t_1, (m1, v1), _ = s1[sz]
            p2, t_2, (m2, v2), _ = s2[sz]
            base["gauss"].setdefault(sz, []).append(float(np.mean(
                (relu_mean(m1, np.sqrt(v1)) - t_1)
                * (relu_mean(m2, np.sqrt(v2)) - t_2))))
            base["pt"].setdefault(sz, []).append(float(np.mean(
                (np.maximum(m1, 0.0) - t_1) * (np.maximum(m2, 0.0) - t_2))))
            base["chain"].setdefault(sz, []).append(
                float(np.mean((mu_chain - t_1) * (mu_chain - t_2))))
            for c, sp in enumerate(specs):
                acc_rows.setdefault(sp[:3], {}).setdefault(sz, []).append(
                    float(np.mean((p1[c] - t_1) * (p2[c] - t_2))))
        print(f"  mlp {i + 1}/{args.mlps} seed={seed}  pilot {t1 - t0:.0f}s "
              f"streams {time.time() - t1:.0f}s   K=1 oracle "
              f"{base['gauss'][N][-1]:.4e}  chain {base['chain'][N][-1]:.4e}",
              flush=True)

    def rich(d):
        """2 * MSE(N) - MSE(N/2): the 1/N term removed."""
        return 2.0 * float(np.mean(d[N])) - float(np.mean(d[N // 2]))

    gb = rich(base["gauss"])
    out = {"n_mlps": args.mlps, "n_cells": N, "n_pilot": args.n_pilot,
           "mlp_seeds": [MLP_SEED_BASE + i for i in range(args.mlps)],
           "baselines": {k: {str(s): float(np.mean(v)) for s, v in d.items()}
                         for k, d in base.items()},
           "baselines_extrap": {k: rich(d) for k, d in base.items()},
           "table": {"|".join(map(str, k)):
                     {str(s): float(np.mean(v)) for s, v in d.items()}
                     for k, d in acc_rows.items()},
           "table_extrap": {"|".join(map(str, k)): rich(d)
                            for k, d in acc_rows.items()}}

    print()
    print(f"# baselines, raw MSE (extrapolated), {args.mlps} MLPs")
    print(f"    analytic Gaussian chain, no oracle      "
          f"{rich(base['chain']):.4e}")
    print(f"    K=1 Gaussian, EXACT mean and variance   {gb:.4e}")
    print(f"    relu(exact mean), i.e. s -> 0           {rich(base['pt']):.4e}")
    print()
    hdr = (f"  {'frame':>6} {'r':>3} {'K':>5}   {'N/4':>11} {'N/2':>11} "
           f"{'N':>11}   {'extrap':>11} {'x K=1':>7} {'rms':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in sorted(acc_rows, key=lambda t: (t[0], t[1], t[2])):
        d = acc_rows[key]
        e = rich(d)
        print(f"  {key[0]:>6} {key[1]:>3} {key[2]:>5}   "
              f"{np.mean(d[N // 4]):11.4e} {np.mean(d[N // 2]):11.4e} "
              f"{np.mean(d[N]):11.4e}   {e:11.4e} {gb / e if e > 0 else float('nan'):7.2f} "
              f"{np.sqrt(e) if e > 0 else float('nan'):10.3e}")

    (ART / f"ceiling{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(f"\n# wrote {ART / f'ceiling{args.tag}.json'}")
    return 0


# ---------------------------------------------------------------------------
def mode_anatomy(args) -> int:
    """The two premises the mixture hypothesis rests on, measured.

    (a) "the closure error at L=32 is concentrated in the top eigendirection of
        Cov(h^32)" — project the error onto the eigenbasis and read the
        cumulative share.
    (b) "the non-Gaussianity is concentrated in very few directions" — the
        participation ratio of the variance versus the participation ratio of
        the *excess kurtosis*, which is the quantity a mixture actually
        removes.
    """
    ART.mkdir(parents=True, exist_ok=True)
    out = {"mlps": [], "n_samples": args.n_pilot}
    for i in range(args.mlps):
        seed = MLP_SEED_BASE + i
        W = make_mlp(256, 32, seed)
        _, Z = collect(W, args.n_pilot, seed=5_000_000 + i)
        m0, U, ev = frame_eig(Z)
        Zd = Z.astype(np.float64)
        truth = np.maximum(Zd, 0.0).mean(axis=0)
        s = np.sqrt(np.maximum(np.diag(np.cov(Zd.T, bias=True)), 1e-30))
        err_oracle = relu_mean(m0, s) - truth
        err_chain = chain_run(W)[-1] - truth

        # (a) error energy in the eigenframe of Cov(relu(z))
        H = np.maximum(Zd, 0.0)
        Hc = H - H.mean(axis=0)
        Sh = (Hc.T @ Hc) / Zd.shape[0]
        wh, Vh = np.linalg.eigh(Sh)
        o = np.argsort(wh)[::-1]
        Vh = Vh[:, o]
        wh = wh[o]
        cum = {}
        for e, nm in ((err_oracle, "oracle"), (err_chain, "chain")):
            c = (Vh.T @ e) ** 2
            c = np.cumsum(c) / c.sum()
            cum[nm] = [float(c[j - 1]) for j in (1, 2, 4, 8, 16, 32, 64, 256)]

        # (b) participation ratios
        pr_var = float(ev.sum() ** 2 / (ev ** 2).sum())
        T = (Zd - m0) @ U
        t = T / np.sqrt(np.maximum(ev, 1e-30))
        kurt = np.abs((t ** 4).mean(axis=0) - 3.0)
        skew = np.abs((t ** 3).mean(axis=0))
        pr_kurt = float(kurt.sum() ** 2 / (kurt ** 2).sum())
        pr_skew = float(skew.sum() ** 2 / (skew ** 2).sum())
        rec = {
            "seed": seed,
            "ev_top1_share": float(ev[0] / ev.sum()),
            "pr_variance": pr_var,
            "pr_abs_excess_kurtosis": pr_kurt,
            "pr_abs_skew": pr_skew,
            "kurt_top1_share": float(kurt[0] / kurt.sum()),
            "kurt_of_top_eigendir": float(kurt[0]),
            "kurt_max_over_eigendirs": float(kurt.max()),
            "kurt_argmax_eigendir": int(np.argmax(kurt)),
            "err_cum_share_oracle": cum["oracle"],
            "err_cum_share_chain": cum["chain"],
            "rms_oracle": float(np.sqrt(np.mean(err_oracle ** 2))),
            "rms_chain": float(np.sqrt(np.mean(err_chain ** 2))),
        }
        out["mlps"].append(rec)
        print(f"  seed {seed}: PR(var)={pr_var:.1f}  top1 var share="
              f"{rec['ev_top1_share']:.3f}  PR(|kurt|)={pr_kurt:.1f}  "
              f"kurt top1 share={rec['kurt_top1_share']:.3f}", flush=True)

    def avg(k):
        return np.mean([m[k] for m in out["mlps"]], axis=0)

    print()
    print("# premise (a): is the closure error concentrated in the top "
          "eigendirections of Cov(h^32)?")
    print(f"  {'directions':>12} {'oracle closure':>16} {'analytic chain':>16}"
          f" {'isotropic':>11}")
    for j, nd in enumerate((1, 2, 4, 8, 16, 32, 64, 256)):
        print(f"  {nd:>12} {avg('err_cum_share_oracle')[j]:>16.3f} "
              f"{avg('err_cum_share_chain')[j]:>16.3f} {nd / 256:>11.3f}")
    print()
    print("# premise (b): variance is low-rank; is the non-Gaussianity?")
    print(f"  participation ratio of the variance        {avg('pr_variance'):.1f}")
    print(f"  participation ratio of |excess kurtosis|   "
          f"{avg('pr_abs_excess_kurtosis'):.1f}")
    print(f"  participation ratio of |skewness|          {avg('pr_abs_skew'):.1f}")
    print(f"  top-1 eigendirection: share of variance    {avg('ev_top1_share'):.3f}")
    print(f"  top-1 eigendirection: share of |kurtosis|  {avg('kurt_top1_share'):.3f}")
    (ART / "anatomy.json").write_text(json.dumps(out, indent=1))
    print(f"\n# wrote {ART / 'anatomy.json'}")
    return 0


# ---------------------------------------------------------------------------
def mixture_predict(W, K, r=1, nodes=None, split_every=1, kmax=8,
                    reduce_mode="runnalls"):
    """The deployable propagator.  Pure NumPy; ``mode_propagate`` bills it."""
    n = W[0].shape[1]
    W64 = [w.astype(np.float64) for w in W]
    st = MixtureState.gaussian(np.zeros(n), W64[0].T @ W64[0])
    nodes = nodes or K
    for l in range(len(W)):
        if l % split_every == 0 and st.K * nodes <= K * nodes:
            for _ in range(r):
                u = st.top_direction()
                st = st.split(u, nodes)
            if st.K > K:
                st = st.reduce_to(K)
        if l + 1 < len(W):
            st = st.relu_step(W64[l + 1], kmax=kmax,
                              exact_acos=(l == 0 and st.K == 1))
        else:
            return st.relu_mean_mixture()
    raise AssertionError


def mode_propagate(args) -> int:
    import flopscope as flops  # noqa: PLC0415

    ART.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.mlps):
        seed = MLP_SEED_BASE + i
        W = make_mlp(256, 32, seed)
        a = truth_mean(W, args.n_truth, seed=8_000_000 + 2 * i, chunk=args.chunk)
        b = truth_mean(W, args.n_truth, seed=8_000_000 + 2 * i + 1, chunk=args.chunk)
        base = umse(chain_run(W)[-1], a, b)
        for K in [int(v) for v in args.k.split(",")]:
            t0 = time.time()
            p = mixture_predict(W, K, r=args.rsplit, split_every=args.split_every)
            rows.append({"seed": seed, "K": K, "raw": umse(p, a, b),
                         "base": base, "wall": time.time() - t0})
            print(f"  seed {seed} K={K:3d}  raw={rows[-1]['raw']:.4e}  "
                  f"chain={base:.4e}  {rows[-1]['wall']:.1f}s", flush=True)
    print()
    print(f"  {'K':>4} {'raw MSE':>12} {'x chain':>9} {'F (flopscope)':>15} "
          f"{'x fwd pass':>11}")
    fwd = forward_pass_flops()
    for K in sorted({r["K"] for r in rows}):
        v = float(np.mean([r["raw"] for r in rows if r["K"] == K]))
        bv = float(np.mean([r["base"] for r in rows if r["K"] == K]))
        rows_f = [r for r in rows if r["K"] == K]
        F = rows_f[0].get("F", 0)
        print(f"  {K:>4} {v:12.4e} {bv / v:9.2f} {F:15,d} {F / fwd:11.0f}")
    (ART / "propagate.json").write_text(json.dumps(rows, indent=1))
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ceiling",
                    choices=("ceiling", "anatomy", "propagate"))
    ap.add_argument("--mlps", type=int, default=4)
    ap.add_argument("--n-pilot", type=int, default=150_000)
    ap.add_argument("--n-cells", type=int, default=1_000_000)
    ap.add_argument("--n-truth", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--k", default="2,4,6,8")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rsplit", type=int, default=1)
    ap.add_argument("--split-every", type=int, default=1)
    args = ap.parse_args()
    return {"ceiling": mode_ceiling, "anatomy": mode_anatomy,
            "propagate": mode_propagate}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
