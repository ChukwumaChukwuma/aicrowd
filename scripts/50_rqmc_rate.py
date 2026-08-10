#!/usr/bin/env python
"""Does the lattice change the RATE, or only the constant?

``scripts/27_rqmc.py --mode probe`` measured 1.910x +- 0.159 over iid at
matched compute, at ONE sample count (N = 8501).  A ratio at one ``N`` cannot
tell a better constant from a better exponent, and the whole leaderboard
argument is about the exponent: fitting ``raw = v / N^p`` to the public
per-MLP telemetry of the top entries gives ``p`` of 2.11 / 1.89 / 1.74 for
ranks 1-3 against our shipped 1.11, and ``p = 1`` is iid Monte Carlo while
``p ~ 2`` is the classical lattice rate.

So this script sweeps ``N`` over seven doublings and fits ``p`` for each
sampler, with a standard error.  Everything else here exists to make that one
fit trustworthy:

``--mode rate``
    The sweep.  ``N`` = primes just under ``2^10 .. 2^17``, iid vs
    randomly-shifted rank-1 lattice, on the SAME official MLPs, dense forward
    pass (no mask, no pilot) so the measured quantity is the sampler's own
    variance and nothing else.  Variance is measured across independent
    randomisations, which is exactly the MSE of an unbiased estimator, so no
    ground truth is needed and no reference noise enters.

``--mode unbiased``
    The proof obligation.  ``frac(k g + U) ~ U[0,1)^d`` exactly is two lines of
    algebra (see ``whestfloor/rqmc.py``); this checks it numerically — marginal
    uniformity of every one of the 256 coordinates, and common-random-number
    agreement of the lattice mean with a dense iid pass on the same MLP.

``--mode order``
    Dimension ordering: none, ``||W^1 row||`` permutation, and the mean-field
    active-subspace rotation.  A rotation is free — it folds into ``W^1`` —
    and it is exactly unbiased because ``Q x ~ N(0, I)`` for orthogonal ``Q``.

``--mode redundancy``
    The 2x2 that decides whether any of this ships: lattice x layer-1 Hermite
    control variates, all four cells, same MLPs, same randomisations.
    radiant-allomancer (18085 §2.1) reports the two levers are partly
    redundant and that this is what turned 5-7x into 1.40x for them.

Credit: the construction is evaaaz's (18053); the redundancy warning and the
antithetic refutation are radiant-allomancer's (18085).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import rqmc as R  # noqa: E402
from whestfloor.official_seeds import make_official_mlp  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402

WIDTH, DEPTH = 256, 32
DENSE_FLOPS = 4_198_656.0          # one dense forward pass, the leaderboard unit

#: primes just under 2^10 .. 2^17
N_GRID = (1021, 2039, 4093, 8191, 16381, 32749, 65521, 131071)

#: reps per (mlp, N, arm).  Falls with N so the compute per decade is roughly
#: flat; every entry is >= 8 so the variance estimate keeps ~2/(R-1) precision.
REPS = {1021: 32, 2039: 32, 4093: 32, 8191: 32,
        16381: 20, 32749: 16, 65521: 12, 131071: 8}


def art() -> Path:
    p = Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve() / "rqmc"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_suite(path: str | None) -> Suite:
    p = (Path(path) if path else
         Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()
         / "suites" / "official_mini.npz")
    return Suite.load(p)


# ---------------------------------------------------------------------------
# forward passes
# ---------------------------------------------------------------------------
def _fwd_mean(W, xgen, n_total, chunk):
    """Mean of ``relu(z^L)`` over ``n_total`` inputs produced by ``xgen(i0,i1)``."""
    acc = np.zeros(WIDTH, dtype=np.float64)
    i0 = 0
    while i0 < n_total:
        i1 = min(i0 + chunk, n_total)
        x = xgen(i0, i1)
        for w in W:
            x = np.maximum(x @ w, 0.0)
        acc += x.sum(0, dtype=np.float64)
        i0 = i1
    return acc / n_total


def iid_mean(W, n, rng, chunk=32768):
    return _fwd_mean(W, lambda a, b: rng.standard_normal((b - a, WIDTH),
                                                         dtype=np.float32),
                     n, chunk)


def lattice_mean(W, n, z, rng, chunk=32768):
    shift = rng.random(WIDTH)
    return _fwd_mean(
        W, lambda a, b: R.shifted_normals(R.lattice_rows(a, b, n, z), shift),
        n, chunk)


def _row_gen(arm, n, z, rng):
    """``gen(i0, i1) -> (i1-i0, WIDTH) float32`` for one randomisation."""
    if arm == "iid":
        return lambda a, b: rng.standard_normal((b - a, WIDTH),
                                                dtype=np.float32)
    shift = rng.random(WIDTH)
    return lambda a, b: R.shifted_normals(R.lattice_rows(a, b, n, z), shift)


def rep_means(W, n, reps, arm, z, seeds, chunk=65536):
    """Per-randomisation means, with randomisations PACKED into shared GEMMs.

    At ``N = 1021`` a per-rep forward pass is a (1021, 256) @ (256, 256)
    contraction, which runs at a fraction of peak; packing 32 of them into one
    (32672, 256) call is the same arithmetic three times faster.  The packing
    is invisible to the result — rows do not interact in a forward pass.
    """
    acc = np.zeros((reps, WIDTH))
    buf: list = []
    tags: list = []
    state = {"filled": 0}

    def flush():
        if not buf:
            return
        x = np.concatenate(buf, 0) if len(buf) > 1 else buf[0]
        for w in W:
            x = np.maximum(x @ w, 0.0)
        o = 0
        for (r, m) in tags:
            acc[r] += x[o:o + m].sum(0, dtype=np.float64)
            o += m
        buf.clear()
        tags.clear()
        state["filled"] = 0

    for r in range(reps):
        gen = _row_gen(arm, n, z, np.random.default_rng(seeds[r]))
        i0 = 0
        while i0 < n:
            take = min(n - i0, chunk - state["filled"])
            buf.append(gen(i0, i0 + take))
            tags.append((r, take))
            state["filled"] += take
            i0 += take
            if state["filled"] >= chunk:
                flush()
    flush()
    return acc / n


# ---------------------------------------------------------------------------
# mode: rate
# ---------------------------------------------------------------------------
def mode_rate(suite, n_mlps, seed, arms, ngrid, out):
    seeds = list(suite.mlp_seeds[:n_mlps])
    zc = {n: R.get_z(n, WIDTH, "cbc", verbose=False) for n in ngrid} \
        if "cbc" in arms else {}
    zr = {n: R.get_z(n, WIDTH, "roberts") for n in ngrid} if "roberts" in arms \
        else {}
    print(f"# rate sweep: {len(seeds)} official MLPs, arms={arms}")
    print(f"# N grid {ngrid}")
    print(f"# reps    {[REPS[n] for n in ngrid]}")
    store: dict = {}
    if Path(out).is_file():
        # Resume.  The sweep at N = 131,071 x 8 randomisations x 6 MLPs x 2 arms
        # is over an hour of arithmetic and this box has restarted mid-run once;
        # every (arm, N, mlp) cell already on disk is reused verbatim.
        with np.load(out) as d:
            store = {k: d[k] for k in d.files}
        print(f"# resumed {len(store)} cells from {out}")
    t0 = time.time()
    for n in ngrid:
        reps = REPS[n]
        for mi, s in enumerate(seeds):
            W = make_official_mlp(WIDTH, DEPTH, s)
            for arm in arms:
                key = f"{arm}|{n}|{mi}"
                if key in store:
                    continue
                # COMMON RANDOM NUMBERS across arms: the same rep index gets
                # the same seed in every arm, so the iid/lattice comparison is
                # paired rather than two independent draws.
                sds = [(seed + 1) * 1_000_003 + r * 7919 + mi * 104_729
                       for r in range(reps)]
                zz = {"iid": None, "cbc": zc.get(n),
                      "roberts": zr.get(n)}[arm]
                store[key] = rep_means(
                    W, n, reps, "iid" if arm == "iid" else "lat", zz, sds)
            print(f"  N={n:<7} mlp {mi + 1}/{len(seeds)} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            np.savez_compressed(out, **store)   # checkpoint per MLP, not per N
    print(f"saved {out}")
    report_rate(out, arms, ngrid, len(seeds))


def _fit_p(ns, v, w=None):
    """WLS fit of ``log v = a - p log N``; returns ``(p, a)``."""
    x = np.log(np.asarray(ns, float))
    y = np.log(np.asarray(v, float))
    if w is None:
        w = np.ones_like(y)
    A = np.stack([np.ones_like(x), -x], 1)
    WA = A * w[:, None]
    beta = np.linalg.solve(A.T @ WA, WA.T @ y)
    return float(beta[1]), float(beta[0])


def report_rate(path, arms, ngrid, n_mlps, n_boot=4000):
    d = np.load(path)
    print("\n=== per-sample-count variance of the estimator "
          "(mean over 256 neurons, over MLPs) ===")
    hdr = f"{'N':>8} " + " ".join(f"{a:>13}" for a in arms) + f"{'ratio':>10}"
    print(hdr)
    print("-" * len(hdr))
    V = {a: [] for a in arms}
    per = {a: {} for a in arms}          # per (N, mlp) rep matrices
    for n in ngrid:
        row = []
        for a in arms:
            vs = []
            for mi in range(n_mlps):
                k = f"{a}|{n}|{mi}"
                if k not in d:
                    vs = None
                    break
                M = d[k]
                per[a][(n, mi)] = M
                vs.append(M.var(0, ddof=1).mean())
            if vs is None:
                row.append(float("nan"))
                V[a].append(float("nan"))
            else:
                row.append(float(np.mean(vs)))
                V[a].append(float(np.mean(vs)))
        rat = (row[0] / row[1]) if len(row) > 1 and row[1] == row[1] else \
            float("nan")
        print(f"{n:8d} " + " ".join(f"{v:13.5e}" for v in row) +
              f"{rat:10.3f}")

    print("\n=== fitted exponent  v(N) = v0 / N^p ===")
    print(f"{'arm':>10} {'p':>8} {'+- se':>7} {'v0':>11} {'r2':>7}")
    fits = {}
    for a in arms:
        v = np.array(V[a])
        ok = np.isfinite(v)
        if ok.sum() < 3:
            continue
        ns = np.array(ngrid)[ok]
        p, a0 = _fit_p(ns, v[ok])
        yhat = a0 - p * np.log(ns)
        y = np.log(v[ok])
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # bootstrap over (reps, MLPs) jointly
        rng = np.random.default_rng(0)
        ps = []
        for _ in range(n_boot):
            mi_s = rng.integers(0, n_mlps, n_mlps)
            vb = []
            for n in ns:
                acc = []
                for mi in mi_s:
                    M = per[a][(n, int(mi))]
                    ridx = rng.integers(0, M.shape[0], M.shape[0])
                    acc.append(M[ridx].var(0, ddof=1).mean())
                vb.append(np.mean(acc))
            ps.append(_fit_p(ns, vb)[0])
        se = float(np.std(ps))
        fits[a] = (p, se, math.exp(a0), r2)
        print(f"{a:>10} {p:8.3f} {se:7.3f} {math.exp(a0):11.4e} {r2:7.4f}")
    js = art() / "rate_fit.json"
    js.write_text(json.dumps({"V": {a: V[a] for a in arms}, "N": list(ngrid),
                              "fits": fits}, indent=1))
    print(f"\nsaved {js}")
    return fits


# ---------------------------------------------------------------------------
# mode: unbiased
# ---------------------------------------------------------------------------
def mode_unbiased(suite, seed):
    rng = np.random.default_rng(seed)
    n = 8191
    z = R.get_z(n, WIDTH, "cbc")
    print("# 1. marginal uniformity of frac(k z_j / N + U_j)\n")
    ks = []
    for _ in range(20):
        shift = rng.random(WIDTH)
        u = R.lattice_rows(0, n, n, z) + shift
        u -= np.floor(u)
        us = np.sort(u, axis=0)
        grid = (np.arange(1, n + 1) / n)[:, None]
        ks.append(np.abs(us - grid).max(0))
    ks = np.array(ks)
    # For a fixed shift the empirical CDF of a 1-D lattice projection is the
    # exact grid shifted, so sup|F_n - F| <= 1/N deterministically.  That is a
    # far stronger statement than a KS test would give and it is what makes
    # every FIRST-ORDER term exact to O(N^-2).
    print(f"  sup_j sup_i |F_n(u) - u| over 20 shifts : {ks.max():.3e}")
    print(f"  1/N                                     : {1.0 / n:.3e}")
    print(f"  verdict: {'EXACT GRID' if ks.max() <= 1.5 / n else 'FAIL'}\n")

    print("# 2. mean of the estimator vs a dense iid pass, common seeds")
    W = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[0])
    gt = suite.gt[0, -1, :]
    reps = 96
    acc_l = np.zeros(WIDTH)
    acc_i = np.zeros(WIDTH)
    for r in range(reps):
        rr = np.random.default_rng(90210 + r)
        acc_l += lattice_mean(W, n, z, rr)
        rr = np.random.default_rng(90210 + r)
        acc_i += iid_mean(W, n, rr)
    ml, mi_ = acc_l / reps, acc_i / reps
    vl = float(np.mean((ml - gt) ** 2))
    vi = float(np.mean((mi_ - gt) ** 2))
    print(f"  {reps} randomisations, N={n}, official MLP {suite.mlp_seeds[0]}")
    print(f"    lattice mean vs 1e9 ground truth : {vl:.4e}")
    print(f"    iid     mean vs 1e9 ground truth : {vi:.4e}")
    print(f"    lattice-vs-iid mean discrepancy  : "
          f"{float(np.mean((ml - mi_) ** 2)):.4e}")
    print(f"    per-neuron rms bias bound (lattice): {math.sqrt(vl):.3e}")
    print("\n  Both arms are unbiased, so both of these are just "
          f"variance/{reps};\n  what matters is that the lattice number is "
          "not systematically offset.")


# ---------------------------------------------------------------------------
# mode: order
# ---------------------------------------------------------------------------
def _pilot_alpha(W, n_pilot, rng):
    """Per-layer ``alpha = m/s`` from a small pilot, as the shipped kernel does."""
    x = rng.standard_normal((n_pilot, WIDTH), dtype=np.float32)
    alpha = []
    for w in W:
        z = x @ w
        m = z.mean(0, dtype=np.float64)
        s = z.std(0, dtype=np.float64) + 1e-30
        alpha.append(m / s)
        x = np.maximum(z, 0.0)
    return alpha


def mode_order(suite, n_mlps, n, reps, seed):
    z = R.get_z(n, WIDTH, "cbc")
    seeds = list(suite.mlp_seeds[:n_mlps])
    arms = ("iid", "lattice", "lattice+rownorm", "lattice+activesub")
    out = {a: np.zeros((n_mlps, reps, WIDTH)) for a in arms}
    t0 = time.time()
    for mi, s in enumerate(seeds):
        W = make_official_mlp(WIDTH, DEPTH, s)
        alpha = _pilot_alpha(W, 4096, np.random.default_rng(5150 + mi))
        perm = R.rownorm_perm(W[0])
        Q = R.active_subspace_rotation(R.meanfield_jacobian(W, alpha))
        Wp = [W[0][perm, :].copy()] + W[1:]
        Wq = [(Q @ W[0].astype(np.float64)).astype(np.float32)] + W[1:]
        for r in range(reps):
            sd = 4242 + 31 * r + 1013 * mi
            out["iid"][mi, r] = iid_mean(W, n, np.random.default_rng(sd))
            # COMMON RANDOM SHIFT across the three lattice arms: they differ
            # only in which input direction feeds which lattice dimension, so
            # pairing them on the shift removes the shift-to-shift variance
            # from the comparison.
            for a, Wa in (("lattice", W), ("lattice+rownorm", Wp),
                          ("lattice+activesub", Wq)):
                out[a][mi, r] = lattice_mean(Wa, n, z,
                                             np.random.default_rng(sd))
        print(f"  mlp {mi + 1}/{n_mlps} [{time.time() - t0:.0f}s]", flush=True)
    print(f"\n=== dimension ordering, N={n}, {reps} randomisations, "
          f"{n_mlps} MLPs ===")
    print(f"{'arm':<20} {'variance':>12} {'x over iid':>11} {'+- se':>8}")
    base = np.mean([out['iid'][mi].var(0, ddof=1).mean()
                    for mi in range(n_mlps)])
    res = {}
    for a in arms:
        per = np.array([out[a][mi].var(0, ddof=1).mean()
                        for mi in range(n_mlps)])
        pb = np.array([out['iid'][mi].var(0, ddof=1).mean()
                       for mi in range(n_mlps)])
        g = pb.mean() / per.mean()
        rng = np.random.default_rng(0)
        bs = [pb[i].mean() / per[i].mean()
              for i in rng.integers(0, n_mlps, (2000, n_mlps))]
        res[a] = (float(per.mean()), float(g), float(np.std(bs)))
        print(f"{a:<20} {per.mean():12.5e} {g:11.3f} {np.std(bs):8.3f}")
    _ = base
    (art() / "order.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("rate", "unbiased", "order", "report"))
    ap.add_argument("--suite", default=None)
    ap.add_argument("--n-mlps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="iid,cbc")
    ap.add_argument("--ngrid", default=",".join(str(v) for v in N_GRID))
    ap.add_argument("--out", default=None)
    ap.add_argument("--N", type=int, default=32749)
    ap.add_argument("--reps", type=int, default=16)
    args = ap.parse_args()
    suite = load_suite(args.suite)
    arms = args.arms.split(",")
    ngrid = tuple(int(v) for v in args.ngrid.split(","))
    out = Path(args.out) if args.out else art() / "rate.npz"
    if args.mode == "rate":
        mode_rate(suite, args.n_mlps, args.seed, arms, ngrid, out)
    elif args.mode == "report":
        report_rate(out, arms, ngrid, args.n_mlps)
    elif args.mode == "unbiased":
        mode_unbiased(suite, args.seed)
    elif args.mode == "order":
        mode_order(suite, args.n_mlps, args.N, args.reps, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
