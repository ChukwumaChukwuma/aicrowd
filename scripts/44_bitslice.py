#!/usr/bin/env python
"""Bit-sliced, stochastically-quantised forward pass: price it, then bound it.

Why this is not the residual-time lane
--------------------------------------
Forum 18125.  @thylinao measured the bit-packing discount and @dipam replied
for the AIcrowd team: *"We are keeping the current billing and treating
bit-packing as a legitimate optimization.  The reason is that we cannot
distinguish it.  A packed word and an ordinary integer array are byte-for-byte
identical with the same dtype."*  So this is metered compute, priced by the
same meter as everything else, and the discount is an ALGORITHMIC one: 32
boolean lanes per billed FLOP.

The question this script answers
--------------------------------
``docs/graded.md`` reduced a sampler's score to ``v_eff * c / B``.  Bit-slicing
cuts ``c``; stochastic rounding pays for it in ``v_eff`` and, because the
rounding is fresh per sample, it pays in VARIANCE and not in bias -- variance
divides by ``N``, which is exactly the currency the score is denominated in.
So the whole lane is one two-dimensional optimisation,

    minimise   v_eff(b_a, b_w, schedule) * c(b_a, b_w, schedule)

against the shipped ``68,400``.

Modes
-----
``probe``   billing of every primitive, in a real ``BudgetContext``.
``ranges``  the per-neuron quantisation windows and how many octaves they span
            (this decides whether per-neuron scaling is affordable).
``sweep``   the variance/bias measurement -- raw NumPy, exact-integer
            simulation of the packed arithmetic, paired against the float pass
            on common random numbers.
``bias``    the unbiasedness proof: many independent rounding draws against one
            fixed input stream, bias reported with its standard error.
``price``   ``dF/dN`` of the real packed kernel inside a ``BudgetContext``.
``score``   the official 100-MLP suite.
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

from whestfloor.bitslice import (  # noqa: E402
    Cfg,
    np_forward,
    np_plan,
    np_ranges,
    packed_layer_cost,
)
from whestfloor.contract import DEPTH, FLOP_BUDGET, WIDTH  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402

#: The shipped operating point, from docs/cost_floor.md.
SHIP_V_EFF = 0.0245
SHIP_C = 2.79e6
SHIP_PRODUCT = SHIP_V_EFF * SHIP_C          # 68,355 ~ the documented 68,400


def artifacts() -> Path:
    p = Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve() / "bitslice"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
def oracle_stats(weights, n, seed, chunk=2048):
    """Streamed exact ``(m_z, s_z, keep)`` per layer -- the ORACLE pilot.

    Using oracle statistics isolates the quantisation effect from the pilot's
    own Monte-Carlo error, which is a separate, already-understood term.
    """
    depth, width = len(weights), weights[0].shape[0]
    s1 = np.zeros((depth, width))
    s2 = np.zeros((depth, width))
    rng = np.random.default_rng(seed)
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x = rng.standard_normal((m, width), dtype=np.float32)
        for l, w in enumerate(weights):
            z = x @ w
            s1[l] += z.sum(axis=0, dtype=np.float64)
            s2[l] += (z.astype(np.float64) ** 2).sum(axis=0)
            x = np.maximum(z, 0.0)
        done += m
    m_z = s1 / n
    v_z = np.maximum(s2 / n - m_z ** 2, 1e-30)
    return m_z.astype(np.float32), np.sqrt(v_z).astype(np.float32)


def exact_forward(x0, weights):
    h = np.asarray(x0, dtype=np.float32)
    for w in weights:
        h = np.maximum(h @ w, 0.0)
    return h


# ---------------------------------------------------------------------------
def mode_probe() -> None:
    """Every billing claim this file rests on, measured, never inferred."""
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    def bill(fn):
        with flops.BudgetContext(flop_budget=10 ** 13, quiet=True) as ctx:
            out = fn()
        return int(ctx.flops_used), out

    rng = np.random.default_rng(0)
    N, W = 64, 7                       # 7 uint32 words = 224 lanes
    a = fnp.asarray(rng.integers(0, 2 ** 32, (N, W), dtype=np.uint32))
    b = fnp.asarray(rng.integers(0, 2 ** 32, (N, W), dtype=np.uint32))
    print(f"# uint32 array {N}x{W} = {N * W} elements = {N * W * 32} boolean lanes\n")
    hdr = f"{'op':<44}{'billed':>9}{'/elem':>8}{'bools/FLOP':>12}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, fn in [
        ("bitwise_and(u32, u32)", lambda: fnp.bitwise_and(a, b)),
        ("bitwise_count(u32)", lambda: fnp.bitwise_count(a)),
        ("bitwise_or / xor / shift (each)", lambda: fnp.bitwise_or(a, b)),
        ("and + count", lambda: fnp.bitwise_count(fnp.bitwise_and(a, b))),
        ("sum(axis=-1) DEFAULT accumulator",
         lambda: fnp.sum(fnp.bitwise_count(fnp.bitwise_and(a, b)), axis=-1)),
        ("sum(axis=-1, dtype=int32)",
         lambda: fnp.sum(fnp.bitwise_count(fnp.bitwise_and(a, b)), axis=-1,
                         dtype=fnp.int32)),
    ]:
        f, o = bill(fn)
        print(f"{name:<44}{f:>9,}{f / (N * W):>8.2f}{N * W * 32 / f:>12.1f}")
        rows.append({"op": name, "billed": f, "per_elem": f / (N * W)})
    print("\n  -> the DEFAULT sum accumulator is uint64, billed at rate 2.0.")
    print("     dtype=int32 halves it.  This is the single easiest way to")
    print("     lose a third of the discount by accident.\n")

    # the float32 contraction it has to beat
    n = 224
    x = fnp.asarray(rng.standard_normal((N, n)).astype(np.float32))
    w = fnp.asarray(rng.standard_normal((n, 256)).astype(np.float32))
    f32, _ = bill(lambda: x @ w)
    print(f"float32 (N,{n}) @ ({n},256)  {f32:,} billed = "
          f"{f32 / N:,.0f}/sample = 2*{n}*256 - 256")

    # one packed (p,q) plane pair, same contraction shape
    aw = fnp.asarray(rng.integers(0, 2 ** 32, (N, W, 1), dtype=np.uint32))
    ww = fnp.asarray(rng.integers(0, 2 ** 32, (1, W, 256), dtype=np.uint32))
    fp, o = bill(lambda: fnp.sum(fnp.bitwise_count(fnp.bitwise_and(aw, ww)),
                                 axis=1, dtype=fnp.int32))
    print(f"packed  one (p,q) pair, same shape  {fp:,} billed = "
          f"{fp / N:,.0f}/sample  ({(3 * W - 1) * 256} predicted)")
    print(f"  ratio float32 / one-pair = {f32 / fp:.2f}x   "
          f"-> the CEILING is {f32 / fp:.1f}/(b_a b_w), not 32/(b_a b_w)")

    print("\n# availability on the reduced surface")
    for nm in ("packbits", "unpackbits", "bitwise_and", "bitwise_count",
               "bitwise_or", "bitwise_xor", "left_shift", "right_shift",
               "invert", "astype", "sum", "take", "cumsum", "frombuffer"):
        print(f"  fnp.{nm:<14} {hasattr(fnp, nm)}")
    print(f"  fnp.view         {hasattr(fnp, 'view')}   "
          f"<- ABSENT: uint8 -> uint32 must go through astype+shift+or")

    (artifacts() / "probe.json").write_text(json.dumps(
        {"rows": rows, "f32_per_sample": f32 / N, "packed_pair_per_sample": fp / N,
         "ceiling": f32 / fp}, indent=1))


# ---------------------------------------------------------------------------
def mode_ranges(seeds, n_stat, tau) -> None:
    """How wide, and how spread, are the per-neuron quantisation windows?

    Per-neuron scaling is only implementable in a bit-sliced product if the
    scales are octaves of a common base (see ``np_group_steps``), and each
    extra octave group costs an extra partial contraction.  So the question
    is: after the ``tau`` mask has already removed the always-off neurons,
    how many octaves are left?
    """
    print(f"# per-neuron windows, kappa=3, oracle stats from {n_stat} samples\n")
    hdr = (f"{'layer':>6}{'kept':>7}{'rms|a|':>9}{'oct p50':>9}{'oct p90':>9}"
           f"{'oct max':>9}{'sum r^2 / K max r^2':>22}")
    print(hdr)
    print("-" * len(hdr))
    agg = []
    for s in seeds:
        W = make_mlp(WIDTH, DEPTH, s)
        m_z, s_z = oracle_stats(W, n_stat, s + 7)
        alpha = m_z / s_z
        for l in range(DEPTH):
            keep = alpha[l] > -tau
            lo, hi = np_ranges(m_z[l], s_z[l], 3.0)
            r = np.maximum(hi - lo, 0.0)[keep]
            oct_ = np.log2(np.maximum(r.max(), 1e-30) / np.maximum(r, 1e-30))
            eff = float(np.sum(r ** 2) / (r.size * r.max() ** 2))
            agg.append((l, int(keep.sum()), float(np.sqrt((alpha[l] ** 2).mean())),
                        float(np.percentile(oct_, 50)),
                        float(np.percentile(oct_, 90)), float(oct_.max()), eff))
    A = np.asarray(agg)
    for l in range(DEPTH):
        r = A[A[:, 0] == l].mean(axis=0)
        if l % 4 == 0 or l >= DEPTH - 2:
            print(f"{l + 1:>6}{r[1]:>7.0f}{r[2]:>9.2f}{r[3]:>9.2f}{r[4]:>9.2f}"
                  f"{r[5]:>9.2f}{r[6]:>22.4f}")
    print(f"\nmean over layers: kept {A[:, 1].mean():.0f}/256, "
          f"octave p90 {A[:, 4].mean():.2f}, max {A[:, 5].max():.2f}")
    print("  'sum r^2 / K max r^2' is the factor a SINGLE per-layer scale")
    print("  would lose against ideal per-neuron scaling: "
          f"{1 / A[:, 6].mean():.2f}x more injected variance.")
    (artifacts() / "ranges.json").write_text(json.dumps(
        {"per_layer": A.tolist(), "tau": tau}, indent=1))


# ---------------------------------------------------------------------------
def _v_and_bias(weights, stats, cfg, n, seed, reps):
    """Paired measurement of injected variance and residual bias.

    Common random numbers: the SAME input draw feeds the exact float pass and
    every quantised replicate, so the difference isolates the quantiser.
    """
    width = weights[0].shape[0]
    rng_x = np.random.default_rng(seed)
    x0 = rng_x.standard_normal((n, width), dtype=np.float32)
    y_ex = exact_forward(x0, weights)
    v_nat = float(np.mean(np.var(y_ex, axis=0, dtype=np.float64)))
    mu_ex = y_ex.mean(axis=0, dtype=np.float64)

    # the weight quantisation is FIXED for the run -- one plan, many rounding
    # replicates -- otherwise its bias would average away across replicates
    # and be reported as zero.
    plan = np_plan(weights, stats, cfg, seed * 7 + 3)
    half = (n + 1) // 2
    v_inj, dmu = [], []
    for r in range(reps):
        rq = np.random.default_rng(seed * 131 + 977 + r)
        y_q = np_forward(x0, plan, cfg, rq)
        d = (y_q.astype(np.float64) - y_ex.astype(np.float64))
        # v_eff is N * Var(batch mean), NOT the per-sample variance: with
        # antithetic rounding the two differ by 2x and only the former is what
        # the score charges.  With sample s paired to s+half,
        #   Var(mean) = (V + C) / N,   V = per-sample variance,
        #                              C = within-pair covariance
        # so v_q_eff = V + C.  C is ~0 without pairing, which is the control.
        da, db = d[:half], d[half:half * 2]
        V = float(np.mean(np.var(d, axis=0)))
        C = float(np.mean(np.mean((da - da.mean(0)) * (db - db.mean(0)), axis=0)))
        v_inj.append(V + (C if cfg.anti else 0.0))
        dmu.append(y_q.mean(axis=0, dtype=np.float64) - mu_ex)
    D = np.asarray(dmu)                       # (reps, width)
    v_q = float(np.mean(v_inj))
    # bias = expectation of the mean shift over rounding draws; its own
    # standard error comes from the spread ACROSS replicates.
    bias = D.mean(axis=0)
    se = D.std(axis=0, ddof=1) / math.sqrt(reps) if reps > 1 else np.full(width, np.nan)
    return v_nat, v_q, bias, se, mu_ex


def mode_sweep(seeds, n, n_stat, reps, tau, grid) -> None:
    print(f"# injected variance, paired on common random numbers.\n"
          f"# {len(seeds)} local MLPs, N={n} samples, {reps} rounding replicates,\n"
          f"# oracle pilot ({n_stat} samples).  v_q is the EXTRA per-sample\n"
          f"# variance the quantiser injects into the scored neuron mean.\n")
    print(f"# ship: v_eff={SHIP_V_EFF}, c={SHIP_C:,.0f}, product={SHIP_PRODUCT:,.0f}")
    print(f"# a config wins if (v_eff + v_q) * c(b) < {SHIP_PRODUCT / 1.15:,.0f}\n")

    hdr = (f"{'config':<24}{'v_q':>9}{'c/samp':>10}{'x c':>6}{'v*c':>10}"
           f"{'x ship':>7}{'rms bias':>10}{'0.1b^2':>10}{'adjusted':>11}"
           f"{'x ship adj':>11}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for cfg in grid:
        vn, vq, bias, se, mu = [], [], [], [], []
        cost = []
        for s in seeds:
            W = make_mlp(WIDTH, DEPTH, s)
            m_z, s_z = oracle_stats(W, n_stat, s + 7)
            a, b, bi, e, m0 = _v_and_bias(W, (m_z, s_z), cfg, n, s + 11, reps)
            vn.append(a)
            vq.append(b)
            bias.append(bi)
            se.append(e)
            mu.append(m0)
            cost.append(_cost_of(cfg, m_z / s_z, tau))
        vn_, vq_ = float(np.mean(vn)), float(np.mean(vq))
        c = float(np.mean(cost))
        v_eff = SHIP_V_EFF + vq_
        prod = v_eff * c
        B, S = np.concatenate(bias), np.concatenate(se)
        rb = float(np.sqrt(np.mean(B ** 2)))
        b2 = max(float(np.mean(B ** 2) - np.mean(S ** 2)), 0.0)
        # the score in full: adjusted = 0.1 b^2 + v_eff c / B  (docs/graded.md
        # sec 3).  Reporting v_eff*c alone is only honest for an UNBIASED
        # estimator, and part B of --mode bias shows this one is not.
        adj = 0.1 * b2 + prod / FLOP_BUDGET
        print(f"{cfg.label():<24}{vq_:>9.4f}{c:>10,.0f}{SHIP_C / c:>6.2f}"
              f"{prod:>10,.0f}{SHIP_PRODUCT / prod:>7.3f}{rb:>10.2e}"
              f"{0.1 * b2:>10.2e}{adj:>11.3e}{2.4646e-07 / adj:>11.4f}")
        out.append({"config": cfg.label(), "ba": cfg.ba, "bw": cfg.bw,
                    "kappa": cfg.kappa, "groups": cfg.groups, "anti": cfg.anti,
                    "wmeanfix": cfg.wmeanfix, "v_nat": vn_, "v_q": vq_,
                    "v_eff": v_eff, "c": c, "v_eff_times_c": prod,
                    "gain_vs_ship": SHIP_PRODUCT / prod, "rms_bias": rb,
                    "bias_sq_unbiased": b2, "adjusted": adj,
                    "gain_vs_ship_adjusted": 2.4646e-07 / adj})
    (artifacts() / "sweep.json").write_text(json.dumps(
        {"n": n, "reps": reps, "seeds": seeds, "rows": out}, indent=1))
    best = min(out, key=lambda r: r["v_eff_times_c"])
    print(f"\nbest: {best['config']}  v_eff*c = {best['v_eff_times_c']:,.0f}  "
          f"= {best['gain_vs_ship']:.3f}x the ship")


def mode_layers(seeds, n, n_stat, reps, tau, ba, bw, kappa, groups) -> None:
    """Which LAYER's quantisation noise actually reaches the score?

    Quantise exactly one layer and leave the other 31 exact.  The injected
    variance ``v_q(l)`` is then the per-layer sensitivity, and because the
    noise sources are independent the schedule problem separates:
    ``v_q(total) = sum_l v_q(l)``, each falling as ``4^-b_l`` while the cost
    rises linearly in ``b_l`` -- a water-filling problem with a closed-form
    solution (see ``--mode schedule``).
    """
    print(f"# one layer quantised at a time, a{ba}w{bw} k{kappa} G{groups}, "
          f"N={n}, {len(seeds)} MLPs\n")
    hdr = (f"{'layer':>6}{'v_q(l)':>11}{'share %':>9}{'bias(l) rms':>13}"
           f"{'cum share':>11}")
    print(hdr)
    print("-" * len(hdr))
    per = np.zeros(DEPTH)
    bia = np.zeros(DEPTH)
    for s in seeds:
        W = make_mlp(WIDTH, DEPTH, s)
        m_z, s_z = oracle_stats(W, n_stat, s + 7)
        for l in range(DEPTH):
            cfg = Cfg(ba=ba, bw=bw, kappa=kappa, groups=groups,
                      exact_layers=[k for k in range(DEPTH) if k != l])
            _, vq, bi, _, _ = _v_and_bias(W, (m_z, s_z), cfg, n, s + 11, reps)
            per[l] += vq / len(seeds)
            bia[l] += float(np.sqrt(np.mean(bi ** 2))) / len(seeds)
    tot = per.sum()
    cum = 0.0
    for l in range(DEPTH):
        cum += per[l] / tot
        print(f"{l + 1:>6}{per[l]:>11.3e}{100 * per[l] / tot:>9.2f}"
              f"{bia[l]:>13.3e}{cum:>11.3f}")
    print(f"\nsum of per-layer v_q = {tot:.4f}")
    print(f"top-8 layers carry {np.sort(per)[-8:].sum() / tot:.1%} of it; "
          f"a uniform schedule would say 25%.")
    (artifacts() / "layers.json").write_text(json.dumps(
        {"ba": ba, "bw": bw, "kappa": kappa, "groups": groups,
         "v_q_per_layer": per.tolist(), "bias_per_layer": bia.tolist()}, indent=1))


def mode_price(seeds, tau, n_pilot, grid) -> None:
    """``dF/dN`` of the REAL packed kernel, in a real ``BudgetContext``.

    Never inferred: a two-point difference in ``N`` on the same MLP, which
    cancels the per-MLP plan (packing the weights, the masks, the pilot) and
    leaves exactly the per-sample cost the score charges.
    """
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    from whestfloor import kernels  # noqa: PLC0415
    from whestfloor.bitslice import bitsliced_sparse_kernel  # noqa: PLC0415

    W = [fnp.asarray(w) for w in make_mlp(WIDTH, DEPTH, seeds[0])]

    def run(fn, N):
        with flops.BudgetContext(flop_budget=int(1e13), quiet=True) as c:
            fn(N)
        return int(c.flops_used), float(c.residual_wall_time_s)

    n1, n2 = 3000, 7000
    print(f"# dF/dN from N={n1} -> {n2}, one LOCAL MLP (seed {seeds[0]}), "
          f"tau={tau}, P={n_pilot}\n")
    hdr = (f"{'variant':<22}{'dF/dN':>12}{'model':>12}{'model/meas':>12}"
           f"{'x ship c':>10}{'resid ms':>10}")
    print(hdr)
    print("-" * len(hdr))
    f1, r1 = run(lambda N: kernels.sparse_mc_kernel(
        W, tau=tau, n_samples=N, n_pilot=n_pilot, seed=1, safe=False), n1)
    f2, _ = run(lambda N: kernels.sparse_mc_kernel(
        W, tau=tau, n_samples=N, n_pilot=n_pilot, seed=1, safe=False), n2)
    ship = (f2 - f1) / (n2 - n1)
    print(f"{'ship: float32 sparse':<22}{ship:>12,.0f}{'':>12}{'':>12}"
          f"{1.0:>10.3f}{r1 * 1e3:>10.1f}")
    rows = [{"variant": "ship", "dFdN": ship}]
    for ba, bw, kap in grid:
        def mk(N, ba=ba, bw=bw, kap=kap):
            return bitsliced_sparse_kernel(
                W, tau=tau, n_samples=N, n_pilot=n_pilot, seed=1, ba=ba,
                bw=bw, kappa=kap, safe=False, chunk=None)
        f1b, r1b = run(mk, n1)
        f2b, _ = run(mk, n2)
        c = (f2b - f1b) / (n2 - n1)
        mdl = _model_cost(ba, bw, tau, seeds[0])
        print(f"{f'packed a{ba} w{bw}':<22}{c:>12,.0f}{mdl:>12,.0f}"
              f"{mdl / c:>12.3f}{ship / c:>10.3f}{r1b * 1e3:>10.1f}")
        rows.append({"variant": f"a{ba}w{bw}", "ba": ba, "bw": bw, "dFdN": c,
                     "model": mdl, "gain_vs_ship": ship / c})
    (artifacts() / "price.json").write_text(json.dumps(
        {"tau": tau, "n_pilot": n_pilot, "rows": rows}, indent=1))


def _model_cost(ba, bw, tau, seed, lanes=32):
    """The cost model, on the same mask the kernel would build."""
    W = make_mlp(WIDTH, DEPTH, seed)
    m, s = oracle_stats(W, 8192, seed + 7)
    keep = (m / s) > -tau
    total, k_prev = 0.0, WIDTH
    for l in range(DEPTH):
        k = WIDTH if l == DEPTH - 1 else int(keep[l].sum())
        k = -(-k // lanes) * lanes
        total += packed_layer_cost(k_prev, k, ba, bw)
        k_prev = k
    return total


def _cost_of(cfg, alpha, tau):
    """Billed FLOPs per sample for this schedule, on this MLP's mask."""
    keep = alpha > -tau
    total = 0.0
    k_prev = WIDTH
    for l in range(DEPTH):
        k_out = WIDTH if l == DEPTH - 1 else int(keep[l].sum())
        k_out = -(-k_out // 32) * 32        # whole uint32 words, as the kernel
        total += packed_layer_cost(k_prev, k_out, cfg.ba_at(l), cfg.bw_at(l))
        k_prev = k_out
    return total


# ---------------------------------------------------------------------------
def mode_bias(seeds, n, n_stat, reps, cfgs) -> None:
    """Is the stochastic rounding unbiased?  Yes -- and it does not matter.

    The brief's premise was that stochastic rounding is unbiased, so
    quantisation error is variance, and variance divides by N.  The first half
    is true and is proved here directly.  The second half does not follow, and
    the reason is the only interesting thing in this file.

    PART A runs the quantiser through ONE contraction and no relu.  There the
    rounding is exactly unbiased and the measurement says so: the bias sits
    inside its own standard error at every precision.

    PART B runs the same quantiser through 1, 2, 4, ... 32 relu layers.  relu
    is CONVEX, so E[relu(z + eps)] > E[relu(z)] whenever eps has any variance
    at all: an unbiased perturbation of the pre-activation is a BIASED
    perturbation of the activation, by (1/2) v phi(alpha)/s to leading order,
    and that shift then propagates and compounds.  Unbiasedness of the
    ROUNDING is not unbiasedness of the ESTIMATOR, and only the latter is what
    ``0.1 b^2`` charges for.
    """
    from whestfloor.bitslice import np_quant_act, np_quant_weight  # noqa: PLC0415

    print("# PART A -- one contraction, NO relu.  Is the rounding unbiased?")
    print(f"# {reps} independent rounding draws, N={n} fixed inputs, paired.\n")
    hdr = (f"{'b_a':>5}{'b_w':>5}{'rms bias':>12}{'rms se':>11}"
           f"{'|bias|/se':>11}{'max |t|':>10}{'verdict':>12}")
    print(hdr)
    print("-" * len(hdr))
    rowsA = []
    W = make_mlp(WIDTH, DEPTH, seeds[0])[0]
    rx = np.random.default_rng(4242)
    h = np.maximum(rx.standard_normal((n, WIDTH), dtype=np.float32), 0.0)
    exact = (h @ W).mean(axis=0, dtype=np.float64)
    for ba, bw in ((2, 4), (4, 6), (6, 8)):
        lo = np.zeros(WIDTH, dtype=np.float32)
        step = np.full(WIDTH, float(h.max()) / (2 ** ba - 1), dtype=np.float32)
        qw, ws = np_quant_weight(W, bw, np.random.default_rng(9), stochastic=True)
        what = qw * ws[None, :]
        D = []
        for r in range(reps):
            q = np_quant_act(h, lo, step, float(2 ** ba - 1),
                             np.random.default_rng(600 + r))
            hq = lo[None, :] + q * step[None, :]
            z = hq @ what
            z = z + hq.mean(axis=0, dtype=np.float64).astype(np.float32) @ (W - what)
            D.append(z.mean(axis=0, dtype=np.float64) - exact)
        D = np.asarray(D)
        bi, se = D.mean(0), D.std(0, ddof=1) / math.sqrt(reps)
        rb, rs = float(np.sqrt(np.mean(bi ** 2))), float(np.sqrt(np.mean(se ** 2)))
        t = np.abs(bi) / np.maximum(se, 1e-300)
        v = "UNBIASED" if rb < 2 * rs else "biased"
        print(f"{ba:>5}{bw:>5}{rb:>12.3e}{rs:>11.3e}{rb / rs:>11.2f}"
              f"{t.max():>10.2f}{v:>12}")
        rowsA.append({"ba": ba, "bw": bw, "rms_bias": rb, "rms_se": rs,
                      "ratio": rb / rs, "max_t": float(t.max())})

    print("\n# PART B -- the SAME quantiser through d relu layers.")
    print(f"# bias of the layer-d mean, {reps} rounding draws, N={n}.\n")
    hdr = (f"{'config':<18}{'depth':>6}{'v_q':>10}{'rms bias':>11}"
           f"{'rms se':>10}{'b/se':>8}{'0.1b^2':>11}{'x score':>9}")
    print(hdr)
    print("-" * len(hdr))
    rowsB = []
    for cfg in cfgs:
        for d in (1, 2, 4, 8, 16, 32):
            for s in seeds[:1]:
                Wd = make_mlp(WIDTH, DEPTH, s)[:d]
                m_z, s_z = oracle_stats(Wd, n_stat, s + 7)
                _, vq, bi, se, _ = _v_and_bias(Wd, (m_z, s_z), cfg, n, s + 11, reps)
                rb = float(np.sqrt(np.mean(bi ** 2)))
                rs = float(np.sqrt(np.mean(se ** 2)))
                b2 = max(float(np.mean(bi ** 2) - np.mean(se ** 2)), 0.0)
                print(f"{cfg.label():<18}{d:>6}{vq:>10.5f}{rb:>11.3e}"
                      f"{rs:>10.2e}{rb / rs:>8.1f}{0.1 * b2:>11.3e}"
                      f"{0.1 * b2 / 2.4646e-07:>9.1f}")
                rowsB.append({"config": cfg.label(), "depth": d, "v_q": vq,
                              "rms_bias": rb, "rms_se": rs,
                              "adjusted_cost_of_bias": 0.1 * b2,
                              "x_shipped_score": 0.1 * b2 / 2.4646e-07})
    print("\nPART A says the rounding is unbiased.  PART B says the ESTIMATOR")
    print("is not, and that the bias appears the moment a relu is in the path")
    print("and then grows with depth.  ``0.1 b^2`` is charged on the second.")
    (artifacts() / "bias.json").write_text(json.dumps(
        {"n": n, "reps": reps, "linear": rowsA, "relu": rowsB}, indent=1))


# ---------------------------------------------------------------------------
_SANDBOX_PROBE = '''# Fresh-interpreter probe: the packed pipeline with the repo off sys.path
import sys, json
sys.dont_write_bytecode = True
import importlib.util
assert importlib.util.find_spec("whestfloor") is None, "repo still importable"
import flopscope as flops
import flopscope.numpy as fnp
out = {}
out["numpy_importable_by_us"] = importlib.util.find_spec("numpy") is not None
missing = [n for n in ("packbits","bitwise_and","bitwise_count","bitwise_or",
                       "left_shift","astype","sum","uint32","uint8","int32",
                       "minimum","maximum","floor","random")
           if not hasattr(fnp, n)]
out["missing"] = missing
with flops.BudgetContext(flop_budget=10**11, quiet=True) as ctx:
    g = fnp.random.default_rng(0)
    K, M, N, ba, bw = 64, 8, 6, 3, 4
    h = g.standard_normal((N, K), dtype=fnp.float32)
    lv = float(2 ** ba - 1)
    q = fnp.minimum(fnp.maximum(fnp.floor((h + 2.0) * (lv / 4.0)
                    + g.random((N, K), dtype=fnp.float32)), 0.0), lv)
    qi = q.astype(fnp.uint8)
    wq = (g.random((K, M), dtype=fnp.float32) * (2 ** bw - 1)).astype(fnp.uint8)

    def pack_rows(x, nb):
        r = []
        for p in range(nb):
            p8 = fnp.packbits(fnp.bitwise_and(x, 1 << p), axis=1)
            a = p8[:, 0::4].astype(fnp.uint32)
            for j in (1, 2, 3):
                a = fnp.bitwise_or(a, fnp.left_shift(
                    p8[:, j::4].astype(fnp.uint32), 8 * j))
            r.append(a)
        return r

    def pack_cols(x, nb):
        r = []
        for p in range(nb):
            p8 = fnp.packbits(fnp.bitwise_and(x, 1 << p), axis=0)
            a = p8[0::4, :].astype(fnp.uint32)
            for j in (1, 2, 3):
                a = fnp.bitwise_or(a, fnp.left_shift(
                    p8[j::4, :].astype(fnp.uint32), 8 * j))
            r.append(a)
        return r

    A, W = pack_rows(qi, ba), pack_cols(wq, bw)
    acc = None
    for p, a in enumerate(A):
        for qq, wv in enumerate(W):
            s = fnp.sum(fnp.bitwise_count(fnp.bitwise_and(a[:, :, None],
                                                          wv[None, :, :])),
                        axis=1, dtype=fnp.int32)
            s = s if p + qq == 0 else fnp.left_shift(s, p + qq)
            acc = s if acc is None else acc + s
    out["packed_shape"] = list(acc.shape)
    out["packed_dtype"] = str(acc.dtype)
    out["flops"] = int(ctx.flops_used)
    # exactness check WITHOUT numpy: reconstruct the integer product from the
    # flopscope arrays themselves via a float32 matmul, then compare.
    ref = qi.astype(fnp.float32) @ wq.astype(fnp.float32)
    d = fnp.max(fnp.abs(acc.astype(fnp.float32) - ref))
    out["max_abs_diff_vs_float_matmul"] = float(d)
print("@@" + json.dumps(out))
'''


def mode_sandbox() -> None:
    """Run the packed pipeline in a FRESH interpreter with the repo off sys.path.

    The grader supplies flopscope, whestbench and a reduced stdlib -- no numpy
    -- and on the eval servers ``fnp.ndarray`` is a ``RemoteArray`` with no
    ``.base``.  This probe therefore uses NOTHING but ``flopscope.numpy``: no
    ``np.`` anywhere, no ``.base``, no ``.view`` (which does not exist in
    ``fnp`` at all, hence the hand-rolled uint8 -> uint32 gather).  It also
    checks the packed product against a float32 matmul of the same codes
    *inside* flopscope, so the exactness assertion needs no host-side numpy
    either.
    """
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "probe.py"
        f.write_text(_SANDBOX_PROBE)
        env = dict(os.environ)
        # keep flopscope (and the opt_einsum it imports) reachable, drop the
        # repo -- matched on the resolved repo root, not on a substring, since
        # the scratch prefix can itself contain the repo's name
        root = str(Path(__file__).resolve().parent.parent)
        keep = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                if p and str(Path(p).resolve()) != root]
        env["PYTHONPATH"] = os.pathsep.join(keep)
        r = subprocess.run([sys.executable, str(f)], capture_output=True,
                           text=True, cwd=td, env=env, timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        print("SANDBOX PROBE FAILED\n", r.stdout, r.stderr)
        raise SystemExit(1)
    out = json.loads(line[2:])
    print("# fresh interpreter, repo off sys.path, flopscope.numpy only\n")
    for k, v in out.items():
        print(f"  {k:<32} {v}")
    ok = (not out["missing"]) and out["max_abs_diff_vs_float_matmul"] == 0.0
    print(f"\n  {'PASS' if ok else 'FAIL'}: every packed primitive is present "
          f"and the packed product is exact in a repo-free interpreter.")
    (artifacts() / "sandbox.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("probe", "sandbox", "ranges", "sweep", "bias",
                             "layers", "price", "score"))
    ap.add_argument("--seeds", default="700000,700001,700002")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--n-stat", type=int, default=32768)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--grid", default="main")
    ap.add_argument("--ba", type=int, default=4)
    ap.add_argument("--bw", type=int, default=6)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--groups", type=int, default=6)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]

    t0 = time.time()
    if a.mode == "probe":
        mode_probe()
    elif a.mode == "sandbox":
        mode_sandbox()
    elif a.mode == "ranges":
        mode_ranges(seeds, a.n_stat, a.tau)
    elif a.mode == "layers":
        mode_layers(seeds, a.n, a.n_stat, a.reps, a.tau,
                    a.ba, a.bw, a.kappa, a.groups)
    elif a.mode == "price":
        grid = [(2, 4, 1.5), (3, 5, 1.9), (4, 6, 2.2), (5, 6, 2.5), (6, 6, 2.8)]
        mode_price(seeds, a.tau, 225, grid)
    elif a.mode in ("sweep", "bias"):
        grid = build_grid(a.grid)
        if a.mode == "sweep":
            mode_sweep(seeds, a.n, a.n_stat, a.reps, a.tau, grid)
        else:
            mode_bias(seeds, a.n, a.n_stat, a.reps, grid)
    else:
        print(f"mode {a.mode} not implemented")
        return 1
    print(f"\n[{time.time() - t0:.1f}s]")
    return 0


def build_grid(name):
    if name == "main":
        return [Cfg(ba=b, bw=b, kappa=3.0, groups=6) for b in (2, 3, 4, 5, 6)]
    if name == "opt":
        # the definitive grid: best kappa, octave groups, antithetic rounding
        return [Cfg(ba=ba, bw=bw, kappa=2.0, groups=6, anti=True)
                for ba in (2, 3, 4, 5, 6) for bw in (4, 5, 6)]
    if name == "anti":
        return [Cfg(ba=ba, bw=5, kappa=2.0, groups=6, anti=a)
                for ba in (3, 4, 5) for a in (False, True)]
    if name == "kappa2":
        return [Cfg(ba=4, bw=5, kappa=k, groups=6, anti=True)
                for k in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)]
    if name == "bakappa":
        # kappa MUST be co-optimised with b_a: the injected variance is
        # step^2/6 + the clipped tail, and only the first term falls with b.
        return [Cfg(ba=ba, bw=6, kappa=k, groups=6, anti=True)
                for ba in (3, 4, 5, 6)
                for k in (1.75, 2.0, 2.25, 2.5, 2.75, 3.0)]
    if name == "final":
        # kappa is the per-b_a argmin of the k-sweep; b_w spans the range over
        # which the weight term goes from dominant to negligible.
        return [Cfg(ba=ba, bw=bw, kappa=k, groups=6, anti=True)
                for (ba, k) in ((2, 1.5), (3, 1.9), (4, 2.2), (5, 2.5),
                                (6, 2.8), (7, 3.0))
                for bw in (4, 5, 6)]
    if name == "groups2":
        return [Cfg(ba=4, bw=5, kappa=2.0, groups=G, anti=True)
                for G in (1, 2, 4, 6, 8, 12)]
    if name == "asym":
        return [Cfg(ba=ba, bw=bw, kappa=3.0, groups=6)
                for ba in (2, 3, 4, 5) for bw in (3, 4, 5, 6, 8)]
    if name == "kappa":
        return [Cfg(ba=4, bw=6, kappa=k, groups=6) for k in (2.0, 2.5, 3.0, 3.5, 4.0)]
    if name == "groups":
        return [Cfg(ba=4, bw=6, kappa=3.0, groups=G) for G in (1, 2, 3, 4, 6, 8, 12)]
    if name == "switches":
        base = dict(ba=4, bw=6, kappa=3.0, groups=6)
        return [Cfg(**base), Cfg(**base, anti=True),
                Cfg(**base, wmeanfix=False), Cfg(**base, stoch_w=False),
                Cfg(**base, stoch_w=False, wmeanfix=False),
                Cfg(**base, stochastic=False)]
    if name == "biasproof":
        return [Cfg(ba=2, bw=4, kappa=1.5, groups=6, anti=True),
                Cfg(ba=4, bw=6, kappa=2.2, groups=6, anti=True),
                Cfg(ba=6, bw=6, kappa=2.8, groups=6, anti=True)]
    if name == "wbits":
        return [Cfg(ba=4, bw=bw, kappa=3.0, groups=6) for bw in (3, 4, 5, 6, 8, 32)]
    raise SystemExit(f"unknown grid {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())
