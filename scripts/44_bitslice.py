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
    dense_layer_cost,
    np_forward,
    np_group_steps,
    np_quant_weight,
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

    v_inj, dmu = [], []
    for r in range(reps):
        rq = np.random.default_rng(seed * 131 + 977 + r)
        y_q = np_forward(x0, weights, stats, cfg, rq)
        d = (y_q.astype(np.float64) - y_ex.astype(np.float64))
        v_inj.append(float(np.mean(np.var(d, axis=0))))
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

    hdr = (f"{'config':<26}{'v_nat':>9}{'v_q':>10}{'v_q/v_nat':>11}"
           f"{'c/samp':>10}{'x c':>7}{'v*c':>11}{'x ship':>8}{'rms bias':>10}")
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
        rb = float(np.sqrt(np.mean(np.concatenate(bias) ** 2)))
        print(f"{cfg.label():<26}{vn_:>9.4f}{vq_:>10.4f}{vq_ / vn_:>11.3f}"
              f"{c:>10,.0f}{SHIP_C / c:>7.2f}{prod:>11,.0f}"
              f"{SHIP_PRODUCT / prod:>8.3f}{rb:>10.2e}")
        out.append({"config": cfg.label(), "ba": cfg.ba, "bw": cfg.bw,
                    "kappa": cfg.kappa, "groups": cfg.groups, "anti": cfg.anti,
                    "wmeanfix": cfg.wmeanfix, "v_nat": vn_, "v_q": vq_,
                    "v_eff": v_eff, "c": c, "v_eff_times_c": prod,
                    "gain_vs_ship": SHIP_PRODUCT / prod, "rms_bias": rb})
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


def _cost_of(cfg, alpha, tau):
    """Billed FLOPs per sample for this schedule, on this MLP's mask."""
    keep = alpha > -tau
    total = 0.0
    k_prev = WIDTH
    for l in range(DEPTH):
        k_out = WIDTH if l == DEPTH - 1 else int(keep[l].sum())
        total += packed_layer_cost(k_prev, k_out, cfg.ba_at(l), cfg.bw_at(l))
        k_prev = k_out
    return total


# ---------------------------------------------------------------------------
def mode_bias(seeds, n, n_stat, reps, cfgs) -> None:
    """Is the stochastic rounding unbiased?  The whole lane depends on it.

    A biased quantiser enters the score as ``0.1 * b^2`` and does NOT divide by
    ``N``; an unbiased one enters as variance and does.  ``reps`` independent
    rounding draws share ONE input stream and one exact reference, so the
    difference of the means is an estimate of the bias whose standard error is
    the across-replicate spread -- no reference-noise term at all.
    """
    print(f"# unbiasedness: {reps} independent rounding draws on ONE fixed\n"
          f"# input stream of N={n}, vs the exact float pass on the same stream.\n")
    hdr = (f"{'config':<26}{'rms bias':>11}{'rms se':>11}{'bias/se':>9}"
           f"{'max |t|':>9}{'0.1*b^2':>11}{'vs score':>10}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for cfg in cfgs:
        B, S = [], []
        for s in seeds:
            W = make_mlp(WIDTH, DEPTH, s)
            m_z, s_z = oracle_stats(W, n_stat, s + 7)
            _, _, bi, e, _ = _v_and_bias(W, (m_z, s_z), cfg, n, s + 11, reps)
            B.append(bi)
            S.append(e)
        B, S = np.concatenate(B), np.concatenate(S)
        rb, rs = float(np.sqrt(np.mean(B ** 2))), float(np.sqrt(np.mean(S ** 2)))
        t = np.abs(B) / np.maximum(S, 1e-30)
        # the bias estimate is itself noisy; the UNBIASED estimate of the
        # squared bias removes the estimator's own variance
        b2 = max(float(np.mean(B ** 2) - np.mean(S ** 2)), 0.0)
        print(f"{cfg.label():<26}{rb:>11.3e}{rs:>11.3e}{rb / rs:>9.2f}"
              f"{t.max():>9.2f}{0.1 * b2:>11.3e}{0.1 * b2 / 2.4646e-07:>10.3f}")
        rows.append({"config": cfg.label(), "rms_bias": rb, "rms_se": rs,
                     "unbiased_b2": b2, "adjusted_cost_of_bias": 0.1 * b2,
                     "fraction_of_shipped_score": 0.1 * b2 / 2.4646e-07})
    print("\n'0.1*b^2' is what a bias of this size would ADD to the adjusted")
    print("score; 'vs score' expresses it as a fraction of the shipped")
    print("2.4646e-07.  A value well under 1 means the rounding is unbiased")
    print("enough that the lane lives or dies on variance alone.")
    (artifacts() / "bias.json").write_text(json.dumps(
        {"n": n, "reps": reps, "rows": rows}, indent=1))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("probe", "ranges", "sweep", "bias", "layers",
                             "price", "score"))
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
    elif a.mode == "ranges":
        mode_ranges(seeds, a.n_stat, a.tau)
    elif a.mode == "layers":
        mode_layers(seeds, a.n, a.n_stat, a.reps, a.tau,
                    a.ba, a.bw, a.kappa, a.groups)
    elif a.mode in ("sweep", "bias"):
        grid = build_grid(a.grid)
        (mode_sweep if a.mode == "sweep" else mode_bias)(
            seeds, a.n, a.n_stat, a.reps,
            *( (a.tau, grid) if a.mode == "sweep" else (grid,) ))
    else:
        print(f"mode {a.mode} not implemented yet")
        return 1
    print(f"\n[{time.time() - t0:.1f}s]")
    return 0


def build_grid(name):
    if name == "main":
        g = []
        for b in (2, 3, 4, 5, 6):
            g.append(Cfg(ba=b, bw=b, kappa=3.0, groups=6))
        return g
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
    if name == "wbits":
        return [Cfg(ba=4, bw=bw, kappa=3.0, groups=6) for bw in (3, 4, 5, 6, 8, 32)]
    raise SystemExit(f"unknown grid {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())
