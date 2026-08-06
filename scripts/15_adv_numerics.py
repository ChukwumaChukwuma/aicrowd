#!/usr/bin/env python
"""ADVERSARIAL: numerical failure hunting in the shipped estimator.

This runs the **real flopscope arithmetic** -- the same calls, in the same
order, as ``whestfloor.kernels.cov_prop_edgeworth`` / ``submission/estimator``
-- inside a real ``BudgetContext``, and pulls the intermediates out to count:

  * how often ``var_pre`` is clamped at the 1e-12 floor;
  * how often ``rho`` is clipped at +-1 off the diagonal (a clip means the
    propagated covariance is inconsistent with its own diagonal);
  * how often ``sig`` underflows to zero or ``alpha`` goes non-finite;
  * how often the correction drives the predicted mean NEGATIVE.  The estimand
    is E[relu(.)] >= 0, so a negative prediction is out of range by
    construction, and the uncorrected propagation can never produce one
    (``m Phi(a) + s phi(a) > 0`` for every finite ``a``).  Any negative is the
    correction's doing;
  * how often the correction exceeds the quantity it corrects;
  * any non-finite entry in the returned prediction.

A parity check against ``cov_prop_edgeworth`` is printed first, so the counts
are known to belong to the shipped path and not to a look-alike.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

from whestfloor import kernels  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402

VAR_FLOOR = 1e-12
BIG_BUDGET = 10 ** 15


def instrumented(weights_np, kmax=4, umax=1, damp=1.0):
    """cov_prop_edgeworth, statement for statement, with counters."""
    st = dict(var_floor_hits=0, rho_clip=0, rho_clip_hi=0, sig_zero=0,
              alpha_nonfinite=0, neg_mean=0, neg_mean_before=0,
              corr_dominates=0, nonfinite=0, n_cells=0, n_offdiag=0,
              max_corr_ratio=0.0, worst_neg=0.0, k3_nonfinite=0,
              min_var_pre=np.inf)

    with flops.BudgetContext(flop_budget=BIG_BUDGET, quiet=True):
        weights = [fnp.asarray(w) for w in weights_np]
        n = weights[0].shape[0]
        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32),
                                 symmetry=(0, 1))
        prev = None
        rows = []
        for w in weights:
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            raw = np.asarray(fnp.diag(cov_pre))
            st["var_floor_hits"] += int(np.sum(raw <= VAR_FLOOR))
            st["min_var_pre"] = min(st["min_var_pre"], float(raw.min()))
            st["n_cells"] += n
            var_pre = fnp.maximum(fnp.diag(cov_pre), VAR_FLOOR)
            sig = fnp.sqrt(var_pre)
            st["sig_zero"] += int(np.sum(np.asarray(sig) == 0.0))
            alpha = mu_pre / sig
            st["alpha_nonfinite"] += int(np.sum(~np.isfinite(np.asarray(alpha))))
            ph = flops.stats.norm.pdf(alpha)
            Ph = flops.stats.norm.cdf(alpha)
            mu = mu_pre * Ph + sig * ph
            ez2 = (mu_pre * mu_pre + var_pre) * Ph + mu_pre * sig * ph
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)
            st["neg_mean_before"] += int(np.sum(np.asarray(mu) < 0.0))

            if prev is not None:
                k3 = kernels._kappa3_star(w, prev[0], prev[1], umax)
                k3n = np.asarray(k3)
                st["k3_nonfinite"] += int(np.sum(~np.isfinite(k3n)))
                delta = (damp / 6.0) * k3 * (mu_pre / (var_pre * sig)) * ph
                dn = np.asarray(delta)
                mn = np.asarray(mu)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.abs(dn) / np.maximum(np.abs(mn), 1e-300)
                st["corr_dominates"] += int(np.sum(r > 1.0))
                if np.isfinite(r).any():
                    st["max_corr_ratio"] = max(st["max_corr_ratio"],
                                               float(np.nanmax(r)))
                mu = mu - delta
            mn = np.asarray(mu)
            nneg = int(np.sum(mn < 0.0))
            st["neg_mean"] += nneg
            if nneg:
                st["worst_neg"] = min(st["worst_neg"], float(mn.min()))

            inv_sig = 1.0 / sig
            rho = cov_pre * fnp.outer(inv_sig, inv_sig)
            rn = np.asarray(rho)
            off = ~np.eye(n, dtype=bool)
            st["rho_clip"] += int(np.sum(np.abs(rn[off]) > 1.0))
            st["rho_clip_hi"] += int(np.sum(rn[off] > 1.0))
            st["n_offdiag"] += int(off.sum())
            rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)

            a = kernels._hermite_coeffs(alpha, sig, ph, Ph,
                                        max(kmax, 2 * umax))
            acc = fnp.outer(a[1], a[1]) * rho
            rho_k = rho
            fact = 1.0
            for k in range(2, kmax + 1):
                rho_k = rho_k * rho
                fact *= k
                acc = acc + fnp.outer(a[k], a[k]) * (rho_k * (1.0 / fact))
            cov = acc
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))
            prev = (a, rho)
            rows.append(mu)
        out = np.asarray(fnp.stack(rows, axis=0), dtype=np.float64)
    st["nonfinite"] = int(np.sum(~np.isfinite(out)))
    return out, st


def parity_check():
    from whestfloor.harness import run_billed
    W = make_mlp(256, 32, 12345)
    ref, _f, _r = run_billed(functools.partial(kernels.cov_prop_edgeworth,
                                               kmax=4, umax=1, damp=1.0), W)
    mine, _ = instrumented(W, kmax=4, umax=1, damp=1.0)
    d = float(np.max(np.abs(ref - mine)))
    print(f"parity with cov_prop_edgeworth(kmax=4,umax=1): max|diff| = {d:.3e}"
          f"   {'IDENTICAL' if d == 0.0 else 'DIFFERS'}")
    print()


def run_batch(seeds, width, depth, damp, umax, label):
    tot = {}
    per_neg, per_floor = [], []
    for sd in seeds:
        W = make_mlp(width, depth, sd)
        _out, st = instrumented(W, umax=umax, damp=damp)
        for k, v in st.items():
            if k.startswith("max_"):
                tot[k] = max(tot.get(k, 0.0), v)
            elif k.startswith("worst_") or k.startswith("min_"):
                tot[k] = min(tot.get(k, np.inf), v)
            else:
                tot[k] = tot.get(k, 0) + v
        per_neg.append(st["neg_mean"])
        per_floor.append(st["var_floor_hits"])
    nc, no = tot["n_cells"], tot["n_offdiag"]
    print(f"[{label}] {width}x{depth} damp={damp} umax={umax} "
          f"n_mlps={len(seeds)}  ({nc:,} neuron-layer cells)")
    print(f"   var_pre at 1e-12 floor : {tot['var_floor_hits']:,} "
          f"({tot['var_floor_hits'] / nc:.4%})  min var_pre seen "
          f"{tot['min_var_pre']:.3e}")
    print(f"   sig==0 / alpha nonfin  : {tot['sig_zero']:,} / "
          f"{tot['alpha_nonfinite']:,}")
    print(f"   |rho|>1 clipped        : {tot['rho_clip']:,} of {no:,} "
          f"({tot['rho_clip'] / no:.4%})")
    print(f"   NEGATIVE mean predicted: {tot['neg_mean']:,} "
          f"({tot['neg_mean'] / nc:.4%})  worst {tot['worst_neg']:.4e}"
          f"   [uncorrected: {tot['neg_mean_before']:,}]")
    print(f"   |correction|>|mean|    : {tot['corr_dominates']:,} "
          f"({tot['corr_dominates'] / nc:.4%})  max ratio "
          f"{tot['max_corr_ratio']:.3e}")
    print(f"   non-finite out / k3    : {tot['nonfinite']:,} / "
          f"{tot['k3_nonfinite']:,}")
    print(f"   per-MLP negatives      : {per_neg}")
    print()
    return tot


def range_cost(suite_path):
    """Does enforcing E[relu] >= 0 -- the estimand's own range -- cost MSE?

    If clipping the out-of-range predictions back to the feasible set makes the
    score WORSE, the correction is buying accuracy by leaving the range of the
    quantity it estimates, which is not a property a correct estimator has.
    """
    from whestfloor.harness import run_billed
    from whestfloor.suite import Suite
    s = Suite.load(suite_path)
    print("=" * 76)
    print(f"D. cost of enforcing the range constraint  (suite {s.name})")
    print("=" * 76)
    for nm, fn in (("mehler_k4",
                    functools.partial(kernels.cov_prop_mehler, kmax=4)),
                   ("edgeworth d=1",
                    functools.partial(kernels.cov_prop_edgeworth, kmax=4,
                                      umax=1, damp=1.0)),
                   ("edgeworth d=2.25",
                    functools.partial(kernels.cov_prop_edgeworth, kmax=4,
                                      umax=1, damp=2.25))):
        raw, clip, nneg = [], [], 0
        for i in range(s.n_mlps):
            w = s.weights(i)
            pred, _f, _r = run_billed(fn, w)
            p = pred[-1]
            nneg += int(np.sum(p < 0))
            a, b = s.gt_a[i][-1], s.gt_b[i][-1]
            raw.append(float(np.mean((p - a) * (p - b))))
            pc = np.maximum(p, 0.0)
            clip.append(float(np.mean((pc - a) * (pc - b))))
            del w
        r, c = float(np.mean(raw)), float(np.mean(clip))
        print(f"  {nm:18s} unclipped {r:.4e}   clipped>=0 {c:.4e}   "
              f"delta {(c - r) / r:+.3%}   final-layer negatives "
              f"{nneg}/{s.n_mlps * s.width}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mlps", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=31000)
    ap.add_argument("--skip-hostile", action="store_true")
    ap.add_argument("--only-range", type=str, default=None,
                    help="path to a suite; run only the range-cost section")
    args = ap.parse_args()

    if args.only_range:
        range_cost(args.only_range)
        return 0

    parity_check()
    seeds = [args.seed_base + i for i in range(args.n_mlps)]

    print("=" * 76)
    print("A. competition shape 256x32, shipped configuration")
    print("=" * 76)
    run_batch(seeds, 256, 32, 1.0, 1, "shipped damp=1")
    run_batch(seeds[:6], 256, 32, 0.0, 1, "control damp=0")
    run_batch(seeds[:6], 256, 32, 2.25, 1, "damp=2.25")

    print("=" * 76)
    print("B. shapes where the correction compounds further")
    print("=" * 76)
    run_batch(seeds[:6], 256, 64, 1.0, 1, "256x64")
    run_batch(seeds[:6], 64, 64, 1.0, 1, "64x64")
    run_batch(seeds[:6], 32, 96, 1.0, 1, "32x96")
    run_batch(seeds[:6], 16, 128, 1.0, 1, "16x128")

    if args.skip_hostile:
        return 0
    print("=" * 76)
    print("C. hostile weight matrices (not He-initialised)")
    print("=" * 76)
    n, d = 64, 16
    rng = np.random.default_rng(7)
    base = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32)
            for _ in range(d)]
    dead = [w.copy() for w in base]
    dead[3][:, 0] = 0.0
    cases = {
        "rank-1 layers": [np.outer(rng.standard_normal(n),
                                   rng.standard_normal(n)).astype(np.float32)
                          for _ in range(d)],
        "tiny scale 1e-8": [(rng.standard_normal((n, n)) * 1e-8).astype(np.float32)
                            for _ in range(d)],
        "huge scale 1e3": [(rng.standard_normal((n, n)) * 1e3).astype(np.float32)
                           for _ in range(d)],
        "all zeros": [np.zeros((n, n), dtype=np.float32) for _ in range(d)],
        "identity": [np.eye(n, dtype=np.float32) for _ in range(d)],
        "one dead column": dead,
    }
    for nm, W in cases.items():
        try:
            out, st = instrumented(W, umax=1, damp=1.0)
            fin = out[np.isfinite(out)]
            rng_s = (f"[{fin.min():.3e}, {fin.max():.3e}]" if fin.size
                     else "ALL NON-FINITE")
            print(f"  {nm:18s} floor {st['var_floor_hits']:6d}  "
                  f"rho-clip {st['rho_clip']:8d}  neg {st['neg_mean']:5d}  "
                  f"nonfin {st['nonfinite']:5d}  "
                  f"max|d|/|mu| {st['max_corr_ratio']:.2e}  out {rng_s}")
        except Exception as e:  # noqa: BLE001
            print(f"  {nm:18s} RAISED {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
