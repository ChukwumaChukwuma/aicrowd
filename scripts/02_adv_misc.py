#!/usr/bin/env python
"""ADVERSARIAL CHECK 5 -- everything else that could make a reported number wrong.

  5a  float32 quantisation of BOTH sides of the scored comparison.
      whestbench scoring.py:849 casts the ESTIMATOR'S PREDICTION to float32
      before scoring, and dataset targets are float32 (dataset.py:313).  Neither
      is mentioned in contract.py.  How much MSE does that cost?
  5b  precision of fnp.mean / np.mean over 256 float32 summands of size ~5e-11.
  5c  mc.layer_means chunk=1024 (repo default) vs 4096 (whestbench): does the
      SCORED final layer change?  How big is the intermediate-layer error, and
      how does it scale with n?
  5d  all_layers=False vs all_layers=True: same final layer?  Same shape?
  5e  harness.measure_flops bills ONE MLP and reuses flops_used AND
      residual_wall_time_s for the whole suite.  Is either MLP-dependent or
      run-to-run noisy, and by how much relative to the free-compute allowance?
  5f  contract.forward_pass_flops() vs a real flopscope measurement.
  5g  harness.evaluate's "whestbench-identical" final_layer_mse is measured
      against a 2*n_per_half reference, not the official 1e9 one.

Run:
    . <prefix>/runenv.sh && OPENBLAS_NUM_THREADS=1 $PY scripts/02_adv_misc.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whestfloor import contract  # noqa: E402
from whestfloor.harness import measure_flops  # noqa: E402
from whestfloor.kernels import cov_prop_gain, mean_prop  # noqa: E402
from whestfloor.mc import layer_means, make_mlp  # noqa: E402

WIDTH, DEPTH = 256, 32
V_MEASURED = 0.0550
FLOOR = V_MEASURED / 1e9


def sep(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# --------------------------------------------------------------------------
def a_quantisation():
    sep("5a  float32 quantisation of prediction and reference")
    mus = []
    for s in range(6):
        W = make_mlp(WIDTH, DEPTH, s)
        m, _ = layer_means(W, 16384, 3000 + s, want_var=False, all_layers=False)
        mus.append(m[-1])
        del W
    mu = np.concatenate(mus)
    live = mu > 0
    ulp = np.zeros_like(mu)
    ulp[live] = np.spacing(mu[live].astype(np.float32)).astype(np.float64)
    print(f"  final-layer means over 6 MLPs: n={mu.size}, "
          f"{int((~live).sum())} exactly zero (dead)")
    print(f"    mean {mu[live].mean():.4f}  max {mu.max():.4f}  "
          f"p99 {np.percentile(mu[live], 99):.4f}")
    q_mse = float(np.mean(ulp ** 2) / 12.0)
    print(f"  reference cast to float32 (simulation.py:137, dataset.py:313):")
    print(f"    mean_j ulp(mu_j)^2/12 = {q_mse:.3e}  "
          f"= {q_mse / FLOOR * 100:.4f}% of the {FLOOR:.2e} MC floor")
    print(f"  prediction cast to float32 (scoring.py:849): SAME magnitude again")
    print(f"    combined irreducible quantisation MSE ~ {2 * q_mse:.3e}  "
          f"({2 * q_mse / FLOOR * 100:.4f}% of the floor)")
    print(f"    -> per-neuron RMS {np.sqrt(2 * q_mse):.2e} vs target "
          f"{np.sqrt(FLOOR):.2e}: NEGLIGIBLE, but it is a real extra floor term")
    return mu


def b_mean_precision(mu):
    sep("5b  precision of the float32 mean over 256 summands of size ~5e-11")
    rng = np.random.default_rng(11)
    target = mu[:WIDTH].astype(np.float32)
    worst = 0.0
    for trial in range(2000):
        err = rng.standard_normal(WIDTH) * np.sqrt(FLOOR)
        pred = (target.astype(np.float64) + err).astype(np.float32)
        d32 = (pred - target) ** 2                      # float32
        m32 = float(np.mean(d32))                        # float32 accumulator
        m64 = float(np.mean(d32.astype(np.float64)))     # float64 accumulator
        worst = max(worst, abs(m32 - m64) / m64)
    print(f"  np.mean(float32) vs float64 accumulator, 2000 trials at MSE~{FLOOR:.1e}")
    print(f"    worst relative error = {worst:.3e}")
    print(f"    (256 non-negative summands, pairwise summation -> "
          f"~log2(256)*2^-24 = {8 * 2 ** -24:.1e} bound)")
    print("    VERDICT: fnp.mean does NOT lose meaningful precision here.")
    # underflow check
    tiny = np.float32(1e-9)
    print(f"    smallest squarable difference before float32 subnormals: "
          f"~{np.sqrt(np.finfo(np.float32).tiny):.2e}; "
          f"a 1e-9 difference squares to {float(tiny * tiny):.2e} (normal)")


def c_chunk():
    sep("5c  chunk 1024 (repo default) vs 4096 (whestbench _pick_chunk_size(256))")
    W = make_mlp(WIDTH, DEPTH, 0)
    print("     n      final bit-identical?  max|diff| FINAL   max|diff| all   "
          "rms diff layer 16")
    for n in (8192, 32768, 131072):
        m1, _ = layer_means(W, n, 555, chunk=1024, want_var=False, all_layers=True)
        m4, _ = layer_means(W, n, 555, chunk=4096, want_var=False, all_layers=True)
        same = np.array_equal(m1[-1], m4[-1])
        d = m1 - m4
        print(f"  {n:7d}      {str(same):>5s}                 "
              f"{np.abs(d[-1]).max():.3e}       {np.abs(d).max():.3e}      "
              f"{np.sqrt((d[15] ** 2).mean()):.3e}")
    print("  -> the SCORED final layer is bit-identical (both accumulate it in")
    print("     float64).  Intermediate layers differ because mc.py:100 sums them")
    print("     in float32 per chunk.  That error scales as ~1/sqrt(n_chunks), so")
    print("     at production n it is far below the target; at small n it is not.")
    n = 8192
    for n in (8192, 131072):
        m1, _ = layer_means(W, n, 555, chunk=1024, want_var=False, all_layers=True)
        # float64 reference for the intermediate layers
        rng = np.random.default_rng(555)
        sums = np.zeros((DEPTH, WIDTH))
        done = 0
        while done < n:
            nb = min(1024, n - done)
            x = rng.standard_normal((nb, WIDTH), dtype=np.float32)
            for li in range(DEPTH):
                x = np.maximum(x @ W[li], np.float32(0.0))
                sums[li] += x.sum(axis=0, dtype=np.float64)
            done += nb
        ref = sums / done
        d = m1 - ref
        print(f"  n={n:7d}: float32-chunk-sum error vs exact float64 accumulation: "
              f"max {np.abs(d[:-1]).max():.2e}, rms {np.sqrt((d[:-1] ** 2).mean()):.2e} "
              f"(target RMS {np.sqrt(FLOOR):.2e})")


def d_all_layers():
    sep("5d  all_layers=False vs all_layers=True")
    W = make_mlp(64, 8, 1)
    a, va = layer_means(W, 8192, 77, want_var=True, all_layers=True)
    b, vb = layer_means(W, 8192, 77, want_var=True, all_layers=False)
    print(f"  all_layers=True  -> shape {a.shape}")
    print(f"  all_layers=False -> shape {b.shape}")
    print(f"  final layer bit-identical: {np.array_equal(a[-1], b[-1])}")
    print(f"  variance bit-identical   : {np.array_equal(va, vb)}")
    print("  TRAP: with all_layers=False the array is (1, width); `means[-1]` is")
    print("  correct but `means[k]` for any k silently returns the FINAL layer,")
    print("  and broadcasting against a (depth, width) reference will not raise.")
    g = np.zeros((8, 64))
    print(f"  e.g. np.mean((b - g)**2) broadcasts silently -> "
          f"{float(np.mean((b - g) ** 2)):.4f} "
          f"instead of the all-layers value {float(np.mean((a - g) ** 2)):.4f}")


def e_flop_reuse():
    sep("5e  measure_flops bills ONE MLP and reuses it for the whole suite")
    for name, k in (("mean_prop", mean_prop), ("cov_prop_gain", cov_prop_gain)):
        fl, res, back = [], [], []
        for s in range(5):
            W = make_mlp(WIDTH, DEPTH, s)
            f, r, b = measure_flops(k, W)
            fl.append(f)
            res.append(r)
            back.append(b)
            del W
        fl = np.array(fl)
        res = np.array(res)
        print(f"  {name}")
        print(f"    flops_used over 5 MLPs      : {fl.min()} .. {fl.max()}  "
              f"(all equal: {fl.min() == fl.max()})")
        print(f"    residual_wall_time_s        : min {res.min():.4f} "
              f"max {res.max():.4f} mean {res.mean():.4f} sd {res.std(ddof=1):.4f}")
        print(f"    first-run residual          : {res[0]:.4f}  "
              f"({res[0] / res[1:].mean():.2f}x the mean of the rest)")
        lam = contract.LAMBDA_FLOPS_PER_SECOND
        c_min = fl.min() + lam * res.min()
        c_max = fl.max() + lam * res.max()
        print(f"    effective compute C         : {c_min:.3e} .. {c_max:.3e}  "
              f"(spread {100 * (c_max - c_min) / c_min:.0f}%)")
        print(f"    free allowance is {contract.FREE_COMPUTE:.3e}; "
              f"residual alone consumes "
              f"{100 * lam * res.mean() / contract.FREE_COMPUTE:.1f}% of it")
        print(f"    -> billing one MLP records a single draw of a quantity with "
              f"{100 * res.std(ddof=1) / res.mean():.0f}% run-to-run spread, "
              "with no error bar.")


def f_forward_flops():
    sep("5f  contract.forward_pass_flops() vs a real flopscope measurement")
    import flopscope as flops
    import flopscope.numpy as fnp

    W = make_mlp(WIDTH, DEPTH, 0)
    fw = [fnp.asarray(w) for w in W]
    with flops.BudgetContext(flop_budget=10 ** 12, quiet=True) as ctx:
        rng = fnp.random.default_rng(0)
        x = fnp.asarray(rng.standard_normal((1, WIDTH), dtype=fnp.float32))
        for w in fw:
            x = fnp.maximum(fnp.matmul(x, w), fnp.float32(0.0))
        _ = np.asarray(x)
    got = int(ctx.flops_used)
    want = contract.forward_pass_flops()
    print(f"  contract.forward_pass_flops(256, 32) = {want:,}")
    print(f"  flopscope measured (1 sample)        = {got:,}")
    print(f"  difference                           = {got - want:+,}  "
          f"({100 * (got - want) / want:+.3f}%)")
    print(f"  contract.MC_SAMPLES_AT_BUDGET        = {contract.MC_SAMPLES_AT_BUDGET:,}")
    print(f"  recomputed from measurement          = "
          f"{contract.FLOP_BUDGET // got:,}")


def g_reference_size():
    sep("5g  harness.final_layer_mse is NOT the leaderboard number "
        "unless gt_samples == 1e9")
    print("  harness.evaluate:123 -> fl_mse = mean((pred[-1] - g[-1])**2), "
          "g = (a+b)/2")
    print("  so the local raw MSE carries v/(2*n_per_half), not v/1e9.")
    print()
    print("   n_per_half     gt_samples     local floor     leaderboard floor   "
          "inflation")
    for nh in (2_000_000, 50_000_000, 250_000_000, 500_000_000):
        local = V_MEASURED / (2 * nh)
        print(f"  {nh:>11,}   {2 * nh:>12,}    {local:.3e}        "
              f"{FLOOR:.3e}          {local / FLOOR:7.1f}x")
    print()
    print("  A raw final_layer_mse from a dev suite is therefore NOT comparable to")
    print("  a leaderboard score; only unbiased_true_mse + v/1e9 is.")


def main():
    mu = a_quantisation()
    b_mean_precision(mu)
    c_chunk()
    d_all_layers()
    e_flop_reuse()
    f_forward_flops()
    g_reference_size()


if __name__ == "__main__":
    main()
