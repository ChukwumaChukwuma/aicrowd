#!/usr/bin/env python
"""ADVERSARIAL CHECK 2 -- is the float32 reference biased vs exact arithmetic?

The grader's reference is  E[ fl32(forward(x)) ]  with float32 sgemm at every
layer.  A white-box estimator computes  E[ exact_forward(x) ]  in real
arithmetic.  If those differ by more than the per-neuron target RMS error, the
project is capped no matter how good the estimator is.

Three gaps are separated here, all with COMMON RANDOM NUMBERS so the O(1e-3)
Monte-Carlo noise cancels between arms and only the systematic part survives:

  x64  = float64 standard normal draw
  x32  = fl32(x64)                     (exactly representable in float64)

  A : x32  -> float32 sgemm chain, ReLU in float32     (the grader's pipeline)
  B : x32  -> float64 dgemm chain, ReLU in float64     (exact arith, same input)
  C : x64  -> float64 dgemm chain, ReLU in float64     (exact arith, exact input)

  A - B  = pure ARITHMETIC gap        <- the headline number
  B - C  = pure INPUT-QUANTISATION gap
  A - C  = total gap the estimator cannot remove

Every gap is reported as an UNBIASED squared bias, using two independent halves
of the sample stream:  bias^2_j estimated by  d_j^(1) * d_j^(2),  which removes
the residual MC variance exactly.  Without that subtraction the reported RMS is
just the MC noise level and says nothing.

A fourth check bounds the float32 standard-normal GENERATOR (numpy's float32
ziggurat is a different algorithm from the float64 one, so it is a fourth,
independent source of bias in the reference).

Run:
    . <prefix>/runenv.sh && OPENBLAS_NUM_THREADS=1 $PY scripts/02_adv_f32bias.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whestfloor.mc import make_mlp  # noqa: E402

WIDTH = 256
DEPTH = 32
N_MLPS = int(os.environ.get("ADV_N_MLPS", 3))
N_SAMPLES = int(os.environ.get("ADV_N_SAMPLES", 131072))  # per MLP, split in 2 halves
CHUNK = 1024

# Target per-neuron RMS error.  Both the contract's assumption and the value
# measured by 02_adv_variance.py are shown, because claim 2 must be judged
# against the *smaller* (tighter) one.
TARGET_RMS_CONTRACT = float(np.sqrt(0.18 / 1e9))     # 1.342e-5
TARGET_RMS_MEASURED = float(np.sqrt(0.0550 / 1e9))   # 7.42e-6


def run_one_mlp(seed: int, n_samples: int, rng_seed: int):
    """Return per-layer half-sums for the three arms, shape (2, DEPTH, WIDTH)."""
    W32 = make_mlp(WIDTH, DEPTH, seed)
    W64 = [w.astype(np.float64) for w in W32]
    rng = np.random.default_rng(rng_seed)

    sums = np.zeros((3, 2, DEPTH, WIDTH), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    half_edge = n_samples // 2

    done = 0
    while done < n_samples:
        nb = min(CHUNK, n_samples - done)
        h = 0 if done < half_edge else 1
        x64 = rng.standard_normal((nb, WIDTH))
        x32 = x64.astype(np.float32)

        a = x32
        b = x32.astype(np.float64)
        c = x64
        for li in range(DEPTH):
            a = np.maximum(a @ W32[li], np.float32(0.0))
            b = np.maximum(b @ W64[li], 0.0)
            c = np.maximum(c @ W64[li], 0.0)
            sums[0, h, li] += a.sum(axis=0, dtype=np.float64)
            sums[1, h, li] += b.sum(axis=0)
            sums[2, h, li] += c.sum(axis=0)
        counts[h] += nb
        done += nb

    means = sums / counts[None, :, None, None]
    del W32, W64
    return means  # (3, 2, DEPTH, WIDTH)


def unbiased_sq_bias(d1: np.ndarray, d2: np.ndarray):
    """Unbiased mean squared systematic difference from two independent halves.

    d1, d2 are the (mean_arm1 - mean_arm2) vectors measured on half 1 / half 2.
    Returns (sq_bias, se, tau2, naive_sq).
    """
    j = d1.shape[0]
    tau2 = float(np.mean((d1 - d2) ** 2) / 2.0)  # per-neuron variance of a half-mean
    prod = d1 * d2
    sq = float(np.mean(prod))
    d2c = np.maximum(prod, 0.0)
    se = float(np.sqrt(np.sum(2.0 * d2c * tau2 + tau2 * tau2) / (j * j)))
    naive = float(np.mean(((d1 + d2) / 2.0) ** 2))
    return sq, se, tau2, naive


def rms(x: float) -> float:
    return float(np.sqrt(x)) if x > 0 else -float(np.sqrt(-x))


def generator_check() -> None:
    print()
    print("=== 2d: numpy float32 standard-normal generator ===")
    rng = np.random.default_rng(31337)
    n_total = 200_000_000
    s = 0.0
    s2 = 0.0
    amax = 0.0
    done = 0
    t0 = time.perf_counter()
    while done < n_total:
        nb = min(20_000_000, n_total - done)
        x = rng.standard_normal(nb, dtype=np.float32)
        s += float(x.sum(dtype=np.float64))
        s2 += float(np.dot(x.astype(np.float64), x.astype(np.float64)))
        amax = max(amax, float(np.abs(x).max()))
        done += nb
    m = s / done
    m2 = s2 / done
    print(f"  n = {done:,}  ({time.perf_counter() - t0:.1f}s)")
    print(f"  E[x]   = {m:+.3e}   (MC SE = {1 / np.sqrt(done):.2e})")
    print(f"  E[x^2] = {m2:.8f}   (MC SE = {np.sqrt(2 / done):.2e}), "
          f"deviation {m2 - 1:+.2e}")
    print(f"  max |x| = {amax:.4f}")
    print("  ULP argument: every draw is a float32, so each value carries a")
    print("  relative rounding of <= 2^-24 = 5.96e-08; the ziggurat strip table is")
    print("  also stored in float32.  Hence |E[x^2]/1 - 1| <~ 2*2^-24 = 1.2e-07.")
    print("  A relative error delta in E[x^2] moves the layer-32 mean by about")
    print("  0.5*delta*mean ~ 0.35*delta, i.e. <~ 4.2e-08 -- 180x below the")
    print(f"  tightest target RMS ({TARGET_RMS_MEASURED:.2e}).")


def main() -> None:
    print(f"CRN float32-vs-float64 study: {N_MLPS} MLPs x {N_SAMPLES:,} samples "
          f"(width {WIDTH}, depth {DEPTH})")
    all_means = []
    t0 = time.perf_counter()
    for m in range(N_MLPS):
        all_means.append(run_one_mlp(seed=m, n_samples=N_SAMPLES,
                                     rng_seed=700_000 + m))
        print(f"  mlp {m} done ({time.perf_counter() - t0:.1f}s)")
    means = np.stack(all_means)  # (M, 3, 2, DEPTH, WIDTH)

    print()
    print("=== 2a: ARITHMETIC gap (A float32 sgemm  vs  B float64, same inputs) ===")
    print("  depth   unbiased RMS bias    naive RMS      MC noise tau     mean bias")
    for li in range(DEPTH):
        sq_s, se_s, tau_s, nv_s, mb_s = [], [], [], [], []
        for m in range(N_MLPS):
            d1 = means[m, 0, 0, li] - means[m, 1, 0, li]
            d2 = means[m, 0, 1, li] - means[m, 1, 1, li]
            sq, se, tau2, naive = unbiased_sq_bias(d1, d2)
            sq_s.append(sq)
            se_s.append(se)
            tau_s.append(tau2)
            nv_s.append(naive)
            mb_s.append(float(np.mean((d1 + d2) / 2)))
        sq = float(np.mean(sq_s))
        se = float(np.sqrt(np.sum(np.square(se_s))) / N_MLPS)
        if li in (0, 1, 3, 7, 11, 15, 19, 23, 27, 29, 30, 31):
            print(f"  {li + 1:5d}   {rms(sq):+.4e} +/- {rms(se + abs(sq)) - rms(sq) if sq > 0 else 0:.1e}"
                  f"   {np.sqrt(np.mean(nv_s)):.4e}   {np.sqrt(np.mean(tau_s)):.4e}"
                  f"   {np.mean(mb_s):+.3e}")
    print("  (unbiased RMS bias = sqrt(mean_j d1_j*d2_j); negative value printed")
    print("   as negative sqrt means the unbiased squared bias came out below 0,")
    print("   i.e. no systematic difference is resolvable.)")

    li = DEPTH - 1
    print()
    print("=== HEADLINE, layer 32 ===")
    for label, i0, i1 in (("A-B  arithmetic only", 0, 1),
                          ("B-C  input quantisation only", 1, 2),
                          ("A-C  total float32 pipeline gap", 0, 2)):
        sq_s, se_s, tau_s, mb_s = [], [], [], []
        for m in range(N_MLPS):
            d1 = means[m, i0, 0, li] - means[m, i1, 0, li]
            d2 = means[m, i0, 1, li] - means[m, i1, 1, li]
            sq, se, tau2, _ = unbiased_sq_bias(d1, d2)
            sq_s.append(sq)
            se_s.append(se)
            tau_s.append(tau2)
            mb_s.append(float(np.mean((d1 + d2) / 2)))
        sq = float(np.mean(sq_s))
        se = float(np.sqrt(np.sum(np.square(se_s))) / N_MLPS)
        bound = np.sqrt(max(sq, 0.0) + 2 * se)
        print(f"  {label}")
        print(f"     unbiased  mean_j bias^2 = {sq:+.4e}  +/- {se:.2e}")
        print(f"     -> RMS bias per neuron  = {rms(sq):+.4e}   "
              f"(95%-ish upper bound {bound:.3e})")
        print(f"     mean bias over neurons  = {np.mean(mb_s):+.4e}")
        print(f"     resolution (MC tau)     = {np.sqrt(np.mean(tau_s)):.3e}")

    print()
    print("=== verdict inputs ===")
    print(f"  target per-neuron RMS (contract v=0.18)  = {TARGET_RMS_CONTRACT:.3e}")
    print(f"  target per-neuron RMS (measured v=0.055) = {TARGET_RMS_MEASURED:.3e}")

    generator_check()


if __name__ == "__main__":
    main()
