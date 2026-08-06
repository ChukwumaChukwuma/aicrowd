#!/usr/bin/env python
"""ADVERSARIAL CHECK 1 -- is PUBLISHED_AVG_VARIANCE = 0.18 right?

contract.PUBLISHED_AVG_VARIANCE = 0.18 sets RAW_MSE_FLOOR = 1.8e-10 and hence
every "we are at the floor" claim in this repo.  Its cited provenance
(docs/concepts/ground-truth.md) does not exist in the repo, and it appears
nowhere in the whestbench sources or docs.  So it is measured here two
independent ways.

Definition being measured (whestbench simulation.py:141-143):

    avg_variance = mean_j Var_x[ a_{depth,j}(x) ]        x ~ N(0, I_width)

evaluated per MLP, then averaged over MLPs the way the leaderboard averages
per-MLP MSEs (arithmetic mean over the suite).

ROUTE A -- direct Monte Carlo, many MLPs, moderate n.
ROUTE B -- mean-field order-parameter recursion.  Write
    q_l = E_x[ ||a_l(x)||^2 ] / n           (length)
    c_l = E_{x,x'}[ a_l(x).a_l(x') ] / (n q_l)   (two-input overlap)
  then, exactly,
    mean_j Var_x[a_j] = mean_j E[a_j^2] - mean_j (E[a_j])^2
                      = q_L - E[a(x).a(x')]/n = q_L (1 - c_L).
  In the infinite-width limit q_l == 1 for all l>=1 and c_l obeys the
  arc-cosine (ReLU) correlation map
    c_{l+1} = (1/pi) ( sqrt(1-c^2) + c (pi - arccos c) ),  c_0 = 0,
  whose approach to the c=1 fixed point is  1-c_l ~ 4/(k^2 l^2),
  k = 2*sqrt(2)/(3*pi).  So mean-field predicts avg_variance = 1 - c_depth.

ROUTE C -- width scan at fixed depth 32, to show the finite-width inflation of
  route B and to confirm the two routes are measuring the same object.

Run:
    . <prefix>/runenv.sh && OPENBLAS_NUM_THREADS=1 $PY scripts/02_adv_variance.py
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

from whestfloor.mc import layer_means, make_mlp  # noqa: E402

WIDTH = 256
DEPTH = 32
N_MLPS = int(os.environ.get("ADV_N_MLPS", 128))
N_SAMPLES = int(os.environ.get("ADV_N_SAMPLES", 8192))
GT_N = 1_000_000_000
PUBLISHED = 0.18


# --------------------------------------------------------------------------
# ROUTE A
# --------------------------------------------------------------------------
def route_a() -> np.ndarray:
    print(f"ROUTE A: direct MC, {N_MLPS} MLPs x {N_SAMPLES} samples, "
          f"width {WIDTH} depth {DEPTH}")
    vs = np.empty(N_MLPS)
    qs = np.empty(N_MLPS)
    t0 = time.perf_counter()
    for m in range(N_MLPS):
        w = make_mlp(WIDTH, DEPTH, m)
        mean, var = layer_means(w, N_SAMPLES, 900_000 + m, want_var=True,
                                all_layers=False)
        vs[m] = var.mean()
        qs[m] = (var + mean[-1] ** 2).mean()  # mean_j E[a_j^2]
        del w
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    return vs, qs


def bootstrap_ci(x: np.ndarray, reps: int = 20000, seed: int = 1234):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------
# ROUTE B
# --------------------------------------------------------------------------
def relu_corr_map(c: float) -> float:
    c = min(1.0, max(-1.0, c))
    return (np.sqrt(1.0 - c * c) + c * (np.pi - np.arccos(c))) / np.pi


def route_b(depth: int = DEPTH) -> float:
    c = 0.0  # two independent N(0,I) inputs are orthogonal in the width->inf limit
    traj = []
    for _ in range(depth):
        c = relu_corr_map(c)
        traj.append(c)
    k = 2.0 * np.sqrt(2.0) / (3.0 * np.pi)
    asym = 4.0 / (k * k * depth * depth)
    return traj, asym


# --------------------------------------------------------------------------
# ROUTE C
# --------------------------------------------------------------------------
def route_c():
    out = []
    for width in (64, 128, 256, 512):
        n_mlps = 24
        n_s = 4096
        vv = []
        for m in range(n_mlps):
            w = make_mlp(width, DEPTH, 5_000 + m)
            mean, var = layer_means(w, n_s, 950_000 + m, want_var=True,
                                    all_layers=False)
            vv.append((var.mean(), (var + mean[-1] ** 2).mean()))
            del w
        vv = np.asarray(vv)
        out.append((width, vv[:, 0].mean(), vv[:, 1].mean()))
    return out


def route_d():
    """Where did 0.18 come from?  Scan depth at fixed width 256.

    whest-starterkit docs/concepts/ground-truth.md:29 says "Against the official
    datasets (N = 1e9 samples) the ground-truth noise floor is ~2e-10
    (= avg_variance / N)".  Every dataset example in the whestbench docs
    (dataset-format.md:382,401,565; gpu-dataset-generation.md) bakes at
    --width 256 --depth 8.  The competition shape is depth 32.
    """
    out = []
    for depth in (4, 8, 16, 32):
        vv = []
        for m in range(48):
            w = make_mlp(WIDTH, depth, m)
            _, var = layer_means(w, 4096, 880_000 + m, want_var=True,
                                 all_layers=False)
            vv.append(var.mean())
            del w
        vv = np.asarray(vv)
        out.append((depth, vv.mean(), vv.std(ddof=1) / np.sqrt(len(vv))))
    return out


def main() -> None:
    vs, qs = route_a()
    v_mean = float(vs.mean())
    lo, hi = bootstrap_ci(vs)
    se = float(vs.std(ddof=1) / np.sqrt(len(vs)))

    print()
    print("=== ROUTE A: measured avg_variance, width 256 depth 32 ===")
    print(f"  mean over {N_MLPS} MLPs : {v_mean:.5f}  +/- {se:.5f} (SE)")
    print(f"  bootstrap 95% CI       : [{lo:.5f}, {hi:.5f}]")
    print(f"  median                 : {float(np.median(vs)):.5f}")
    print(f"  min / max              : {vs.min():.5f} / {vs.max():.5f}")
    print(f"  sd across MLPs         : {vs.std(ddof=1):.5f}  "
          f"(CV = {vs.std(ddof=1) / v_mean:.2f})")
    print(f"  mean_j E[a^2] (q_L)    : {qs.mean():.5f}   "
          f"=> implied 1-c = {v_mean / qs.mean():.5f}")
    print(f"  PUBLISHED_AVG_VARIANCE : {PUBLISHED}")
    print(f"  ratio measured/published: {v_mean / PUBLISHED:.3f}")
    print(f"  published inside CI?    : {lo <= PUBLISHED <= hi}")

    print()
    print("=== consequence for the noise floor ===")
    print(f"  contract RAW_MSE_FLOOR      = {PUBLISHED / GT_N:.4e}")
    print(f"  measured  RAW_MSE_FLOOR     = {v_mean / GT_N:.4e}  "
          f"[{lo / GT_N:.3e}, {hi / GT_N:.3e}]")
    print(f"  contract ADJUSTED_FLOOR     = {PUBLISHED / GT_N * 0.1:.4e}")
    print(f"  measured ADJUSTED_FLOOR     = {v_mean / GT_N * 0.1:.4e}")
    print(f"  per-neuron target RMS err   = {np.sqrt(v_mean / GT_N):.4e}  "
          f"(contract assumed {np.sqrt(PUBLISHED / GT_N):.4e})")

    traj, asym = route_b()
    v_mf = 1.0 - traj[-1]
    print()
    print("=== ROUTE B: infinite-width mean-field ===")
    print("  1-c_l for l = 1,2,4,8,16,32:")
    for l in (1, 2, 4, 8, 16, 32):
        print(f"    l={l:3d}   1-c = {1 - traj[l - 1]:.6f}")
    print(f"  mean-field avg_variance at depth 32 = q*(1-c) = {v_mf:.5f}")
    print(f"  asymptotic 4/(k^2 l^2) prediction    = {asym:.5f}")
    print(f"  measured / mean-field inflation      = {v_mean / v_mf:.3f}x  "
          "(finite-width, width=256)")

    print()
    print("=== ROUTE C: width scan at depth 32 (24 MLPs x 4096 samples each) ===")
    print("   width   avg_variance   q_L      1-c     mean-field 1-c")
    for width, v, q in route_c():
        print(f"   {width:5d}   {v:.5f}      {q:.4f}   {v / q:.5f}   {v_mf:.5f}")

    print()
    print("=== ROUTE D: provenance -- 0.18 is the DEPTH-8 value ===")
    print("   width 256, 48 MLPs x 4096 samples per depth")
    print("   depth    avg_variance      floor v/1e9     matches 'published' 0.18?")
    for depth, v, se in route_d():
        print(f"   {depth:5d}    {v:.4f} +/- {se:.4f}   {v / GT_N:.3e}      "
              f"{'YES <-- this is where 0.18 comes from' if abs(v - 0.18) < 3 * se else 'no'}")
    print("   whest-starterkit docs/concepts/ground-truth.md:29 quotes ~2e-10;")
    print("   every dataset-bake example in the whestbench docs uses --depth 8.")
    print("   The competition shape is depth 32.")

    print()
    verdict = "HOLDS" if lo <= PUBLISHED <= hi else "BROKEN"
    print(f"VERDICT Claim 1: {verdict}  "
          f"(measured v = {v_mean:.4f}, published 0.18)")


if __name__ == "__main__":
    main()
