#!/usr/bin/env python
"""ADVERSARIAL CHECK 3 -- the two-half unbiased estimator and its stated SE.

harness.evaluate reports

    unbiased_true_mse = mean_i (p_i - a_i)(p_i - b_i)
    unbiased_true_mse_stderr from   Var = (1/n^2) sum_i [2 d_i^2 tau^2 + tau^4]

Exact theory.  Write a_i = mu_i + e1_i, b_i = mu_i + e2_i with e1 |= e2 (two
independent reference halves), d_i = p_i - mu_i, and let

    C = Cov(e1) = Sigma / n_half           (Sigma = per-sample activation cov)

Then, using ONLY independence of the halves and second moments (no Gaussianity),

    E[T]   = mean_i d_i^2                                       (unbiased, ok)
    Var(T) = (1/n^2) * sum_{i,j} [ 2 d_i d_j C_ij + C_ij^2 ]
           = (1/n^2) * [ 2 d^T C d + ||C||_F^2 ]

The harness keeps only the i == j terms:

    Var_harness = (1/n^2) * [ 2 sum_i d_i^2 C_ii + sum_i C_ii^2 ]

i.e. it assumes the reference noise is INDEPENDENT ACROSS NEURONS.  It is not:
all neurons in a layer share the input, so Sigma has large off-diagonals.  The
size of the error is  ||C||_F^2 / sum_i C_ii^2, measured below.

Also tested: the `d2 = np.maximum(da*db, 0.0)` clamp (harness.py:132), the
behaviour at and below tau^2, and the real harness.evaluate code path.

Run:
    . <prefix>/runenv.sh && OPENBLAS_NUM_THREADS=1 $PY scripts/02_adv_unbiased.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whestfloor.harness import evaluate  # noqa: E402
from whestfloor.mc import layer_means, make_mlp  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402

WIDTH = 256
DEPTH = 32
N_HALF = 500_000_000  # the regime the project targets (1e9 total reference)
REPS = 20_000


# --------------------------------------------------------------------------
def measure_sigma(mlp_seed: int = 0, n: int = 16384, chunk: int = 1024):
    """Per-sample covariance of the final-layer activations of one real MLP."""
    W = make_mlp(WIDTH, DEPTH, mlp_seed)
    rng = np.random.default_rng(4242 + mlp_seed)
    acc = []
    done = 0
    while done < n:
        x = rng.standard_normal((chunk, WIDTH), dtype=np.float32)
        for w in W:
            x = np.maximum(x @ w, np.float32(0.0))
        acc.append(x.astype(np.float64))
        done += chunk
    A = np.concatenate(acc)
    mu = A.mean(axis=0)
    S = np.cov(A, rowvar=False, bias=True)
    return mu, S, A


def harness_var(d_a, d_b, tau2_vec, n):
    """Byte-for-byte the harness.py:130-134 computation."""
    d2 = np.maximum(d_a * d_b, 0.0)
    return float(np.sum(2.0 * d2 * tau2_vec + tau2_vec * tau2_vec) / (n * n))


def exact_var(delta, C, n):
    return float((2.0 * delta @ C @ delta + np.sum(C * C)) / (n * n))


def diag_var(delta, C, n):
    c = np.diag(C)
    return float((2.0 * np.sum(delta * delta * c) + np.sum(c * c)) / (n * n))


# --------------------------------------------------------------------------
def section_ab():
    print("=== measuring a real final-layer covariance (width 256, depth 32) ===")
    mu, S, A = measure_sigma(0)
    n = WIDTH
    dead = int((np.diag(S) == 0).sum())
    frob = float(np.sum(S * S))
    diag2 = float(np.sum(np.diag(S) ** 2))
    print(f"  dead neurons (zero variance)        : {dead}/{WIDTH}")
    print(f"  mean_j Var(a_j)                     : {np.diag(S).mean():.5f}")
    print(f"  ||Sigma||_F^2 / sum_j Sigma_jj^2    : {frob / diag2:.2f}   "
          "<-- variance inflation of the tau^4 term")
    print(f"  => SE understated by up to          : {np.sqrt(frob / diag2):.2f}x "
          "in the delta -> 0 regime")

    C = S / N_HALF
    tau2_vec = np.diag(C)
    tau2 = float(tau2_vec.mean())
    print(f"  tau^2 = v/n_half at n_half={N_HALF:.0e}      : {tau2:.4e}")

    # Cholesky-free sampler for e ~ N(0, C)
    evals, evecs = np.linalg.eigh(S)
    keep = evals > 1e-14 * evals.max()
    L = evecs[:, keep] * np.sqrt(evals[keep] / N_HALF)
    k = int(keep.sum())
    print(f"  numerical rank of Sigma             : {k}")

    rng = np.random.default_rng(20260806)
    print()
    print("=== replication study: is T unbiased, and is the stated SE right? ===")
    print("  20,000 replications per regime, exact covariance model")
    print()
    hdr = ("  regime           E[T]        true mean_i d^2   sd(T) empirical"
           "   harness SE   exact SE   emp/harness")
    print(hdr)
    for label, scale in (("delta = 0", 0.0),
                         ("d^2 = 0.1 tau^2", 0.1),
                         ("d^2 = 1.0 tau^2", 1.0),
                         ("d^2 = 10 tau^2", 10.0),
                         ("d^2 = 100 tau^2", 100.0)):
        # a realistic error shape: correlated across neurons like the signal is
        base = rng.standard_normal(WIDTH)
        base += 1.0  # common component: estimator errors are structured, not iid
        base[np.diag(S) == 0] = 0.0  # dead neurons are predicted exactly
        if scale == 0.0:
            delta = np.zeros(WIDTH)
        else:
            delta = base * np.sqrt(scale * tau2 / np.mean(base ** 2))
        true_mse = float(np.mean(delta ** 2))

        e1 = rng.standard_normal((REPS, k)) @ L.T
        e2 = rng.standard_normal((REPS, k)) @ L.T
        da = delta[None, :] - e1
        db = delta[None, :] - e2
        T = (da * db).mean(axis=1)

        hv = np.array([harness_var(da[r], db[r], tau2_vec, n) for r in range(0, REPS, 20)])
        harness_se = float(np.sqrt(hv).mean())
        ex_se = float(np.sqrt(exact_var(delta, C, n)))
        emp = float(T.std(ddof=1))
        print(f"  {label:16s} {T.mean():+.3e}   {true_mse:.3e}      "
              f"{emp:.3e}    {harness_se:.3e}  {ex_se:.3e}   "
              f"{emp / harness_se:6.2f}x")

        if scale == 0.0:
            print(f"     P(T < 0) = {float((T < 0).mean()):.3f}; "
                  f"bias of E[T] vs 0 = {T.mean():+.2e} "
                  f"(MC SE {emp / np.sqrt(REPS):.1e})")
            print(f"     diag-only SE (no clamp) = {np.sqrt(diag_var(delta, C, n)):.3e}; "
                  f"clamp inflates it to {harness_se:.3e} "
                  f"({harness_se / np.sqrt(diag_var(delta, C, n)):.2f}x)")
    return S, C, tau2_vec


# --------------------------------------------------------------------------
def section_c(S, C, tau2_vec):
    """Same test, but through the REAL harness.evaluate code path."""
    print()
    print("=== running the real harness.evaluate over replications ===")
    w, d = WIDTH, 2  # full width (that is where the correlation lives); depth 2
    rng = np.random.default_rng(7)                     # keeps make_mlp cheap
    # The measured covariance of a real final layer, unmodified.
    Sw = S[:w, :w] + 1e-12 * np.eye(w)
    n_half = 500_000_000
    Cw = Sw / n_half
    evals, evecs = np.linalg.eigh(Sw)
    keep = evals > 1e-14 * evals.max()
    L = evecs[:, keep] * np.sqrt(evals[keep] / n_half)
    k = int(keep.sum())

    mu = np.zeros((d, w))
    mu[-1] = np.abs(rng.standard_normal(w)) + 0.5
    delta = np.zeros(w)  # the "perfect estimator" regime we care about
    pred = mu.copy()
    pred[-1] = mu[-1] + delta

    reps = 2000
    Ts, SEs = [], []
    for r in range(reps):
        a = mu.copy()
        b = mu.copy()
        a[-1] = mu[-1] + rng.standard_normal(k) @ L.T
        b[-1] = mu[-1] + rng.standard_normal(k) @ L.T
        s = Suite(name="synthetic", width=w, depth=d, mlp_seeds=[0],
                  gt_a=a[None], gt_b=b[None], n_per_half=n_half,
                  gt_seed_a=[0], gt_seed_b=[1],
                  final_var=np.diag(Sw)[None])
        rep = evaluate(lambda ws, xp, p=pred: p, s, name="const",
                       flops_used=0, bill_first_mlp=False)
        Ts.append(rep.unbiased_true_mse)
        SEs.append(rep.unbiased_true_mse_stderr)
    Ts = np.asarray(Ts)
    SEs = np.asarray(SEs)
    emp = float(Ts.std(ddof=1))
    ex = float(np.sqrt(exact_var(delta, Cw, w)))
    print(f"  width {w}, 1 MLP, delta = 0, n_half = {n_half:.0e}, {reps} replications")
    print(f"  empirical sd(unbiased_true_mse)   = {emp:.4e}")
    print(f"  harness unbiased_true_mse_stderr  = {SEs.mean():.4e}  (mean over reps)")
    print(f"  exact SE incl. cross-neuron terms = {ex:.4e}")
    print(f"  UNDERSTATEMENT FACTOR             = {emp / SEs.mean():.2f}x")
    print(f"  mean reported unbiased_true_mse   = {Ts.mean():+.3e}  "
          f"(truth 0, MC SE {emp / np.sqrt(reps):.1e})  -> unbiasedness OK")
    print(f"  fraction of runs reporting T < 0  = {float((Ts < 0).mean()):.3f}")
    inside = float((np.abs(Ts) <= 2 * SEs).mean())
    inside_ex = float((np.abs(Ts) <= 2 * ex).mean())
    print(f"  coverage of the +/-2*SE interval  = {inside:.3f}  "
          f"(nominal ~0.95)   with exact SE: {inside_ex:.3f}")


# --------------------------------------------------------------------------
def section_d():
    """Are the two ground-truth halves (consecutive integer seeds) independent?"""
    print()
    print("=== are gt_seed_a = base+2k and gt_seed_b = base+2k+1 independent? ===")
    w, dep, n, K = 64, 8, 8192, 192
    W = make_mlp(w, dep, 0)
    A = np.empty((K, w))
    B = np.empty((K, w))
    for kk in range(K):
        ma, _ = layer_means(W, n, 1_000_000 + 2 * kk, want_var=False, all_layers=False)
        mb, _ = layer_means(W, n, 1_000_000 + 2 * kk + 1, want_var=False,
                            all_layers=False)
        A[kk] = ma[-1]
        B[kk] = mb[-1]
    mu = 0.5 * (A.mean(axis=0) + B.mean(axis=0))
    ea = A - mu
    eb = B - mu
    alive = ea.std(axis=0) > 0
    ea = ea[:, alive]
    eb = eb[:, alive]

    def corr(i):
        num = float((ea[i] * eb[i]).mean())
        den = float(np.sqrt((ea[i] ** 2).mean() * (eb[i] ** 2).mean()))
        return num / den

    r = corr(np.arange(K))
    # Neurons are correlated, so the independent unit is the SEED PAIR.
    # Bootstrap over seed pairs -- do NOT use 1/sqrt(K*n_neurons).
    brng = np.random.default_rng(5)
    boot = np.array([corr(brng.integers(0, K, K)) for _ in range(2000)])
    se = float(boot.std(ddof=1))
    bias = -1.0 / (2 * K)  # induced by estimating mu from the same 2K half-means
    print(f"  width {w} depth {dep}, {K} seed pairs x {n} samples, "
          f"{int(alive.sum())} live neurons")
    print(f"  corr(e_a, e_b) = {r:+.4f}  bootstrap SE over seed pairs = {se:.4f}")
    print(f"  (naive iid-neuron SE would be {1 / np.sqrt(ea.size):.4f} -- "
          f"{se * np.sqrt(ea.size):.1f}x too small, the same mistake as harness.py:134)")
    print(f"  expected bias from estimating mu = {bias:+.4f}; "
          f"corrected rho = {r - bias:+.4f} +/- {se:.4f}")
    print(f"  -> residual correlation rho biases T by rho*tau^2; "
          f"here |rho| < {abs(r - bias) + 2 * se:.3f} at 2 sigma")


def main():
    S, C, tau2_vec = section_ab()
    section_c(S, C, tau2_vec)
    section_d()


if __name__ == "__main__":
    main()
