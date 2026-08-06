# whest-floor — ARC White-Box Estimation Challenge 2026

A mechanistic estimator for the per-neuron mean activation of a random ReLU MLP
under `N(0, I)` input, targeting the benchmark's **noise floor** rather than the
leaderboard.

## The target

The leaderboard ranks on

```
adjusted_final_layer_score = mean_m [ final_layer_mse_m  ×  max(0.1, C_m / B) ]
C_m = F_m + λ·R_m        B = 2.72e11 FLOPs        λ = 1e11 FLOPs/s
```

Two consequences set the whole design:

1. **The multiplier clamps at 0.1 below 10% budget use.** Anything under
   `C = 2.72e10` FLOPs is *free*. The strongest bundled baseline (full
   covariance propagation) spends 3.2e9 — 12% of that. There is roughly 8×
   more compute available at zero score cost, and accuracy is the only lever
   that remains.
2. **The reference is itself a Monte-Carlo mean at N = 1e9.** Its own sampling
   variance is `v/N ≈ 0.18/1e9 ≈ 1.8e-10`, so a *perfect* estimator still
   measures that. Raw MSE cannot go below ≈1.8e-10; the adjusted score cannot
   go below ≈**1.8e-11**. That is the floor this repository aims at, and it is
   derived independently in `scripts/01_derive_noise_floor.py`.

Sampling cannot get there. Reaching `1.8e-10` by Monte Carlo needs ~9e8 samples
against a budget of ~64,000 — four orders of magnitude short, and no variance
reduction closes that. The target therefore *mandates* reading the weights and
computing the expectation.

## Layout

```
submission/estimator.py   the graded algorithm; imports only flopscope + whestbench
whestfloor/               research library
  contract.py             FROZEN constants, types, scoring replica  (single owner)
  kernels.py              estimator variants, flopscope-only — same code that ships
  relu_moments.py         exact rectified-Gaussian moments and Hermite coefficients
  mc.py                   raw-NumPy Monte Carlo (ground truth); never shipped
  suite.py                evaluation suites: seeds + two independent GT halves
  harness.py              scoring, including the unbiased true-MSE estimator
  ledger.py               append-only experiment ledger, schema-enforced
scripts/                  every artifact has exactly one committed producer
ledger/experiments.jsonl  every number ever reported, with its acceptance bar
docs/                     write-ups
```

## Measuring near the floor

A reference built from `N` samples carries per-neuron noise `τ² = v/N`. Naively
comparing against it biases every MSE upward by `τ²` — and near the floor that
bias *is* the signal. So each suite stores **two independent ground-truth
halves** `a` and `b`, and the harness reports

```
unbiased_true_mse = mean_i (p_i − a_i)(p_i − b_i)
```

which is unbiased for the true MSE because the two noise terms are independent,
together with its standard error. Every claim in the ledger carries that
standard error; a claim at the floor without one is not a claim.

## Reproducing the environment

This work was done in a sandbox where **only `github.com` is reachable** — PyPI,
conda and the Ubuntu archive all return 403, and there was no NumPy on the box.
`scripts/00_bootstrap_env.sh` rebuilds the entire numeric stack from Git
sources (OpenBLAS → NumPy → flopscope). It is committed so the results here are
reproducible, not because a normal user needs it.

## License

MIT — see `LICENSE`.
