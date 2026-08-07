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
   `C = 2.72e10` FLOPs is *free*. Above it the adjusted score is flat in the
   sample count, so the operating point is exactly at the clamp and accuracy
   per FLOP is the only lever.
2. **The reference is itself a Monte-Carlo mean at N = 1e9.** Its own sampling
   variance is `v/N`, so a *perfect* estimator still measures that. Measured
   on the official suite `v = 0.0495`, giving a raw floor of **4.95e-11** and
   an adjusted floor of **4.95e-12** (`docs/floor_theorem.md`; the challenge
   docs' 0.18 is the depth-8 warm-up value and is wrong by 3.3× for this
   shape).

Nothing reaches that floor: it needs a 154,000× variance reduction over plain
sampling, and the public leaderboard leader is at 73× the floor. What this
repository does reach is documented in `docs/state_of_play.md`, and the current
estimator is **sparse Monte Carlo with layer-1 Hermite control variates and a
15-feature offline-trained residual head** (`docs/learned_corrector.md`,
trimmed in `docs/stein_cv.md`): raw **3.7157e-6**, adjusted **3.95e-7**,
`F/B` **0.0919**, 0 raises in 100.

The one surrogate family that provably escapes the low-order barrier — Stein
control variates driven by the network's own gradient — was built, verified
exactly unbiased, and measured at `R² = 11.5%` against a pre-registered bar of
0.75 (`docs/stein_cv.md`). It is not shipped.

## Layout

```
submission/estimator.py   the graded algorithm; imports only flopscope + whestbench
submission/corrector.npz  the offline-trained head; loaded at 0 FLOPs in setup
whestfloor/               research library
  contract.py             FROZEN constants, types, scoring replica  (single owner)
  kernels.py              estimator variants, flopscope-only — same code that ships
  corrector.py            control-variate features, ridge/MLP heads, numpy generator
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
