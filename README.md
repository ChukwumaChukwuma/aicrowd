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

**The layer-1 Hermite family is now closed too** (`docs/hermite_rank_ceiling.md`).
`He_d` of a layer-1 pre-activation is a degree-`d` object, so the barrier's
`k ≤ 2` instance never bounded it; the real obstruction is **rank**.
`h_d(⟨a,x⟩)` is exactly a unit *rank-one* symmetric tensor of the degree-`d`
Wiener chaos, so a Hermite dictionary on `m` directions lives in
`Sym^d(span A)` and is capped by `Σ_j Var(E[y_j | Aᵀx])` — at every degree at
once. Measured: the degree spectrum of `relu(z³²)` is
`f = (26.7, 19.0, 11.2, 7.0, …)%` with **mean Hermite degree 10.5**, so 54% of
the variance sits above degree 2 — and the largest dictionary the family can
build (2,494 features, degree ≤ 16, every cross product) reaches
**43.2%**, of which the shipped 512-feature `k ≤ 2` basis already realises
**39.2%**. The argmax of held-out `R²` net of `p/N` over all 71 dictionaries
measured *is* the shipped basis.

The one combination that beat it on paper — swap the 256-feature `k = 1` block
for 24 mean-field-Jacobian directions — was built, the 640-MLP training set
regenerated and the head re-fitted. It delivers **exactly the predicted 3%**
at the level of the raw control variate (1.371× against 1.328× at unit
coefficient) and **+0.002×** inside the fitted head, which was already
insuring against the very `p/N` it removes. On the official suite: raw
3.7039e-6 (0.32% better, machine-independent), adjusted 4.2494e-7 (**0.945×**,
worse). Not shipped.

**And the dictionary-free bound that page left open is a tautology**
(`docs/integrable_cv.md`). "The top-8 eigenfunctions of `Cov(y)` explain 90.1%"
is true and is not a construction: that operator maps every function into
`span{ȳ_j}`, so its eigenfunctions *are* linear combinations of the centred
outputs, and their means are zero only because `E[y]` was subtracted — which is
the answer. Measured, they are 51.4% degree `≤ 2` in `x`, against a 75% bar.

The 90% is nonetheless reachable and free: a **linear** control variate in
`relu(z^L)` reaches `R² = 86%` at `L = 16` and **98.9%** at `L = 32`, on
features the forward pass already computed. What closes it is `E[g]`. `z¹` is
exactly Gaussian, so layer 1 — and hence `E[z²]`, `Cov(z²)` — is exactly
integrable and nothing deeper is; the Gaussian closure's error rises from
`rms 1.6e-3` at `L = 2` to `7.0e-3` at `L = 32` and enters the score as
`0.1 b²`. With the optimal shrinkage `θ* = D/(D+b²)` on the biased correction,
**the entire ladder from `L = 2` to `L = 32` is worth 1.03×.** The
forward-looking number is one scalar: the analytic layer-mean would have to
improve by **4.2–4.6×** to break even, and past that the payoff is 1.9× at
`r = 4.4`, 5.4× at `r = 10`, 25× at `r = 30`. So the control-variate question
reduces to the closure-accuracy question — the basis is not the lever.

What survives is exact and small: `relu(z¹)` in place of `He₂` — 256 features
instead of 512, exact mean `σ_i/√(2π)`, **1.014×** on unbiased true MSE over 48
generated MLPs against a 1.020× prediction. The one dictionary that looked like
more, degree 2 in the layer-1 *activations* (`E[relu·relu]` is the arc-cosine
kernel, so its mean is exact and `z²` is already computed), is **1.10× jointly
fitted and ≤ 1.000× at every deployable block weight**: more than half of its
span is already inside the shipped blocks, so a separately-optimal correction
re-removes signal while adding its own heavier noise, and the cross-block Gram
that would separate them costs 44% of `N`. Same shape as `cva`, a different
reason — hence the standing rule this round adds: **quote every new dictionary
twice, jointly fitted and at deployable block weights on unbiased true MSE.**

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
