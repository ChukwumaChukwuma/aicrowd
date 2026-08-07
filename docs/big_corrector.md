# Scaling the offline-trained corrector: how much of the residual is learnable

*The shipped corrector is a **15-float** ridge head fitted on 640 MLPs. This
page asks what it does with orders of magnitude more features, parameters and
training networks — and answers the question nobody has published: **the
held-out learning curve of an offline-trained residual corrector against
training-set size and parameter count.***

Everything here is reproducible from `scripts/45_big_corrector.py`
(`--mode refcal|data|fit|curve|noise|cost`); machinery in
`whestfloor/bigcorr.py`. `whestfloor/corrector.py`, `scripts/28` and
`submission/estimator.py` are untouched.

---

## 0. What this is not

The brief that launched this work inferred from leaderboard telemetry that rank
1 "spends nothing at runtime, therefore the answer is precomputed". **That
inference was withdrawn before any result on this page was taken**, and nothing
here rests on it. The corrected arithmetic is

| entry | instrumented share | `C/B` | implied `F` | dense forward passes |
|---|---|---|---|---|
| dpskv5 (rank 1) | 0.940 | 0.095 | 2.43e10 | 5,785 |
| huang_chung_yi (rank 2) | 0.772 | 0.030 | 6.30e09 | 1,500 |
| this repo | — | — | 6.22e10 | 14,816 |

so the leaders **do** compute at grade time, at 0.39x and 0.10x our FLOPs, for
299x and 76x better raw MSE. They are deterministic methods with small model
error, not lookups. `docs/traj_closure.md` closes the cumulant route to them
(deployable `r = 2.0`, exact-cumulant ceiling `r ~ 6`, against the ~2,500x that
would be needed).

What survives, and is the premise of this page, is narrower and is measured
rather than inferred: **an offline-trained head is free at grade time**
(`fnp.load` bills 0 FLOPs — remeasured in §1), the submission may carry 50 MiB,
and the shipped 15 floats already buy 1.5x. Nobody has measured where that
curve saturates.

## 1. The two costs that decide whether "large" is even allowed

`--mode cost`, real `flopscope.BudgetContext`, `B = 2.72e11`.

| object | billed FLOPs | share of `B` |
|---|---|---|
| `cv1mf` mean propagation (what ships) | 4,062,976 | 0.0015% |
| **`mfm` + `mfv` two-channel transport (new)** | **8,141,312** | **0.0030%** |
| the `W .^ 2` tables, once per MLP | 2,097,152 | 0.0008% |
| random-feature head, 256 features (6,400 params) | 4,409,088 | 0.0016% |
| random-feature head, 2,048 features (51,200 params) | 35,145,472 | 0.0129% |
| random-feature head, 16,384 features (409,600 params) | 281,036,544 | **0.1033%** |

| `fnp.load` | FLOPs | wall | share of the 5 s setup window |
|---|---|---|---|
| 1 MiB | **0** | 2 ms | 0.0% |
| 8 MiB | **0** | 17 ms | 0.3% |
| 32 MiB | **0** | 64 ms | 1.3% |

Both premises hold at size. A 400,000-parameter head costs **0.10% of the FLOP
budget** to evaluate once per MLP, and 32 MiB of weights load at **zero FLOPs
in 64 ms**, i.e. 1.3% of the setup window — so the 50 MiB / 50 file cap, not
the runtime, is the binding constraint on model size.

**The constraint that actually binds is neither.** It is the offline compute to
*train* on this box: closed-form ridge costs `n_rows x n_out^2` (3.4e12 FLOPs
at 2,048 random features and 800,000 rows) and a gradient epoch costs
`6 n_rows n_params`. At the ~20 GFLOP/s a contended core delivers here, that
puts the trainable ceiling near **1e5 parameters**, not 1.3e7. The learning
curve below is therefore reported against parameters up to that ceiling, and
the question "does it saturate before it?" is the one it can answer.

## 2. The reference is the binding constraint, so it was calibrated first

`--mode refcal`, 3 MLPs x 4 repetitions, against an 800,000-sample yardstick.
`bigcorr.reference` is a dense float64-accumulated pass with the input-linear
(`k = 1`) control variate applied in **split-sample** form — exactly mean zero,
so the reference stays unbiased, at 3% extra FLOPs.

| `n_gt` per half | rms, no CV | rms, + k=1 CV | variance ratio | effective `n_gt` |
|---|---|---|---|---|
| 16,384 | 1.4816e-03 | 1.3885e-03 | 1.139 | 18,655 |
| 32,768 | 1.0117e-03 | 8.7634e-04 | 1.333 | 43,674 |
| 65,536 | 7.8930e-04 | 6.2704e-04 | **1.585** | 103,843 |

The control variate is worth more at larger `n_gt` because its own `p/n`
estimation noise (256 directions against `n/2` samples per split half) shrinks
faster than the signal. At the chosen operating point, `n_gt = 32,768` per
half, it is a free **1.33x** on effective reference samples.

**Chosen: `n_gt = 2 x 32,768` with the CV on.** Label noise is then 8.76e-04
per half, 6.2e-04 on the two-half mean, against a target whose rms is 1.38e-03
— so 20% of the target variance is reference noise. That is deliberate: the
noise is zero-mean and independent of every feature, so it inflates the
variance of the fitted coefficients and nothing else, and §5 measures the
MLPs-versus-precision trade directly rather than assuming it.

## 3. What is generated, and the seed discipline

`--mode data`, at the **deployed** operating point `(tau, N, P) = (2.5, 25000,
225)` — not the `(2.5, 8500, 150)` the shipped head was fitted at.

* MLP seeds `400000+`, disjoint from `scripts/28`'s `100000+` block, from every
  suite in the repository and from the official seeds.
* Per MLP: **one** reference (two independent halves) and **four** independent
  estimator seeds. The reference depends only on the weights and costs about as
  much as four scored passes, so replication buys 4x the rows for 1.5x the
  compute. Rows of one MLP share a reference-noise draw — which matters for the
  seed-independent columns and not at all for the correction columns.
* Train / validation / test are split **by MLP seed**, 60/20/20, so no network
  appears in two splits. Every number quoted is validation (for selection) or
  test (read once). Prizes are decided on freshly generated private networks;
  an in-sample number would be worthless.

## 4. The feature block

Fourteen correction channels x three modulators, plus shape columns, plus the
pooled terms of §4.3.

### 4.1 What ships now

`mu` (the sparse-MC estimate), the layer-1 Hermite control variates `cv1`,
`cv2`, the mean-field propagated layer-1 mean gap `cv1mf`, the final-layer
shape `(s, Phi, phi, alpha)` and `dpilot`.

### 4.2 New, and essentially free

**`mfm`, `mfv`, `mfv2` — a two-channel first-order transport.** `z^1` is
exactly Gaussian, so both

    dmu1_i = mean_s relu(z^1_i) - sigma_i / sqrt(2 pi)
    dv1_i  = mean_s relu(z^1_i)^2 - sigma_i^2 / 2

are exactly mean zero and already in hand. `cv1mf` transports the first and
drops the second. Both are transported here through the exact chain rule of the
Gaussian rectifier moments,

    dmu_l = Phi dm + phi ds ,   dvh_l = 2 mu0 (1-Phi) dm + 2 (s Phi - mu0 phi) ds
    dm_{l+1} = W' dmu_l  (exact) ,   dvz_{l+1} = (W .^ 2)' dvh_l  (diagonal only)

using `dE[relu^2]/dm = 2 E[relu]` and `dE[relu^2]/ds = 2 s Phi`. The diagonal
truncation costs efficiency and **never bias**: the output is a linear
functional of exactly-mean-zero inputs whatever the coefficients are.

`mfv2` removes that truncation at layer 1 entirely. `E[z^2]` and `Cov(z^2)` are
*exact* (arc-cosine kernel), so

    dvz2_i = mean_s (z^2_i - m2_i)^2 - Cov(z^2)_ii

is exactly mean zero **and is the full layer-1 covariance gap already contracted
onto the direction that needs it** — for one elementwise square of an array the
forward pass has already produced.

**`cv1mfg`, `mfmg`, `mfvg`, `mfv2g` — better transport gates.** `Phi(alpha)` is
the transport coefficient and the pilot estimates `alpha` from 225 samples;
4,096 rows of the scored pass estimate it 4x better. The gate is then weakly
correlated with the gap it multiplies, an `O(1/N)` bias against an `O(N^-1/2)`
correction — 0.6% at this `N`, i.e. 4e-5 of the variance.

**`cv3`.** At `N = 8500` each Hermite block cost `p/N = 3.0%` of the residual
in estimation noise against ~2% of gain, so `k <= 2` was the operating point.
At `N = 25000` the same block costs **1.0%**. The arithmetic that closed
`k >= 3` was N-dependent and N has tripled.

**`sd_mc`.** The per-neuron Monte-Carlo noise scale, dropped from the shipped 15
because it measured at exactly 1.000x. A *linear* head cannot use it: the
optimal shrinkage of a control variate is a **ratio** of signal to noise and a
linear head has one coefficient per column. Here it is both an input and the
unit the nonlinear heads work in — every correction column is divided by
`sd_mc` and the prediction multiplied back, which is the parameterisation in
which the map to be learned is a bounded function of bounded inputs.

### 4.3 `u1`, `u2` — the top eigendirections of `Cov(h^32)`, and pooling

The Monte-Carlo error of `mu_hat` has covariance `Cov(h^32)/N` **exactly**, so
the leading eigendirections of `Cov(h^32)` are by construction the directions
in which the residual is largest. Measured here, `lambda_1 = 3.95` against a
trace of 6.4 — **62% of the total residual variance is in one direction.** A
sibling measurement independently finds the deterministic closure's `L = 32`
error concentrated in the same direction (`q_1 = 3.4-7.1`), so both arms want
this feature.

Forming `Cov(h^32)` costs 3.3e9 FLOPs (1.2% of `B`). **Power iteration never
forms it**: eight sweeps of `x @ V` then `x.T @ (x V)` at `k = 2` cost 4.1e8,
0.15% of `B`. Each eigenvector's sign is fixed by `<u, mu> > 0`, a
permutation-invariant rule, so `u_j` is a well-defined per-neuron feature.

This also buys the one thing a pointwise head cannot have at any parameter
count: **a cross-neuron term.** `u1_j <u1, c>` scatters a channel's projection
on the dominant mode back through that mode. It is exactly permutation
equivariant, it costs one inner product per channel, and it is the cheapest
non-pointwise structure available.

## 5. Results

*(see §6 for the pre-registered bars)*

## 6. Bars, fixed before the runs

| # | bar, fixed before the run | measured | verdict |
|---|---|---|---|
| 1 | refitting the shipped 15-column design at the deployed operating point beats the shipped coefficients on held-out TEST by **>= 1.02x** | | |
| 2 | the rich linear design beats the refitted 15 on held-out TEST by **>= 1.05x** | | |
| 3 | a nonlinear head (random-feature or SGD) beats the best linear design on held-out TEST by **>= 1.05x** | | |
| 4 | the whole stack, converted through the score model, clears the current ship's graded **2.4646e-07** | | |
