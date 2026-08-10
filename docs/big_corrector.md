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

## 5. The score model, which is how a variance ratio becomes a graded number

Only ratios transfer — the local harness runs ~1.45x above the grader on
identical seeds and two changes have already measured better locally and graded
worse. So the ratio is measured against the shipped head **in the same run**
and applied to the grader's own calibrated constants.

With `raw = b^2 + v/N` and `C = F0 + cN`, the graded objective
`adjusted = raw x max(0.1, C/B)` expands to
`[b^2 F0 + b^2 c N + v F0/N + v c]/B`, which is convex in `N` with
`d/dN = 0` at `N* = sqrt(v F0 / (b^2 c))`, giving

```
adjusted(N*)  =  ( sqrt(b^2 F0) + sqrt(v c) )^2 / B
```

This closed form was not in the repository and it is worth stating on its own:
the score is the square of a **sum of two independent square roots**, one for
the pilot's bias floor and one for the sampler. Reducing `v` by `r` shrinks
only the second term, so the payoff is sub-linear in `r` and saturates at
`b^2 F0 / B` — currently 1.6e-9 — no matter how good the corrector gets.

Constants from the graded sweeps (`c` and `F0` from `C/B = 0.1001` at
`N = 8500` and `0.2287` at `N = 22000`; `b^2 = 1.2e-7` at `P = 150` from
`raw = 4.846e-7` at `N = 67000`; both rescaled to the shipped `P = 225`).
**Validation of the model, not a fit to it:** it returns adjusted
**2.470e-07** at `N* = 24,160` against the graded **2.4646e-07** at the shipped
`N = 25,000` — the level to 0.2% and the argmin to 3%.

| `v_eff` reduced by | projected adjusted | `N*` | x over the ship |
|---|---|---|---|
| 1.0x | 2.4706e-07 | 24,160 | 1.000 |
| 1.2x | 2.0908e-07 | 22,055 | 1.182 |
| 1.5x | 1.7077e-07 | 19,727 | 1.447 |
| 2.0x | 1.3197e-07 | 17,084 | 1.872 |
| 5.0x | 5.9816e-08 | 10,805 | 4.130 |

## 6. Results

**475 MLPs x 4 estimator seeds = 1,900 rows = 486,400 neurons**, split by MLP
seed into 285 train / 95 validation / 95 test. Everything below is the
**held-out TEST split**, and the shipped head is evaluated in the same run on
the same networks so the ratio is the only thing that has to transfer.

### 6.1 The headline table

| head | parameters | TEST unbiased MSE | x over the raw pass | x over the shipped head |
|---|---|---|---|---|
| uncorrected sparse pass | 0 | 2.0476e-06 | 1.000 | — |
| **shipped 15 floats, as shipped** | 15 | 1.3962e-06 | 1.467 | **1.000** |
| same 15 columns, refitted here | 15 | 1.3306e-06 | 1.539 | 1.049 |
| **rich linear, 83 columns** | 83 | **1.0949e-06** | **1.870** | **1.275** |
| random features, 256 | 8,305 | 1.1246e-06 | 1.821 | 1.242 |
| random features, 1,024 | 32,881 | 1.1427e-06 | 1.792 | 1.222 |
| random features, 4,096 | 131,185 | 1.1457e-06 | 1.787 | 1.219 |
| per-neuron MLP, 64 hidden (boosted) | 7,361 | 1.6512e-06 | 1.240 | 0.846 |
| per-neuron MLP, 256-128 (boosted) | 62,209 | 1.6656e-06 | 1.229 | 0.838 |
| **selected 20-column head (SHIPPED)** | **20** | **1.0771e-06** | **1.901** | **1.296** |

Refitting the shipped design at the deployed operating point is worth 1.049x —
real but small, and *not* for the reason the module docstring first guessed:
the N-independent columns contribute an rms of 2.2e-05 against a target of
1.4e-03, i.e. 0.05% of the MSE. What the refit actually buys is the right
**shrinkage**, because each Hermite block's `p/N` estimation-noise penalty fell
from 3.0% to 1.0% when `N` went from 8,500 to 25,000 and the old coefficients
were shrunk for the old noise.

### 6.2 Capacity is worthless, measured three ways

**Random features, nested.** The linear block of the random-feature head IS the
83-column rich design (divided by `sd_mc`, with the head predicting
`y/sd_mc`, so the two are the same model), the columns are penalised in the
scaled basis, and the loss is the unweighted metric the score uses. Getting all
three right matters: with the loss mis-weighted by `1/sd_mc^2` the same code
reported 1.276x-1.566x, and with the penalty applied in the unscaled basis it
pinned at the grid endpoint. Done correctly, 256 / 1,024 / 4,096 features give
**1.821 / 1.792 / 1.787** — monotonically *worse*, because validation raises the
penalty (3e0 -> 1e1 -> 3e1) until the added features are shrunk away and all
that survives is their contribution to the effective degrees of freedom.

**A trained network, boosted.** A per-neuron numpy MLP fitted by Adam on the
ridge residual reaches **0.84x**. There is no nonlinear structure to find.

**MLP-specific coefficients, which bound every head at once.** Fit the
coefficients separately for each test MLP on reference half `a`, score against
half `b`, and subtract `Var(b)(1 + p/n)` — the reference noise plus the exact
OLS estimation-variance term `p sigma^2 / n`, so there is no in-sample credit
of any kind. Against a pooled fit on the *identical* columns:

| columns fitted per MLP | pooled beta | per-MLP beta | per-MLP / pooled |
|---|---|---|---|
| 13 channels + intercept (14) | 1.876x | 1.566x | **0.835** |
| channels x modulators (40) | 1.878x | 1.761x | **0.937** |
| + shape columns (57) | 1.863x | 1.912x | 1.026 |
| + shape + pooled u1/u2 (83) | 1.870x | 2.189x | 1.170 |

On every design where the per-MLP estimate is well conditioned (`p/n <= 0.16`)
**a head that knows each network's own optimal coefficients is WORSE than one
shared coefficient vector.** The last two rows run at `p/n = 0.22` and `0.32`,
where the subtracted correction is 1.3x the residual being measured and the
estimate is a small difference of large numbers; they are reported but not
relied on. One shared 83-float vector is at or above the per-MLP optimum.

### 6.3 The learning curve

Held-out TEST gain over the uncorrected pass, against training MLPs:

| train MLPs | training rows | shipped 15 design | rich 83 | RF 256 | RF 4,096 |
|---|---|---|---|---|---|
| 8 | 8,192 | 1.543 | 1.736 | 1.652 | 1.616 |
| 15 | 15,360 | 1.535 | 1.712 | 1.680 | 1.634 |
| 30 | 30,720 | 1.538 | 1.750 | 1.698 | 1.650 |
| 60 | 61,440 | 1.536 | 1.847 | 1.825 | 1.788 |
| 120 | 122,880 | 1.569 | 1.900 | 1.862 | 1.834 |
| 240 | 245,760 | 1.558 | **1.915** | 1.855 | 1.838 |

**The shipped 15-column design is saturated at 8 training MLPs.** The 83-column
design is saturated by 120. More capacity is worse at *every* training-set
size, not just asymptotically. Extrapolating the 120 -> 240 step, another
decade of MLPs is worth under 1%.

### 6.4 Label precision is irrelevant — the whole reference budget should buy MLPs

Simulating a reference of `n_gt / f` exactly, by adding independent noise of
the known variance `(f-1) vh_j/(n_gt r)` to each half of the TRAINING labels
only (evaluation labels untouched, paired estimator still unbiased):

| train MLPs | `n_gt` | `n_gt/2` | `n_gt/4` | `n_gt/8` | `n_gt/16` |
|---|---|---|---|---|---|
| 30 | 1.706 | 1.704 | 1.716 | 1.723 | 1.676 |
| 60 | 1.795 | 1.794 | 1.798 | 1.795 | 1.815 |
| 120 | 1.843 | 1.841 | 1.844 | 1.850 | 1.833 |
| 240 | 1.851 | 1.851 | 1.861 | 1.856 | 1.837 |

At `n_gt/16` the label noise is **3x the signal it is a label for** and the
held-out gain is unchanged to within 1%. The pre-registered guess — "label
noise is not fatal, you need many MLPs more than precise labels" — is confirmed
in the strongest possible form: precision is worth *nothing*, and had this been
known first the reference could have been 2,048 samples instead of 32,768.
(It would not have mattered: §6.3 says the MLP axis saturates too.)

### 6.5 Which features carry it

Leave-one-group-out and only-one-group on validation, 83-column design:

| group | without it | it alone |
|---|---|---|
| `cv1` (Hermite k=1) | 1.744x | 1.319x |
| `cv2` (Hermite k=2) | 1.748x | 1.076x |
| `cv3` (Hermite k=3) | 1.758x | 1.010x |
| `cv1mf`/`mfm` (mean transport, pilot gates) | 1.737x | 1.543x |
| `cv1mfg`/`mfmg` (mean transport, scored gates) | 1.748x | 1.539x |
| `mfv`/`mfvg` (variance transport, diagonal) | 1.744x | 1.113x |
| **`mfv2`/`mfv2g` (variance transport, exact `Cov(z^2)`)** | **1.660x** | 1.155x |
| `relu1` | 1.741x | 1.553x |
| `dpilot` | 1.736x | 1.020x |
| pooled `u1`/`u2` cross-neuron terms | 1.746x | 1.630x |
| full design | — | 1.742x |

Three readings.

**The new exact-covariance channel is the one that matters.** Dropping `mfv2`
costs 1.742x -> 1.660x, four times the next-largest loss. It is the layer-1
covariance gap contracted through the exactly-known `Cov(z^2)`, and it costs
one elementwise square of an array the forward pass already produced. Its
diagonal-approximation sibling `mfv` is worth nothing once it is present —
which is the point: the *exact* anchor is what buys the channel, not the
transport.

**`cv3` is still dead.** The `p/N` arithmetic that closed `k >= 3` at
`N = 8500` predicted it should reopen at `N = 25000`, where the cost is 1.0%
instead of 3.0%. It did not: 1.758x without it against 1.742x with. So the
Hermite ladder is bounded by approximation power, not by estimation noise, and
`docs/hermite_rank_ceiling.md`'s rank-one argument is the operative one.

**The pooled cross-neuron terms are worth exactly nothing**, and this is a
clean negative on a well-motivated idea. `lambda_1 = 3.95` of a 6.4 trace is
real — 62% of the residual variance genuinely is in one direction — but a
pointwise head modulated by `Phi` and `alpha` already spans it: 1.746x without
the pooled terms against 1.742x with. Greedy selection rejects them and the
`u1`/`u2`/`lambda` shape columns as well, which removes the 4.09e8-FLOP power
iteration from the shipped block entirely.

### 6.6 What ships, and what it is worth

Greedy forward selection over channels on **validation only**, best prefix,
then the two optional blocks:

```
+ relu1    val 1.4856e-06  1.554x
+ mfv2     val 1.3861e-06  1.666x
+ mfm      val 1.3050e-06  1.769x
+ dpilot   val 1.3005e-06  1.775x
+ mfv2g    val 1.3001e-06  1.776x
+ cv1      val 1.2994e-06  1.777x
  pooled u1/u2 terms   rejected
  eigen shape columns  dropped
```

**32 coefficients, 5,920 bytes.** Held-out TEST: **1.0857e-06**, against
1.3962e-06 for the shipped head on the same 95 networks — **1.286x**. Through
the §5 score model, with the new channels' 2.35e8 billed FLOPs added to `F0`:

```
projected graded  1.9707e-07  at N* = 21,754     1.254x over the shipped 2.4646e-07
```

## 7. Bars, fixed before the runs

| # | bar, fixed before the run | measured | verdict |
|---|---|---|---|
| 1 | refitting the shipped 15-column design at the deployed operating point beats the shipped coefficients on held-out TEST by **>= 1.02x** | **1.049x** | **PASS** |
| 2 | the rich linear design beats the refitted 15 on held-out TEST by **>= 1.05x** | **1.215x** (1.870 / 1.539) | **PASS** |
| 3 | a nonlinear head (random-feature or SGD) beats the best linear design on held-out TEST by **>= 1.05x** | **0.974x** best of six, at 8,305 to 131,185 parameters | **FAIL — ridge ships** |
| 4 | the whole stack, converted through the score model of §5, clears the current ship's graded **2.4646e-07** | **1.9707e-07** projected, 1.254x | **PASS (projected)** |

## 8. What limits it, stated as a bound

The deliverable if this bottomed out was "how accurate an offline-trained
corrector can get as a function of training MLPs and parameters, and what
limits it". It did bottom out, and the answer is unusually clean:

```
gain(n_mlps, n_params)  saturates at  n_mlps ~ 120  and  n_params ~ 83.
```

Neither axis is the constraint. Not data (§6.3: 120 -> 240 MLPs is worth 0.8%,
and §6.4: the labels can be 3x noisier than the signal at no cost). Not
parameters (§6.2: 8,305 -> 131,185 is monotonically worse, and a trained
network is worse still). Not runtime (§1: a 400,000-parameter head is 0.10% of
`B` and 32 MiB of weights load at 0 FLOPs in 64 ms). Not the artifact cap.

**The constraint is the supply of exactly-mean-zero statistics.** Every channel
that carries signal is one whose population mean is known in closed form, and
the network provides exactly two places where anything is: `z^1` is exactly
Gaussian, and `E[z^2]`/`Cov(z^2)` follow exactly from the arc-cosine kernel.
`tests/test_integrable_cv.py` pins layer 3 as not exact at >20 sigma. Every
feature here is a functional of those two layers, the head's job is only to
weight them, and 83 numbers are enough to do that to within measurement noise
of the best any head could do (§6.2).

Getting materially further needs a *new exact anchor*, not a bigger model. The
one concrete candidate this work identifies and did not have time to test: at
layer `l >= 3` the statistic `mean_s (z^l - m^l)^2 - V^l_closure` is
mean-zero only up to the closure's error in `V^l`, so it is a biased control
variate — but the bias enters multiplied by a fitted coefficient the head is
free to shrink, and a held-out fit measures the net effect honestly. The
depth ladder of that trade (`docs/integrable_cv.md` runs the analogous one for
the mean) is the next thing to run, and it is the only remaining direction in
which this family is not already at its ceiling.

## 9. Shipped

`submission/estimator.py` now carries the scaled head. Three channels were
ported into flopscope-only form -- `relu1` (exact-mean control variate on
`relu(z^1)`, arc-cosine Gram, split-sample), `mfv2` (the layer-1 covariance gap
contracted through the exactly-known `Cov(z^2)` and transported forward) and
`mfm` (the same transport started from the exact layer-2 mean anchor, replacing
`cv1mf`) -- and `cv2`, the k=2 Hermite block, is **gone**, because it is not
selected once `mfv2` is present. That removes its Gram, its solve and its pass
over the `(N, width)` array.

### 9.1 What it costs, measured

Width 256, depth 32, `N = 25000`, real `BudgetContext`, same MLP both arms:

| | dispatches | `F` | `F/B` |
|---|---|---|---|
| 15-float head | 563 | 70,897,895,720 | 0.26065 |
| **scaled head** | **852** | **71,302,456,788** | **0.26214** |
| delta | **+289** | **+404,561,068** | **+0.1487%** |

At ~22 us a dispatch the 289 extra calls are 6.4 ms of billed residual, i.e.
0.234% of `B` at `lambda = 1e11`. Total **0.38% of `B`**, which against
`C/B ~ 0.28` is **+1.4% on the multiplier** for a 29.6% accuracy gain.

248 of the 289 are the transport recursion, and they are already batched three
ways: the two channels are the two columns of one `(width, 2)` array so one
pair of matmuls serves both; the `1/(2s)` factor is folded into `phi` and into
the variance gain once for all layers; and every per-layer coefficient is built
in one dispatch on the stacked `(depth, width)` block. That is 8 dispatches a
layer against the 14 two separate scalar recursions would need -- 248 rather
than 868. A further 124 could go by fusing the two 2x2 mixes into one
`einsum('nij,njc->nic')`, worth ~2.7 ms; it is not done, because it changes the
summation order on the shipped path for 0.1% of `B` and this had to ship today.

### 9.2 Packaging

`scripts/35_package.py`: **SHIPPABLE**, 16,932 B tarball.

| check | result |
|---|---|
| setup wall time, both npz loaded | **0.272 s** against the 5.0 s cap |
| `fnp.load` of BOTH files together | **0 FLOPs**, 1.2 ms |
| `predict` | `(32, 256)`, all 8192 finite |
| `F` in the sandbox | 72,068,032,410 = 26.50% of `B` |
| `C` in the sandbox | 8.156e10 = 29.99% of `B` (`R = 0.0949 s`) |
| denied modules | none, on any code path; imports are `__future__`, `flopscope`, `flopscope.numpy`, `os`, `whestbench` |
| archive contents | 2 plain `.npy` arrays, no pickle |

`bigcorr_head.npz` is **342 bytes**. Both files are numeric-only: `fnp.load`
refuses any other dtype outright ("object dtype would require pickle"), so a
column-name array in either would be a hard failure at setup rather than a
warning. The names live in a sidecar JSON the submission never reads.

### 9.3 The degradation ladder, and one bug it caught

Three rungs, each pinned by a test in `tests/test_submission_parity.py`:
scaled head, then the 15-float head if `bigcorr_head.npz` will not load, then
the uncorrected sparse pass if neither will. The two files are loaded in
separate `try` blocks so a corrupt scaled head degrades to the previous ship
rather than to no head. `DAMP = 0` still reproduces the uncorrected estimator
bit for bit and FLOP for FLOP, and a forced raise *inside* the new block --
not at the outer boundary -- still falls back to the dense pass.

That last test exists because this session already lost a submission to a
`SymmetryError` on official MLP 19 that eight local suites never produced. The
new block adds an `eigh`, an `arcsin` and a boolean-masked scatter, so the
guard is asserted against a failure inside it.

**A bug the parity test caught, worth recording.** The first port hard-coded
`1/2 - 1/(2 pi)` as `0.4204482076268573`; it is `0.3408450569081046`. The
estimator still ran, still returned finite numbers and still passed the
shape/finiteness/budget checks -- the only thing that caught it was the bitwise
comparison against `whestfloor/kernels.py`, at `max |diff| = 1.77e-01`. That is
exactly the failure mode the parity test exists for, and it is why the shipped
head is written twice and asserted equal rather than written once.

### 9.4 The coefficients are conditional on iid sampling at N = 25000

Both things matter and neither is a free parameter of the head:

- **iid.** The fitted coefficients absorb the `p/N` estimation noise of each
  control variate, which is a property of the sampler. A randomised QMC lattice
  changes the correlation structure of the scored draw and therefore changes
  the optimal shrinkage -- the channels stay exactly mean zero (each lattice
  point is marginally an exact standard Gaussian), so nothing becomes biased,
  but the weights stop being optimal.
- **`N = 25000`.** The correction columns scale as `N^-1/2` and so does the
  target, but the noise-to-signal ratio inside each channel does not.

If either changes, re-run `--mode data` (the generator takes `--n-samples` and
an `x0_fn`-equivalent is a one-line change) and then `--mode export`. Selection
is validation-driven and needs no hand-tuning; the whole refit is two commands.

## 10. Reproducing the artifact

`submission/bigcorr_head.npz` — **342 bytes**, one float32 vector of **20**
coefficients and nothing else. Regenerable bit-identically by

```
scripts/45_big_corrector.py --mode data --sub bigcorr --n-mlps 6000 \
    --n-seeds 4 --n-gt 32768 --block 25 --shard {0,1,2} --n-shards 3
scripts/45_big_corrector.py --mode export --drop mfv2g --out bigcorr_head.npz
```

`--mode export` selects the channel set by greedy forward selection on the
VALIDATION split, then ablates the optional blocks, then reads test once. The
final selection was:

```
+ relu1    val 1.4856e-06  1.554x        + pooled u1/u2 terms   rejected
+ mfv2     val 1.3861e-06  1.666x        - shape eigen          DROPPED
+ mfm      val 1.3050e-06  1.769x        - shape cumulants      DROPPED
+ dpilot   val 1.3005e-06  1.775x        - shape weightcols     DROPPED
+ cv1      val 1.3001e-06  1.776x        - shape suite          DROPPED
                                         - shape pilot alpha    DROPPED
                                         - shape sd_mc          DROPPED
```

Every optional block was dropped, which is why the shipped design is 20 columns
rather than 83 and why the power iteration for `u1`/`u2` is not in the shipped
kernel at all. `mfv2g` — the same variance channel with gates re-estimated from
4,096 scored rows — was force-dropped before selection: it was worth 0.0003x on
validation and would have cost ~440 dispatches (32 layers of gate reductions
plus a second transport).

The column order is FROZEN and asserted against the generator by
`tests/test_submission_parity.py::test_scaled_head_matches_the_numpy_generator`,
which also pins the flopscope path to `bigcorr.extract` at 1e-6 absolute:

```
one  s  Phi  phi  alpha
relu1   relu1*Phi   relu1*alpha
mfv2    mfv2*Phi    mfv2*alpha
mfm     mfm*Phi     mfm*alpha
dpilot  dpilot*Phi  dpilot*alpha
cv1     cv1*Phi     cv1*alpha
```

The shape columns are computed from `HEAD_ROWS = 4096` rows of the scored
draw, not the full sample. That is part of the contract, not an optimisation
detail: the coefficients were fitted against columns computed exactly that way.

## 11. Superseded as shipped, and refitted for the lattice

**The head is not shipping in the form above, and the reason is a real
mechanism rather than a measurement dispute.** A sibling agent measured a
randomised rank-1 lattice at **1.427x** projected graded (raw 1.5250e-06 ->
1.0530e-06, paired on the official 100) against this head's 1.254x — and the
two do not compose. Measured redundancy **0.573**: the head is worth 1.609x on
iid draws and **0.922x under a lattice**, i.e. net negative.

Why, exactly. §4.1's `cv1` is *provably* the optimal input-linear control
variate — `docs/learned_corrector.md` §3.2 shows `u_1 = rho^-1 d_1` unwinds to
`W^{1,-1} xbar`, so the k=1 block IS the first-order ANOVA projection and not
merely something resembling it. A rank-1 lattice is equidistributed in every
one-dimensional projection, so it annihilates **exactly those** first-order
terms before the head sees them. The coefficients above are therefore fitted
against a residual whose first-order part the lattice has already removed:
right mechanism, wrong residual.

That is a statement about the FIT, not about the family, and the distinction is
worth money — so the head is refitted on lattice draws.

### 11.1 What the lattice does to each channel, measured

One official-protocol MLP, `N = 24989`, `tau = 2.5`, `P = 225`, the same
estimator seed, iid against lattice, rms of each channel:

| channel | iid | lattice | ratio |
|---|---|---|---|
| `mfm` (first-order mean transport) | 3.618e-04 | 1.094e-04 | **0.302** |
| `mfmg` | 3.403e-04 | 9.008e-05 | **0.265** |
| `mfv` (variance transport, diagonal) | 5.512e-05 | 1.566e-05 | **0.284** |
| `cv1` (Hermite k=1, split-sample) | 4.768e-04 | 5.841e-04 | 1.225 |
| `relu1` | 3.654e-04 | 6.176e-04 | 1.690 |
| `mfv2` (exact `Cov(z^2)` anchor) | 1.144e-04 | 3.457e-04 | **3.021** |
| `dpilot` | 1.118e-02 | 1.092e-02 | 0.977 |

The first-order channels collapse to 0.27-0.30x, which is the mechanism read
directly off the features. `dpilot` is untouched because the pilot stays iid.
`cv1` does *not* collapse, and that is the subtle part: the split-sample form
correlates the target against **half-sample** means, and the two halves of a
rank-1 lattice are strided subsets that are not themselves equidistributed. So
`cv1` keeps its magnitude while losing its alignment with the full-sample error
— it becomes noise the old coefficients still pay for. `mfv2` and `relu1` grow,
because a lattice that is exact in 1-D projections is not exact in 2-D ones and
the degree-2 content it leaves behind is larger.

### 11.2 The bar, fixed before the refit

> **`beta` refitted on lattice draws must beat `damp = 0` under a lattice by
> `>= 1.10x` on the held-out TEST split, paired on the same MLPs.**

Not the old iid ship: the incumbent is now the lattice with the head OFF.
Reported with a 95% bootstrap interval over test MLPs, because the local
harness carries ~9% single-seed noise and a 100-MLP local raw figure ~8% of
realisation noise — a 1.02-1.04x point estimate is inside that and is not a
result.

Refit path: the references are functions of the weights alone, so not one label
is invalidated by changing the scored draw. `--mode relattice` keeps them and
recomputes only the features, which is 4x cheaper than regenerating and keeps
the comparison paired on exactly the same 475 networks.

### 11.3 The redundancy reproduces independently

`--mode fit --sub bigcorr_lat`, 75 MLPs of the lattice set (15 held-out MLPs —
underpowered, and quoted only for the sign and the mechanism):

| | val | TEST |
|---|---|---|
| shipped 15-float head, coefficients as shipped | 0.760x | **0.921x** |
| the same 15 columns refitted on lattice draws | 1.008x | 1.022x |
| the rich 83-column design refitted | 1.072x | 1.074x |

**0.921x reproduces the sibling's 0.922x on different networks, a different
harness and an independently written feature block.** The redundancy is real.

Per-channel at unit coefficient, and this is the mechanism in one column:

| channel | iid | lattice |
|---|---|---|
| `cv1` (Hermite k=1) | 1.321x | **0.374x** |
| `relu1` | 1.556x | **0.541x** |
| `cv2` (Hermite k=2) | 1.037x | 0.821x |
| `mfm` (first-order mean transport) | 1.529x | 0.989x |
| **`mfv`/`mfvg` (variance transport)** | 1.100x | **1.091x** |
| **`mfv2`/`mfv2g` (exact `Cov(z^2)` anchor)** | 1.146x | **1.055x** |

Every first-order channel is destroyed and the two VARIANCE channels are the
only survivors — they are degree-2 objects, which a lattice equidistributed in
one-dimensional projections does not touch. Leave-one-group-out on validation
agrees: dropping `mf_var_diag` or `mf_var_exact` costs 1.072x -> 1.048x, and
dropping anything else costs `<= 0.005x`.

`cv1` at **0.374x** is worth dwelling on, because it is not simply "the lattice
already did that job". A channel that had become redundant would measure
1.000x. 0.374x means it is now actively harmful, and §11.4 says why.

### 11.4 The split-sample halves of a lattice are not samples

Every estimated-coefficient channel here is split-sample: `dbar` from one half
against the cross-moment of the other, which under iid draws is what removes
the `Cov(g' G^-1 g, y)/N` self-term. Under a rank-1 lattice the first `N/2`
points of `frac(i z / N)` are a contiguous arc, **not** an equidistributed set.
So `dbar` is `O(N^-1/2)` noise while the full-sample mean it stands in for is
nearly exact — the channel keeps its magnitude (`cv1` rms is 1.225x its iid
value, not 0.3x) and loses its alignment with the error. That is the 0.374x.

The fix is free and is implemented as `--interleave`: for prime `N` the
even-index subset `{2 i z / N}` is itself a rank-1 lattice with generating
vector `2 z` (`gcd(2, N) = 1`), and so is the odd one. Reordering the base as
`[0, 2, 4, …, 1, 3, 5, …]` makes both split halves equidistributed. It is a row
permutation of a table built once in `setup`, so it costs nothing at grade
time, and it should restore `cv1`, `cv2` and `relu1` to informative — or, at
worst, to a correct 1.000x instead of 0.374x.

### 11.5 Result: the refit clears the bar on the point estimate, and the
interval does not exclude a miss

375 MLPs of the lattice set x 4 Cranley-Patterson shifts, split by MLP seed,
**75 held-out test networks**. Selection is validation-only (greedy forward over
channels, then the optional blocks); the test split is read once.

```
+ mfv2g    val 1.0547e-06  1.097x        + pooled u1/u2 terms   rejected
+ cv1mfg   val 1.0165e-06  1.138x        - every shape block    DROPPED
+ dpilot   val 1.0130e-06  1.142x
+ cv1      val 1.0106e-06  1.144x
+ mfv2     val 1.0097e-06  1.146x
```

> **held-out TEST against `damp = 0` under a lattice, paired on the same 75
> networks: `1.1324x`, 95% bootstrap CI over MLPs `[1.0688, 1.1978]`.**

**Verdict against the pre-registered `>= 1.10x`: PASS on the point estimate,
and the interval does not exclude a miss.** The lower bound is 1.069. Going
from 45 to 75 test MLPs moved the interval from `[1.0703, 1.1835]` to
`[1.0688, 1.1978]` — it did not tighten, because the variance is dominated by
MLP-to-MLP heterogeneity rather than by the count. Another 75 networks would
buy little; what would settle it is the grader.

For scale, on the same 75 networks the **iid-fitted** head reads 1.5858e-06
against this one's 1.0909e-06, i.e. the refit is **1.454x** the shipped
coefficients under a lattice — and 0.641x of raw, which is the same
net-negative the sibling measured, seen from the other side.

The selected channel set is *not* the iid one. `mfv2g`/`mfv2` — the exactly
anchored layer-2 covariance gap — lead, as §11.3 predicted; `cv1mfg` and `cv1`
survive with small coefficients; `relu1`, `cv2`, `cv3`, `mfm` and `mfv` are all
dropped. Every shape block is dropped again, so the design is 20 columns.

### 11.6 Deliverable

`heads/lattice_head.npz` — 342 bytes, one float32 vector of 20 coefficients,
nothing else, numeric-only. **It is not wired into `submission/`**, which is the
integrator's; `submission/estimator.py` and `tests/test_submission_parity.py`
are at `c1f8910` and all 8 original parity tests pass.

Operating point it is fitted for: `tau = 2.5`, `N = 24989`, `P = 225`,
`rqmc.ship_lattice_base()` with `lattice_x0_fn`, pilot iid. Column order, frozen:

```
one  s  Phi  phi  alpha
mfv2g   mfv2g*Phi   mfv2g*alpha
cv1mfg  cv1mfg*Phi  cv1mfg*alpha
dpilot  dpilot*Phi  dpilot*alpha
cv1     cv1*Phi     cv1*alpha
mfv2    mfv2*Phi    mfv2*alpha
```

`s`, `Phi`, `phi`, `alpha` come from `HEAD_ROWS = 4096` rows of the scored draw
(part of the contract, not an optimisation). The flopscope implementations of
`mfv2`/`mfv2g`, `cv1mfg` and `mfm` are in `whestfloor/kernels.py`
(`_layer12_exact`, `_relu1_cv`, `_transport_pair`, `_corrected_head2`), verified
against the numpy generator at 1e-6 absolute and against the previous kernel
bitwise and FLOP-for-FLOP with `beta2=None`. `cv1mfg` needs the scored-pass
gates, which the shipped `_corrected_head2` does not currently build — it uses
pilot gates for `mfm` — so wiring this vector needs the 4,096-row per-layer gate
reduction (measured 1.01e8 FLOPs, 0.037% of `B`, ~160 dispatches) added.

Regenerate with:

```
scripts/45_big_corrector.py --mode relattice --sub bigcorr_lat \
    --from-sub bigcorr --shard {0,1,2} --n-shards 3
scripts/45_big_corrector.py --mode export --sub bigcorr_lat --out lat_head.npz
```

### 11.7 The one thing not measured, and it is the next thing to run

`--interleave` is implemented and **was not generated in time**. §11.4 argues it
should recover `cv1`, `cv2` and `relu1` from actively harmful (0.374x, 0.821x,
0.541x at unit coefficient) to at worst neutral, because it makes both
split-sample halves genuine sublattices. Those three channels are the entire
degree-1-and-2 estimated-coefficient family, so if the argument holds the
selected set and the gain both change — and it costs nothing at grade time, being
a row permutation of a table built once in `setup`. One `--mode relattice
--interleave` run and one `--mode export` settles it.

## 12. Honest caveats

- **The graded number is a projection, not a measurement.** §5's model
  reproduces the shipped point to 0.2% and its argmin to 3%, but it has been
  validated at one operating point only. The measured quantity is the
  **1.286x** ratio on 95 held-out networks.
- **The test split has been read more than once.** The channel selection is
  validation-only, but the *design* of the experiment (which channels to
  generate, when to stop) saw test numbers at 40, 75, 225, 300, 375 and 475
  MLPs. The 1.286x therefore carries some optimism; the 95-MLP realisation
  noise on such a ratio is itself several percent.
- **Local MLPs, local seeds.** Seeds `400000+`, drawn from the same generator
  as the official networks but not the same draws. The previous round's
  equivalent transfer went the favourable way (test 1.354x -> official 1.516x).
- **`mfv2g` uses gates correlated with the gap they multiply.** `O(1/N)`
  against an `O(N^-1/2)` correction, 0.6% at this `N`. It is a bias, it is
  small, and it is measured through the end-to-end held-out number.
- **The `q2` block was never generated at scale.** It doubles the extraction
  cost and its 1.087x was retracted upstream (commit 3a7fde0); it is carried as
  a structural zero. If it is revived it belongs in this design.
