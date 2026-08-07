# The offline-trained corrector, and the control variate it turned out to be

**Result: shipped. Raw final-layer MSE `3.7194e-6`, adjusted `3.99e-7`, 0
raises in 100, on the official 100-MLP suite with the N=1e9 reference — against
`5.8050e-6` / `6.05e-7` for the previous ship measured in the same run through
the identical code path. That is 1.561x raw and 1.516x adjusted.** Everything
here is reproducible from `scripts/28_learned_corrector.py`
(`--mode anova|data|refresh|fit|score|ship`).

The headline is not the head. The head is worth about 1.1x on its own; the
mechanism it learned to weight is worth the rest, and it is a **layer-1
control variate** whose expectations are known in closed form. That mechanism
is not something the barrier theorem in `docs/floor_theorem.md` covers as
stated, and finding it was the point of looking.

---

## 1. Why this family was the only one left

`docs/floor_theorem.md` proves that a surrogate whose ANOVA content sits at
order `<= k` caps the variance reduction at `1 / (1 - sum_{d<=k} f_d)`, and
measures `f_1 = 0.276`, giving **1.38x at `k = 1`**. That retrodicts every
mechanism in the ledger: linear control variates 1.33x, MLMC 0.94x,
Rao-Blackwell on decided neurons 1.001x.

Two objects escape it. The network itself (self-defeating: `rho >= 0.975`
needs `r = 251` of 256 at 1.96x dense cost) and **a learned map fitted to the
weight -> mean relation**. Training and precomputation are unbounded and load
at **zero FLOPs** (`fnp.load`, measured), and the submission may carry 50 MiB
in 50 files (`whestbench/limits.py`). The shipped head is 374 bytes.

## 2. What the corrector predicts

Per output neuron `j`, the residual of the shipped estimator:

    target_j = mu_true_j - mu_hat_j ,   mu_hat_j = mean_s relu(z^32_{s,j})

from features the sparse pass already has in hand. The head is a ridge
regression, its output is added back, and `DAMP = 0` recovers `mu_hat` bit for
bit through the identical code path.

**The label-noise argument that makes this affordable.** The target carries
Monte-Carlo noise from a moderate-N reference, but a model with 28 parameters
fitted over `K` (MLP, neuron) pairs recovers its coefficients to
`label_noise / sqrt(K)`. With 256 neurons x 640 MLPs, `K = 163,840`. So the
training set is **640 MLPs at N_gt = 2 x 100,000** rather than a handful at
1e9: cheap labels, many of them. It fits in 14 MB on disk because weights are
never stored.

## 3. The mechanism: exactly-known layer-1 expectations

`z^1_i = x . W^1[:,i]` is *exactly* Gaussian with known scale
`sigma_i = ||W^1[:,i]||` — the only place in the network where anything is
known exactly. Two consequences, both exact:

    E[He_k(t_i)] = 0                                for every k >= 1     (*)
    Cov(He_k(t_i), He_l(t_j)) = delta_kl k! rho_ij^k     (Mehler)       (**)

with `t_i = z^1_i / sigma_i` and `rho` the correlation matrix of the layer-1
pre-activations. (*) makes the sample means of `He_k(t)` **exactly-mean-zero
control variates that the forward pass already computed**; (**) makes their
Gram **analytic and block diagonal in `k`** — no estimated covariance matrix.

The estimator uses this channel *twice*, in two different ways, and ships both
because they fail differently.

### 3.1 `cv1mf` — analytic coefficients, no estimation noise

`E[relu(z^1_i)] = sigma_i / sqrt(2 pi)` exactly, so
`d1 = mean_s relu(z^1) - E[relu(z^1)]` is free and exactly mean zero. Pushing
it through the mean-field linearisation — the rectifier Jacobian replaced by
its expectation `P(z^l > 0) = Phi(alpha^l)`, with `alpha` from the pilot —
costs **31 matvecs, 4.1e6 FLOPs**, and yields a per-output-neuron prediction of
the sampling error. Its coefficients are analytic, so it carries no estimation
noise; they are also only approximate, so it is biased.

### 3.2 `cv1`, `cv2` — exact coefficients, estimated from the sample

The textbook regression control variate needs `beta_j = S^{-1} Cov(g, y_j)`,
a `(width, width)` cross-moment costing `2 x 1.11e9` FLOPs. But the correction
only ever appears contracted:

    correction_j = d' S^{-1} chat_j = u' chat_j ,   u = S^{-1} d

so with `w_s = u . (g_s - d)` it collapses to
`correction_j = (1/N) sum_s w_s y_{sj}` — **two length-N matvecs**,
`O(N * width)` rather than `O(N * width^2)`.

The `k=1` block is evaluated in the *input* basis, where the Gram is exactly
the identity: `u_1 = rho^{-1} d_1` unwinds to `W^{1,-1} xbar`, so no solve is
needed. That matters numerically (`cond(rho) = 3.8e8` at this shape, against
`cond(2 rho .^ 2) = 2.41`) and it *proves* the block is the optimal
input-linear control variate rather than merely resembling one — the object
the barrier caps at 1.38x. `k=2` is what escapes that cap.

Billed in a real `BudgetContext`, the entire feature block — both Hermite
orders, the Gram, two solves, the mean-field propagation, the final-layer
sample moments and the head matvec — is **1.12e8 FLOPs = 0.41% of the free
budget**, plus 10.6 ms of participant Python (3.9% of the free budget at
`lambda = 1e11`). `F/B` goes from 0.0915 to 0.0920.

### 3.3 The order that pays, measured before anything was built

`--mode anova`, 4 local MLPs x 400,000 samples, exact projection onto the
Hermite family. Cumulative population share of `Var(relu(z^32_j))` explained:

| | mlp 0 | mlp 1 | mlp 2 | mlp 3 | implied bound |
|---|---|---|---|---|---|
| `k = 1` | 23.4% | 29.4% | 23.5% | 29.5% | 1.31-1.42x |
| `k <= 2` | 37.8% | 47.9% | 38.5% | 44.7% | **1.61-1.92x** |
| `k <= 3` | 39.6% | 50.3% | 40.7% | 46.8% | +2% |
| `k <= 6` | 43.4% | 55.9% | 45.0% | 50.9% | +5% |

The `k=1` row reproduces `f_1 = 0.276` from a fifth independent estimator, so
the barrier's constant is right. But `f_2 ~ 0.16` **along the network's own
first-layer directions**, and the barrier's `k=2` instance — which nobody had
computed — is `1/(1 - 0.43) = 1.75x`.

Each block costs `p/N = 256/8500 = 3.0%` of the residual in estimation noise
(6.0% for the split form of §3.4), against ~2% of real gain at `k = 3`. **So
`k <= 2` is the operating point**, and that was a prediction from this table
made before the end-to-end run. It held: measured over all 640 generated MLPs,
with the coefficient fixed at 1 and no fitting at all,

| | unbiased true MSE | |
|---|---|---|
| sparse MC (previous ship) | 6.0107e-6 | 1.000x |
| `- cv1` (Hermite k=1) | 4.7972e-6 | 1.253x |
| `- cv1 - cv2` (k<=2) | 4.5255e-6 | **1.328x** |
| `- cv1 - cv2 - cv3` (k<=3) | 5.0210e-6 | 1.197x |
| `- cv1mf` (mean-field) | 4.2034e-6 | **1.430x** |

`k=3` loses, exactly as the `p/N` argument says. And the *analytic-coefficient*
arm beats the *estimated-coefficient* arm at unit weight, because at `N = 8500`
the 3-6% estimation penalty per block outweighs the mean-field approximation
error. Neither dominates, which is why the head gets both.

### 3.4 The split that removes a real bias

Estimating `d` and `chat_j` from the same samples leaves

    E[ dbar' G^-1 chat_j ] = Cov(g' G^-1 g, y_j) / N

which is a **bias, not noise**: `He_k(t)^2` has heavy even-Hermite content that
couples straight to `y`. Measured on 12 local MLPs, one-pass versus a two-half
split (`dbar` from one half against `chat` from the other, both ways — same
cost, exactly unbiased):

| | one pass | split |
|---|---|---|
| `- cv1` | 1.684x | **1.838x** |
| `- cv1 - cv2` | 1.802x | **1.917x** |
| `- cv1 - cv2 - cv3` | 1.193x | 1.865x |

At `k = 3` the one-pass form does not merely lose the gain, it **reverses the
sign of the correction**. The split is what makes the mechanism usable, and it
is why the shipped code halves the sample.

### 3.5 What did not work, and was tried first

| arm | measured (12 local MLPs unless noted) | why |
|---|---|---|
| Gaussian closure `g = m Phi + s phi` — Rao-Blackwell the last layer | **0.809x** | it drops the `k >= 3` Hermite modes of `z^32`, which at `rms\|alpha\| = 3.4` are ~0.3% of `Var(relu z)`, while injecting the full non-Gaussian bias |
| + sample-Edgeworth (skew) | 0.963x | recovers the bias, not the point |
| + sample-Edgeworth (skew + kurtosis) | 0.994x | still no gain |
| mean-field propagation with per-pilot-sample gates (150 samples) instead of `Phi(alpha)` | 1.046x | 150 samples is far too noisy an estimate of `E[J]` |
| empirical Gram instead of the analytic Mehler one | 1.621x vs 1.631x | costs 1.11e9 FLOPs to be slightly worse |
| Hermite `k <= 3` | see 3.3, 3.4 | `p/N` exceeds the gain |
| full order-2 basis (all 32,896 input quadratics) | not run | `p > N` at 8500 samples; unusable whatever it costs |

The Rao-Blackwell row is worth stating plainly, because the competition forum
lists it (`docs/recon.md` A1) as *"the dominant single win in every hybrid"*.
At depth 32 it is not. Writing the ReLU mean in Hermite coefficients
(`a_1 = s Phi`, `a_2 = s phi`, `a_k = (-1)^k s He_{k-2}(alpha) phi`),

    Var(relu z) / s^2 = Phi^2 + phi^2/2 + alpha^2 phi^2/6 + (alpha^2-1)^2 phi^2/24 + ...

and the Gaussian closure reproduces the `k=1,2` terms exactly while dropping
everything above. At `alpha = +3` that is 0.2% of the variance; at `alpha = -3`
it is 17x — of a quantity that is 1e-4 of the total. Summed over the actual
`alpha` distribution it is worth nothing, and the bias it injects is worth
less than nothing. It is kept as a feature (`gap`, `sk`, `ku`) and the head is
free to use it; the ablation in §5 shows it does not.

## 4. Data

`--mode data`, fresh **local** MLP seeds `100000..100639`, disjoint from every
existing suite and from the official one. Each MLP is streamed: weights made,
the instrumented sparse pass run at the shipped `(tau, N, P) = (2.5, 8500,
150)`, two independent Monte-Carlo reference halves of 100,000 samples
accumulated, features and labels appended, weights dropped. Nothing but
`(n_mlps, 256)` arrays touches disk — 640 MLPs is **14 MB**, not 5.4 GB of
weights. Generation took 40 minutes on two cores.

Reference noise per neuron is `sqrt(0.0495 / 200000) = 5.0e-4` against a
residual of `~1.7e-3` after correction. It is independent of the features, so
the ridge is unbiased in it; and every number below uses the **paired unbiased
estimator** `T = mean (p - a)(p - b)` over the two halves, which removes the
reference's own variance exactly.

**The official suite (`official_mini.npz`) was never read by `--mode fit`.**
The penalty, the Hermite order, the split-vs-one-pass choice and the feature
set were all selected on a validation split of the generated data, and a third
split was held back and read once.

## 5. Fit

640 MLPs split by MLP into 384 train / 128 validation / 128 test
(98,304 training rows, 28 features). Ridge is closed-form with per-column
scaling; the penalty is flat over seven decades (`1e-8` to `1e-1` all give
1.501-1.504x on validation), which is what a well-conditioned design looks
like. Selected `lambda = 1e-3`.

    validation  1.504x        TEST (read once)  1.354x

**Leave-one-group-out and only-one-group, on the validation split:**

| group | without it | it alone |
|---|---|---|
| `cv1` (Hermite k=1) | 1.480x | 1.257x |
| `cv2` (Hermite k=2) | **1.437x** | 1.065x |
| `cv3` (Hermite k=3) | 1.504x | 0.998x |
| `cv1mf` (mean-field) | **1.404x** | **1.399x** |
| RB gap + Edgeworth | 1.504x | 0.998x |
| shrink (`mu`, `mu*Phi`) | 1.504x | 0.979x |
| shape (`s, Phi, phi, alpha, sd_mc, dpilot`) | 1.464x | 0.989x |
| weights + suite scalars | 1.504x | 0.990x |
| **full design** | — | **1.504x** |

Read off: `cv1mf` and `cv1` are two estimates of the same channel and are
partly redundant (`cv1mf` alone is 1.399x, but removing `cv1` still costs
0.024x), `cv2` is the only genuinely additive control variate, and the
`shape` group — dominated by `dpilot`, the gap between the scored and pilot
means — is worth 0.04x. **The Rao-Blackwell/Edgeworth, shrink and
weight/suite groups are worth exactly nothing**, which is the measurement the
forum's A1 and B10 claims deserve.

An exhaustive search over all 256 subsets x 7 penalties finds the best subset
at 1.504x — the full design minus the `shrink` group, better by 0.014%, far
under the 0.2% threshold for overriding. So the shipped head is the full
design, and its `cv3` coefficient is exactly zero because the shipped kernel
supplies that column as zero at `CV_KMAX = 2`.

Largest coefficients (scaled units): `cv1mf -7.96e-4`, `cv1_a -5.11e-4`,
`cv1_Phi +3.68e-4`, `cv1 -3.52e-4`, `cv1mf_Phi -3.33e-4`, `cv2_a -2.97e-4`,
`dpilot -2.92e-4`. The `cv1`/`cv1mf` coefficients sum to roughly `-1` in the
directions that matter, which is what an optimally-shrunk pair of control
variates should look like.

**The MLP head fails.** A one-hidden-layer tanh MLP (16 units, full-batch Adam
in numpy, 250 epochs, two learning rates) reaches 1.328x on validation and
**1.063x on the test split, i.e. 0.785x relative to ridge** — a clear miss of
the 1.5x bar, and in fact worse than ridge. The residual after the control
variates is Monte-Carlo noise plus a nearly-linear closure bias; there is no
nonlinear structure left for capacity to find. **Ridge ships.**

## 6. End to end on the official suite

`--mode score`, all 100 official MLPs, N=1e9 reference, so `raw_mse` is
leaderboard-comparable. `damp=0` is the exact ablation: the identical code
path with the head switched off, which reproduces the previous ship bit for
bit — and it lands on `5.8050e-6`, the number `docs/sparse_sign_stable.md`
published for the shipped file, to five significant figures.

| variant | raw_mse | F/B | C/B | adj@1x | adj@2x | adj@3x | raises |
|---|---|---|---|---|---|---|---|
| `damp=0` (= previous ship, bitwise) | 5.8050e-06 | 0.0915 | 0.1042 | 6.050e-07 | 6.785e-07 | 7.521e-07 | 0 |
| **`damp=1` (shipped)** | **3.7194e-06** | **0.0920** | 0.1073 | **3.992e-07** | 4.563e-07 | 5.134e-07 | 0 |
| ratio | **1.561x** | | | **1.516x** | 1.487x | 1.465x | |

`adj@kx` recomputes the multiplier with the residual wall time scaled by `k`.
The raw ratio is machine-independent; the adjusted ratios are not, and they
degrade slowly with residual because the feature block's 10.6 ms is charged at
`lambda`.

The shipped file itself (`--mode ship`, real protocol-3.0 `mlp.seed` per MLP,
run alone on this box): raw **3.7194e-6**, `F/B` **0.0920**, adjusted
**4.039e-7** at 1x residual / 4.658e-7 at 2x / 5.277e-7 at 3x, **0 raises in
100**, worst single MLP 3.2805e-5 (down from 4.2164e-5), max `C` over the
suite 0.1177 of `B`, setup 0.001 s against a 5 s window. The defensive path
was probed by forcing `_sparse` to raise: it returns a finite `(32, 256)`
prediction at raw MSE 1.09e-5 and `C/B` 0.0946.

## 7. Bars, fixed before the run

| # | bar | measured | verdict |
|---|---|---|---|
| 1 | ridge corrector beats the shipped 5.805e-7 adjusted by **>= 1.25x**, fitted only on generated data | **1.516x** (3.992e-7 against a same-run ablation at 6.050e-7); 1.47x against the previously published 5.870e-7 | **PASS** |
| 2 | MLP head beats ridge by a further **>= 1.5x** | **0.785x** on the untouched test split | **FAIL — ridge ships** |
| 3 | **0 raises** on all 100 official MLPs, and a `damp=0` ablation recovering the uncorrected estimator through the identical code path | 0/100; `damp=0` reproduces `sparse_mc_kernel` bitwise **and FLOP-for-FLOP** (`tests/test_submission_parity.py::test_corrector_damp_zero_is_exactly_uncorrected`) | **PASS** |

Bar 3's FLOP equality is why the layer-1 mean is kept as a live array rather
than reduced inside the scored loop: a `mean()` there would be billed before
the early return and the ablation would no longer be exact.

## 8. Honest caveats

- **The head is fitted on locally-seeded MLPs, not official ones.**
  `make_mlp(seed)` and the official
  `default_rng(SeedSequence(seed).spawn(3)[0])` are different networks drawn
  from the same distribution, so the fitted map should transfer — and §6 is
  exactly the test of that, taken once. Note the gap between the internal test
  split (1.354x) and the official suite (1.516x): the direction is favourable,
  but it is a reminder that a 128-MLP ratio has ~10% of realisation noise on
  it.
- **Selection bias inside the fit is real but bounded.** The subset/penalty
  search evaluates ~1,800 candidates on the validation split, so 1.504x is
  optimistic; the test split says 1.354x. The official-suite number is quoted
  first precisely because it is the only one nothing was tuned against.
- **`tau = 2.5`, `N = 8500`, `P = 150` are inherited, not re-swept.** They were
  calibrated on the official suite in the previous round
  (`docs/sparse_sign_stable.md` §6). With the sampling variance now 1.56x
  lower, the closure bias of pruning is a larger share of what is left, which
  argues for a slightly higher `tau`; that sweep has not been run.
- **`C/B` is above the 0.1 floor and the feature block pushed it up.** The
  block costs 0.41% of the free budget in FLOPs and 3.9% in residual — the
  residual dominates, because it is ten passes over an 8.7 MB array. Above the
  floor the adjusted score is flat in `N`, so shrinking `N` does not recover
  it; a cheaper feature set would. Specifically, the RB/Edgeworth group is
  measured at *exactly zero* value and costs four of those passes, so ~1.3% of
  the budget is provably dead weight that a re-fit without it would recover.
  It was left in so the ablation table above is a measurement of the shipped
  code rather than of a variant.
- **The control variates are unbiased; the head is not.** `E[He_k(t)] = 0` is
  exact and the split-sample coefficient estimate keeps the corrections
  mean-zero. The head then adds a fitted term with no such guarantee. Its
  intercept and `mu` columns are the multiplicative/additive shrink that
  `docs/recon.md` B10 records as topping out — and here they measure at
  exactly 1.000x, so nothing rests on them.
- **Residual risk is one-sided.** If the grader's single core is slower than
  this box, the 10.6 ms grows while `F` does not. Hence adj@2x and adj@3x are
  quoted, and they still clear bar 1 (1.487x, 1.465x).
- **This is a `k <= 2` mechanism and it is close to its ceiling.** Getting
  materially past 2x needs exact expectations at a layer deeper than 1, and
  the network provides none: layer 1 is the only place where the
  pre-activation law is known.

## 9. Where this leaves the board

| | adjusted | x floor |
|---|---|---|
| proven noise floor (`docs/floor_theorem.md`) | 4.949e-12 | 1.0 |
| leaderboard #1 (dpskv5) | 3.63e-10 | 73.3 |
| the sign-stable family as a competitor ships it (SOX #41) | 1.60e-07 | 32,300 |
| **this repo, now** | **3.99e-07** | **80,600** |
| this repo, previous ship (sparse MC) | 5.87e-07 | 118,600 |
| grader's plain-MC constant `sampling_mse` | 6.4695e-07 | 130,700 |

Now **1.62x below the grader's own plain-Monte-Carlo line**, up from 1.10x.
Still 2.5x behind the best sign-stable entry on the public board and 1,100x
behind the leader. The 73.3x of total remaining headroom in the benchmark is
unchanged; this took 1.5x of it.
