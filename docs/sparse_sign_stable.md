# Sign-stable sparse Monte Carlo

At depth the network is nearly decided, so most rectifiers almost never flip
sign. This document records what that buys, what it does not, and the three
measurements that separate the two. Everything here is reproducible from
`scripts/25_sparse_sign_stable.py` (`--mode alpha|price|bias|score|ship`)
against the official 100-MLP suite.

**Result: shipped, worth 1.44x, and two-thirds of the idea is dead.** The one
version that pays is the least clever one — prune the neurons that never fire
out of the matmuls. The version this work set out to build, fusing the modal
sign pattern into a single matrix and correcting only the flips, is **4.95x
MORE expensive per sample** than the dense pass it replaces.

---

## 1. The premise, measured

`--mode alpha`, 4 official MLPs x 200,000 samples, with the flip rate taken
from a second independent stream. `alpha = m/s` per neuron per layer.

`rms|alpha|` is **3.44 at layer 32, not 4.4**, and it is 0 at layer 1 — exactly,
because `E[z^1] = E[x] W^1 = 0`. The mean is *generated* by rectification and
accumulates with depth:

| layer | 1 | 5 | 9 | 17 | 25 | 29 | 32 |
|---|---|---|---|---|---|---|---|
| `rms\|alpha\|` | 0.00 | 1.41 | 2.13 | 2.46 | 2.94 | 3.19 | 3.44 |
| `\|alpha\| < 3` | 100% | 97.5% | 82.5% | 72.9% | 58.2% | 49.6% | 41.4% |
| E[flips] / 256 | 128.0 | 47.2 | 30.0 | 23.0 | 17.7 | 14.3 | 12.9 |

`alpha` is close to Gaussian across neurons at each layer, so the fraction with
small `|alpha|` falls only *linearly* near zero. That single fact is what kills
the sparse-correction schemes.

**Sign flips: 987 +- 64 per sample of 8192 neurons = 12.05%.** The planning
estimate was 410 / 5.0%; the truth is 2.4x worse.

### The kink set is not the flip set

A flip-correcting scheme must *evaluate* every neuron that **might** flip, not
just the ones that do. That set is `{|alpha| < tau}`:

| tau | kink `\|a\|<tau` | dead `a<-tau` | flips captured | missed/sample | kink per flip |
|---|---|---|---|---|---|
| 1.0 | 30.9% | 34.8% | 86.36% | 134.6 | 2.6x |
| 2.0 | 52.7% | 23.7% | 99.23% | 7.59 | 4.4x |
| 2.5 | 62.4% | 18.9% | 99.90% | 0.96 | 5.2x |
| 3.0 | 71.4% | 14.4% | 99.99% | 0.073 | 5.9x |
| 4.0 | 87.4% | 6.4% | 100.00% | 0.000 | 7.3x |
| 5.0 | 98.9% | 0.5% | 100% | 0 | 8.2x |
| 6.0 | 100% | 0.0% | 100% | 0 | 8.3x |

**At the thresholds the idea is usually stated with — `|alpha| > {3,4,5,6}` —
the achievable sparsity is 28.6% / 12.6% / 1.1% / 0.0%.** There is nothing to
exploit at 4 and beyond. To capture 99%+ of the flips you must evaluate more
than half the network, which is 4.4x-5.9x more neurons than actually flip.

---

## 2. Three structural findings

### 2.1 `A^T mu_input` is identically zero, and the two arms cancel 15x

On the modal sign pattern the network collapses exactly:

    z^32 = x A + sum_{l<32} eps^l R^l
    A   = W^1 D^1 W^2 ... D^31 W^32          (D^l = diag(1{alpha^l > 0}))
    R^l = W^{l+1} D^{l+1} ... D^31 W^32
    eps^l = relu(z^l) - d^l . z^l            (nonzero only where the sign flips)

Verified numerically to 3e-5 (float32 round-off). The identity is real. The
*estimator* built on it is not, for two reasons.

**The exact part predicts nothing.** `x ~ N(0, I)`, so `mu_input = 0` and
`E[x A] = 0` exactly. The entire answer lives in the correction, which is the
opposite of the intended split.

**The two arms cancel catastrophically.** Measured, pooled over 3 official MLPs:

| quantity | value |
|---|---|
| `Var(x A_j)` — modal arm | 1.968 |
| `Var(correction arm)` | 1.921 |
| `Var(z^32_j)` — their sum | **0.128** |
| `Var(h^32_j)` — what plain MC carries | **0.062** |

A 15.3x cancellation. Sampling only the correction has **31x MORE variance
than plain Monte Carlo**. The `2^{32-l}` ill-conditioning that
`docs/floor_theorem.md` §1 identifies for the ungated tail decomposition does
not go away when the modal gates make the chain norm-preserving; it comes back
as a cancellation between the two arms instead.

### 2.2 There is no Rao-Blackwellisation in the decided neurons

The hope was that conditioning on the decided neurons removes their
contribution to the estimator variance entirely. It does — and their
contribution is nothing. Variance of the correction arm restricted to the kink
set, against the full arm's 1.921:

| tau | Var(kink-only arm) | share carried by the DECIDED neurons | max RB gain |
|---|---|---|---|
| 0.5 | 1.697 | 11.65% | 1.132x |
| 1.0 | 1.865 | 2.90% | 1.030x |
| 2.0 | 1.919 | 0.102% | 1.001x |
| 3.0 | 1.921 | 0.001% | 1.000x |

A decided neuron's ReLU deviation has `Var(eps) ~ 1e-7` against `Var(z) ~ 0.1`
— a 1e-6 ratio. The variance lives entirely in the kink neurons, which are
exactly the ones that must still be sampled. Gain (b) does not exist at any
threshold where the estimator is usable.

### 2.3 Thresholding on alpha is not free, and the pilot's noise does not average away

`alpha` has to come from somewhere. A short dense pilot pass is the cheapest
accurate source, but the frozen constants it produces for the pruned neurons
are *shared by every scored sample*, so the pilot's own sampling error enters
as a fixed offset that **does not shrink with the scored sample count**. This
is why more pilot is not monotonically better: at matched budget, P = 600
scores 6.26e-6 raw against P = 150's 5.65e-6, because the pilot then eats the
samples it was meant to improve. Measured flat over P = 80..250.

---

## 3. Cost, billed rather than modelled

`--mode price`. Per-sample marginal FLOPs from differencing two batch sizes
inside a real `flops.BudgetContext`, flopscope 0.10.0.

| scheme | flops/sample | x cheaper | setup |
|---|---|---|---|
| dense MC (final layer only) | 4,198,656 | 1.00 | 0 |
| dead-prune `tau=None` (= dense, the ablation) | 4,198,656 | 1.00 | 6.3e8 |
| dead-prune `tau=4.0` | 3,677,452 | 1.14 | 6.6e8 |
| dead-prune `tau=3.0` | 3,100,558 | 1.36 | 6.6e8 |
| **dead-prune `tau=2.5`** | **2,793,985** | **1.51** | 6.6e8 |
| dead-prune `tau=2.0` | 2,498,297 | 1.68 | 6.5e8 |
| dead-prune `tau=1.0` | 1,869,930 | 2.25 | 6.5e8 |
| dead-prune `tau=0.0` | 1,098,241 | 3.83 | 6.5e8 |
| **modal fusion + kink correction, `tau=2`** | **20,361,710** | **0.21** | **1.39e10** |
| modal fusion + kink correction, `tau=3` | 35,331,388 | 0.12 | 1.65e10 |

Two things to read off.

**4x is the theoretical ceiling of dead-pruning, not just the measured one.**
Cost scales as `(|ON|/n)^2`, and `ON` must contain every neuron with positive
mean — about half by construction. `(0.5)^2 = 4x` is unreachable except at
`tau = 0`, i.e. zero tolerance for flips, where the estimator is destroyed
(sign error 4.5e-2, 7,700x the whole score). The measured 3.83x at `tau=0`
confirms the bound is tight.

**Modal fusion is worse than dense by 4.95x** and needs 51% of the free budget
in setup before a single sample is drawn, because the kink-to-kink coupling is
O(depth^2) over sets that are half the width. Fusing only the last few layers
is break-even (1.0-1.2x) and not worth the machinery.

---

## 4. The sign error, measured paired

`--mode bias`. The pruned and dense passes run on the **identical sample
stream**, so the Monte Carlo noise cancels and the difference is the pruning
alone. This matters: unpaired, the seed-to-seed spread of the raw MSE is ~50%
on 4 MLPs, larger than every effect being measured. An early unpaired
comparison made pruning look *worse* than dense; it was pure seed noise.

One subtlety the script gets right and an earlier version got wrong: a single
generator feeds the pilot and then the scored draw, so the scored *stream*
depends on `n_pilot`. Pairing therefore only holds at fixed `P`, and each
column below is differenced against its **own** `tau=None` run at the same `P`.
Baselining every column against one shared dense run injects ~8e-7 of stream
noise into a ~1e-7 effect and makes the table meaningless.

Excess MSE over the dense pass on the same stream, 12 official MLPs, N=6000
(dense baseline 8.58e-06 at P=150):

| tau | P = 150 | P = 20000 (oracle control) |
|---|---|---|
| 4.0 | +1.5e-10 | -4.2e-11 |
| 3.0 | -6.0e-09 | -2.8e-08 |
| **2.5** | **-4.7e-07** | **+1.6e-07** |
| 2.0 | +2.0e-06 | +1.3e-06 |
| 1.0 | +2.2e-04 | +1.5e-04 |

Changing `tau` changes the whole nonlinear trajectory, not just a small
perturbation, so the pairing only partially cancels: the residual noise on this
table is about +-5e-7. Read it as a bracket rather than a point measurement.

- `tau >= 3`: sign error <= 3e-08, entirely negligible.
- **`tau = 2.5`: not resolvable above the +-5e-7 noise.** Bracketed by its
  neighbours (~1e-8 at 3.0, ~1.5e-6 at 2.0) it is of order 2-3e-07, i.e. **3-5%
  of the final MSE — inside the 10% accuracy bar.**
- `tau = 2.0`: +1.3e-6 to +2.0e-6, i.e. **20-30% of the final MSE, which fails
  the accuracy bar.** This, not the cost curve, is what fixes the threshold at
  2.5: `tau=2.0` is 1.12x cheaper per sample but is no longer accurate enough.
- `tau = 1.0`: 1.5e-4 to 2.2e-4, 25-40x the entire score.

The oracle column separates the two error sources: the closure error of
freezing a neuron at its mean, which survives as P grows, from the pilot's own
sampling noise, which enters the frozen constants as a fixed offset and so does
not average away over the scored samples. Both are small at `tau >= 2.5`.

---

## 5. End to end

`--mode score`, all 100 official MLPs, N=1e9 reference so `raw_mse` is
leaderboard-comparable. `tau=None` is the exact ablation: the identical code
path with pruning switched off, reproducing plain MC bit for bit
(`tests/test_submission_parity.py::test_sparse_ablation_is_exactly_dense`).

| variant | raw_mse | F/B | C/B | adj@1x | adj@2x | adj@3x | raises |
|---|---|---|---|---|---|---|---|
| `tau=None` n=6200 (ablation) | 8.6466e-06 | 0.0980 | 0.1023 | 8.849e-07 | 9.221e-07 | 9.593e-07 | 0 |
| `tau=2.0` n=8500 | 6.4301e-06 | 0.0824 | 0.0902 | 6.430e-07 | 6.430e-07 | 6.803e-07 | 0 |
| `tau=2.3` n=8500 | 5.9736e-06 | 0.0879 | 0.0963 | 5.974e-07 | 6.256e-07 | 6.757e-07 | 0 |
| **`tau=2.5` n=8500 (shipped)** | **5.6530e-06** | **0.0916** | **0.0999** | **5.653e-07** | 6.120e-07 | 6.592e-07 | 0 |
| `tau=2.7` n=8500 | 5.5954e-06 | 0.0952 | 0.1035 | 5.789e-07 | 6.253e-07 | 6.717e-07 | 0 |
| `tau=3.0` n=8000 | 6.0006e-06 | 0.0947 | 0.1033 | 6.202e-07 | 6.723e-07 | 7.244e-07 | 0 |
| modal fusion `tau=2` n=600 | 1.3275e-04 | — | 0.1409 | 1.785e-05 | — | — | 0 |
| previous ship (blend) | 6.9885e-06 | — | 0.1181 | 8.238e-07 | — | — | 0 |
| blend with the sparse arm | 6.2980e-06 | — | 0.1241 | 7.753e-07 | — | — | 0 |

`adj@kx` recomputes the multiplier with the residual wall time scaled by k.

**The gain replicates across independent draws.** Same suite, four seeds:

| seed | ablation raw | sparse raw | ratio |
|---|---|---|---|
| 0 | 8.647e-06 | 5.653e-06 | 1.53x |
| 1 | 8.433e-06 | 6.171e-06 | 1.37x |
| 2 | 7.858e-06 | 5.919e-06 | 1.33x |
| 3 | 8.224e-06 | 5.256e-06 | 1.56x |
| **mean** | **8.290e-06** | **5.750e-06** | **1.44x** |

The shipped file itself (`--mode ship`, real protocol-3.0 `mlp.seed` per MLP):
raw **5.8050e-06**, F/B **0.0915**, adjusted **5.870e-07** at 1x residual /
6.426e-07 at 2x / 6.982e-07 at 3x, **0 raises in 100**, worst single MLP
4.2164e-05, max C over the suite 0.1335 of B.

### The analytic arm is gone

Pure sparse MC (5.87e-7) beats the blend of sparse MC with the analytic arm
(7.75e-7). The analytic arm costs 2.7e9 FLOPs, which pushes C/B from 0.100 to
0.124, and the 1.24x multiplier penalty exceeds everything the blend buys.
Paying for a second estimator only makes sense below the multiplier floor, and
there is no room below the floor.

### Sizing: `F/B`, and why shrinking N is the wrong hedge

`C = F + 1e11 R`. `F` is machine-independent; `R` is participant wall time and
the grader runs one physical core, so the margin that matters is in `F`. The
shipped point sits at `F/B = 0.0915`, leaving 0.0085 B = 2.3e9 = 23 ms of
residual before the floor is crossed — and crossing it is linear, not a cliff.

The intuitive hedge — cut N so the thing stays under the floor even at 3x
residual — is measurably wrong, because above the floor the adjusted score is
flat in N while below it the score is still falling:

| N | adj @1x | adj @2x | adj @3x |
|---|---|---|---|
| 7,073 (sized to the floor at 3x residual) | 7.12e-07 | 7.12e-07 | 7.12e-07 |
| 8,000 | 6.31e-07 | 6.38e-07 | 6.91e-07 |
| **8,500 (shipped)** | **5.94e-07** | **6.32e-07** | **6.82e-07** |

N = 8500 wins at every residual level. Undershooting costs more than
overshooting.

---

## 6. Honest caveats

- **`tau` is calibrated, not derived.** It was swept on the same 100 official
  MLPs it is scored on. The optimum is broad (2.3 / 2.5 / 2.7 give 5.97 / 5.65
  / 5.79 e-7 adjusted), so the transfer risk is small, but it is real, and the
  private re-evaluation runs on a fresh suite.
- **The mask is derived from the sample stream**, so the estimator is unbiased
  only conditionally on the mask. The pilot draw and the scored draw come from
  the same generator but do not overlap.
- **`P = 150` is the robust choice, not the sharp one.** P = 80/100/150/250 are
  within 3% of each other (realisation noise); P = 600 is clearly worse.
- **Residual risk is one-sided.** Pruning adds ~14 ms of residual over dense on
  the reference machine, from 32 extra sliced matmuls. If the grader's single
  core is much slower the multiplier grows while F does not — hence quoting
  adj@2x and adj@3x rather than adj@1x alone.

## 7. Where this leaves the board

| | adjusted | x floor |
|---|---|---|
| proven noise floor (`docs/floor_theorem.md`) | 4.949e-12 | 1.0 |
| leaderboard #1 (dpskv5) | 3.63e-10 | 73.3 |
| the sign-stable family as a competitor ships it | 1.60e-07 | 32,300 |
| **this repo, now** | **5.87e-07** | **118,600** |
| grader's plain-MC constant `sampling_mse` | 6.4695e-07 | 130,700 |
| this repo, previous ship (blend) | 7.7791e-07 | 157,200 |

First time this repo is below the grader's own plain-MC line — by 1.10x. Still
3.7x behind the competitor entry that motivated the work, and 1,616x behind the
leader. The 4x per-sample cost bar that gated this work **failed** at 1.50x;
the score improved 1.44x anyway, which is why it shipped. Both are in the
ledger.
