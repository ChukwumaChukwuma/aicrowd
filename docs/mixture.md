# The Gaussian-mixture route: the ceiling of `sum_k w_k relu_mean(m_k, s_k)`

**Result. TODO-HEADLINE**

Reproducible from `scripts/47_mixture.py` (`--mode ceiling|anatomy|propagate`);
machinery in `whestfloor/mixture.py`; ledger rows via
`scripts/48_record_mixture.py`. `submission/` untouched.

---

## 0. Why the hypothesis deserved a test

Two routes to the top of the leaderboard are closed with bounds.
`docs/cost_floor.md`: no sampler in the family reaches dpskv5's raw 3.6e-9 at
its billed 2.43e10 FLOPs — a sampler at that budget scores `v/N = 7.8e-6`,
2,160x worse. `docs/traj_closure.md`: no moment closure reaches it either — the
trajectory-calibrated chain is `r = 2.0` deployable and `r ~ 5-7` even with
*exact* cumulants, i.e. ~36x on MSE against the ~2,500x required.

What is left is a deterministic method costing a handful of covariance
propagations, and a **K-component Gaussian mixture** is exactly that shape: each
component is propagated by the exact rectified-Gaussian machinery
(`whestfloor/relu_moments.py`, Mehler), so there is no *moment closure* per
component; only representation error, controlled by `K`.

**The budget, re-priced against a measured propagation.** One Mehler covariance
propagation over 32 layers of width 256, written against flopscope and billed in
a real `BudgetContext`, is **3.24e9 FLOPs** — three times the 1.07e9 an analytic
matmul count suggests, because the einsum, the transcendentals (which
`flops.stats.norm` promotes to float64 and bills at 2x) and the Mehler series
are all charged. Each additional component is **2.364e9**. So:

| entry | billed `F` | `F/B` | measured propagations | raw MSE |
|---|---|---|---|---|
| huang_chung_yi | 6.30e9 | 0.023 | **1.9** | 9.0e-9 |
| dpskv5 | 2.43e10 | 0.089 | **7.5** | 3.6e-9 |
| this repo, shipped | 6.22e10 | 0.229 | 19.2 | 1.0754e-6 |

Both leaders sit under the `0.1` multiplier clamp, and both are within a factor
of two of "a small Gaussian mixture". The hypothesis is well posed.

## 1. The form is its own ceiling

Whatever produces the components, a mixture's answer is

```
E[relu(z^32_j)]  ~=  sum_k w_k relu_mean(m_kj, s_kj).                       (*)
```

So the family can be bounded before anything is built: take the **exact** law of
`z^32` from Monte Carlo, cut it into `K` cells along a chosen `r`-dimensional
frame, and evaluate `(*)` with the **true** per-cell conditional mean and
standard deviation. No propagation error, no closure error, no fitting is
charged — the state is perfect by construction. That bounds every mixture whose
components are indexed by those `r` coordinates and Gaussian in the complement,
which is precisely the construction under test.

Three things make this measurable where `scripts/06` (which reported one row of
it, 2.9x at `k = 3`, and killed the idea prematurely) could not:

* **Lloyd cells, not product bins.** `K` then means *components* — the thing you
  pay for — rather than bins per axis. In one dimension Lloyd-Max is the optimal
  `K`-node quantiser, so a ceiling measured this way bounds one measured with
  fixed Gauss-Hermite nodes.
* **Paired same-stream residuals.** An independent Monte-Carlo reference cannot
  resolve a 1e-8 ceiling: its standard error is `(1/n) sqrt(2 d'Cd + ||C||_F^2)`,
  which at 1e6 reference samples is 2.6e-8 — larger than the quantity. Instead
  each of two independent streams produces both the predictor *and* its own
  sample truth; the leading Monte-Carlo fluctuation cancels inside each
  difference, and the product of the two differences is unbiased for the squared
  model error. Measured: `N/4`, `N/2` and `N` agree to 1-2%, against ~30% for an
  independent reference at 8x the sample cost.
* **Three frames, because variance is not non-Gaussianity** (§2): top-`r`
  eigendirections of `Cov(z^32)`, FastICA directions of maximal
  `|excess kurtosis|`, and input-side directions — which is what a mixture that
  splits at layer 1 actually conditions on.

TODO-CEILING-TABLE

## 2. Why it stops there: three measurements

`--mode anatomy`, 4 MLPs x 150,000 samples.

### 2.1 The closure error is not concentrated in the top eigendirections

Cumulative share of the squared error carried by the leading eigendirections of
`Cov(relu(z^32))`, for the closure fed the **exact** mean and covariance:

| directions | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 256 |
|---|---|---|---|---|---|---|---|---|
| oracle closure | 0.007 | 0.014 | 0.030 | 0.068 | 0.127 | 0.221 | 0.389 | 1.000 |
| analytic chain | 0.843 | 0.851 | 0.860 | 0.903 | 0.933 | 0.963 | 0.987 | 1.000 |
| isotropic | 0.004 | 0.008 | 0.016 | 0.031 | 0.062 | 0.125 | 0.250 | 1.000 |

The rectification error is **1.6-1.9x isotropic**: it is spread over all 256
directions. (The brief's premise that it is concentrated in the top
eigendirection, with per-direction shrinkage `q_1 = 3.4-7.1` cited to
`docs/traj_closure.md` §5, is not in that document — §5 prices the Monte-Carlo
probe.) What *is* concentrated is the **chain's** error, 84.3% in the top
direction alone — but that is state drift along the dominant mode, a different
object, and §3 shows the mixture does not remove it.

### 2.2 The non-Gaussianity is low-rank, in a different subspace than the variance

| quantity | value |
|---|---|
| participation ratio of the variance | 3.1 |
| participation ratio of `\|excess kurtosis\|` | 9.2 |
| participation ratio of `\|skewness\|` | 14.2 |
| top-1 eigendirection, share of the variance | **0.562** |
| top-1 eigendirection, share of `\|excess kurtosis\|` | **0.000** |

The last row holds on all four MLPs. **The top principal component of `z^32` is
the most Gaussian direction in the space**, and for a structural reason: it is a
weighted sum over ~256 neurons, hence the most CLT-averaged object available.
The non-Gaussianity genuinely is low-rank — 9 directions — but not those nine.

This is the mechanism, and it is a two-sided requirement. A conditioning
direction is only useful if it is *both*

* high-variance, so that conditioning actually narrows each neuron's marginal
  and moves `relu_mean`, and
* high-non-Gaussianity, so that there is something for the mixture to resolve.

Here those two sets are disjoint. Conditioning on the top eigendirection shrinks
a neuron's conditional standard deviation by only `1/sqrt(1 - 0.562) = 1.5x`;
conditioning on the top kurtosis direction shrinks it by almost nothing at all.

### 2.3 The readout form is not the wall — the sharing is

Give **every neuron its own** optimal `K`-cell 1-D mixture, on its own
pre-activation. No shared mixture can do this, because one partition has to
serve 256 different kinks; it is a strictly stronger oracle:

| `K` per neuron | 2 | 4 | 6 | 16 | 64 |
|---|---|---|---|---|---|
| raw MSE | 2.08e-6 | 2.47e-7 | 8.59e-8 | 8.73e-9 | 4.45e-10 |
| rms | 1.44e-3 | 4.97e-4 | 2.93e-4 | 9.34e-5 | 2.11e-5 |

So `(*)` reaches dpskv5's 4e-9 at about **24 components per neuron**. The form
is capable. What kills the mixture is that one global partition cannot be 256
local ones at once.

## 3. The deployable propagator

`--mode propagate`, 6 MLPs, reference 2 x 400,000 samples in independent halves,
every `F` billed in a real `flopscope.BudgetContext` on the same kernel that
produced the answer. `mixture_kernel` (flopscope, float32) and
`mixture_predict` (NumPy, float64) agree to rms 2e-6, and at `K = 1` the kernel
reproduces the float64 analytic chain to rms 4.5e-9, so the billed code is the
measured code.

| nodes | `r` | split | resplit | `K` | raw MSE | x chain | `F` | `F/B` |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | 1 | 7.4121e-05 | 1.00 | 3.236e9 | 0.0119 |
| 2 | 1 | 8 | 0 | 2 | **5.6395e-05** | **1.31** | 5.611e9 | 0.0206 |
| 4 | 1 | 8 | 0 | 4 | 5.6943e-05 | 1.30 | 10.338e9 | 0.0380 |
| 6 | 1 | 8 | 0 | 6 | 5.6902e-05 | 1.30 | 15.065e9 | 0.0554 |
| 12 | 1 | 8 | 0 | 12 | 5.6920e-05 | 1.30 | 29.247e9 | 0.1075 |
| 6 | 1 | 16 | 0 | 6 | 6.7549e-05 | 1.10 | 10.958e9 | 0.0403 |
| 6 | 1 | 0 | 0 | 6 | 7.3154e-05 | 1.01 | 19.167e9 | 0.0705 |
| 6 | 1 | 24 | 0 | 6 | 7.3111e-05 | 1.01 | 6.850e9 | 0.0252 |
| 3 | 2 | 16 | 0 | 9 | 4.8752e-04 | 0.15 | 15.618e9 | 0.0574 |
| 3 | 1 | 0 | 4 | 3 | 5.6474e-05 | 1.31 | 9.612e9 | 0.0353 |
| 2 | 1 | 0 | 8 | 2 | 5.7641e-05 | 1.29 | 5.611e9 | 0.0206 |

**Bar `raw < 1.0e-6`: FAIL at 5.6395e-05, by 56x.** Adjusted 5.64e-6 against
the shipped 1.08e-7 — 52x worse than what is already submitted.

Three facts, none of them a tuning problem:

* **It saturates at `K = 2`.** `K = 2, 4, 6, 12` agree to 1%. The ~6
  propagations the telemetry allows are not the constraint; two already exhaust
  the mechanism.
* **The split point has an interior optimum at layer 8**, and splitting `z^1` —
  the one law that is exactly Gaussian, hence the tempting place — is worth
  1.01x. The closure covariance's participation ratio runs
  127 / 101 / 74 / 47 / 29 / 15 / 5.8 / 6.5 at `L = 1,2,3,5,9,17,25,32` with the
  top eigendirection at 1.5% of the trace at `L = 1` and 34% at `L = 32`: early,
  there is no direction to condition on; late, there is no depth left for the
  components to diverge in. Split at layer 31 reproduces the chain to 1e-16, as
  it must.
* **`r = 2` is catastrophic, 0.15x.** Two successive splits deflate two
  directions out of every component's covariance. The mixture's *total*
  covariance is still exactly right — that is the quadrature identity, pinned in
  `tests/test_mixture.py` — but each component now rectifies as if it were far
  more certain than it is, and `relu_mean` is convex in `s`.

The comparison that matters: **huang_chung_yi bills 6.30e9 FLOPs, between our
`K = 1` (3.236e9) and our `K = 2` split@8 (5.611e9), and scores raw 9.0e-9. At
that budget the best mixture here is 5.64e-05 — 6,270x worse.**

## 4. Bars, fixed before each run

TODO-BARS

## 5. What this leaves, and what it specifies for the next attempt

TODO-OPEN
