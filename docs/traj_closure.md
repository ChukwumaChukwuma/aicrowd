# The closure-accuracy scalar `r`: 2.0 probe-free, 5.0 with a perfect probe, and the probe costs 63% of the budget

**Result. `docs/integrable_cv.md` reduced the deep control-variate programme to
one number — `r`, the factor by which a closure's `rms(E[relu(z^L)] - truth)`
beats the Gaussian closure's — with break-even at `r = 4.4`. Measured on the
same yardstick: today's Gaussian closure `r = 1`, this repository's `kappa_3`
arm `r = 1.41` (published 1.45, reproduced), and jamesrahenry's
trajectory-calibrated moment chain, implemented here, `r = 2.0` probe-free.
The same chain fed the *exact* per-neuron `kappa_3`, `kappa_4` and pair field
reaches `r = 6.7 / 6.1 / 5.7 / 5.0` at `L = 8/16/24/32` and **clears the bar** —
but that input is a Monte-Carlo probe of the target network, and at the
`N = 4096` its author used it bills 63% of our entire clamp budget and delivers
only `r = 2.3-3.4`; at an affordable `N = 1024` (16% of the budget) it is
`r = 1.5-1.9`, worse than free. The probe-free arm therefore stands at `r = 2.0`,
which prices at 1.12x after its own FLOPs, and the gap to break-even is one
specific object: the per-neuron cumulant field, which must be known to
`R^2 > 0.99` to be worth anything (§6.2) and which the analytic diagram
catalogue reproduces at `R^2 = 0.00-0.12` at depth.**

**Method credit.** The trajectory-calibrated moment chain, the DAgger framing
and the error-compensation diagnosis are **@jamesrahenry**'s (AIcrowd discourse
topic 18097, "Stabilizing cumulant propagation at depth 32", graded submission
#314695; MIT replication repository
[`jamesrahenry/arc-whitebox-replication`](https://github.com/jamesrahenry/arc-whitebox-replication)).
His fitting targets and ground-truth cumulant tests use **@keenanpepper**'s
public datasets `keenanpepper/arc-whestbench-higher-moments-2026` and
`keenanpepper/whest-k3-tensors-2026`. Every one of his numbers that we can
reproduce, we reproduce (§8).

Reproducible from `scripts/43_traj_closure.py`
(`--mode ref|aux|baseline|ceiling|fit|stepfit|fidelity|price`); machinery in
`whestfloor/trajclosure.py`. `submission/` untouched.

---

## 0. The yardstick, fixed before any run

`docs/integrable_cv.md` §3.2: a linear control variate in `relu(z^L)` breaks
even against the shipped basis at

```
r  :=  rms(mu_gauss^L - mu_true^L) / rms(mu_new^L - mu_true^L),   mu^L = E[relu(z^L)]
```

`r = 4.2-4.6` at every depth from 6 to 32, and past it the payoff is steep
(1.9x at `r = 4.4`, 5.4x at `r = 10`).

> **Pre-registered bar: `r > 4.4` at `L = 8, 16, 24, 32`.**

Reference: 2,000,000 samples per MLP in **two independent halves**, so
`E[(mt - mA)(mt - mB)] = (mt - m_pop)^2` and no figure here is inflated by the
reference's own noise (2.4e-04 rms per half; 1.4e-04 on the average). Networks
are locally generated MLPs, seeds `960000+`, disjoint from every other page in
this repository and from the official suite. Auxiliary Monte Carlo (`--mode
aux`, 400,000 samples) supplies the exact per-layer `kappa_3`, `kappa_4`,
`E[zc_i^2 zc_j]` and `Cov(relu(z^l))` used as fit targets and as oracle inputs.

## 1. The baselines reproduce

`--mode baseline`, 12 MLPs:

| arm | rms at L=8 / 16 / 24 / 32 | `r` at L=8 / 16 / 24 / 32 |
|---|---|---|
| **Gaussian closure** (the incumbent) | 5.59e-03 / 7.41e-03 / 7.86e-03 / 7.11e-03 | **1.00 / 1.00 / 1.00 / 1.00** |
| all-coincident `(3)`,`(4)` diagrams only | 4.80e-03 / 6.63e-03 / 7.13e-03 / 6.54e-03 | 1.17 / 1.12 / 1.10 / 1.09 |
| **`kappa_3` star** (`cov_prop_edgeworth`) | 3.98e-03 / 5.46e-03 / 5.95e-03 / 5.73e-03 | **1.41 / 1.36 / 1.32 / 1.24** |
| `kappa_3` complete tree catalogue | 4.11e-03 / 5.67e-03 / 6.12e-03 / 5.89e-03 | 1.36 / 1.31 / 1.28 / 1.21 |
| `kappa_3`+`kappa_4` complete tree | 3.94e-03 / 5.53e-03 / 5.99e-03 / 5.78e-03 | 1.42 / 1.34 / 1.31 / 1.23 |

The Gaussian closure is `r = 1` by construction and its absolute rms (7.1e-03
at `L = 32`) is within 12% of the 6.4e-03 `docs/integrable_cv.md` §6 measured
on the official pair. The `kappa_3` star arm lands at **1.41** against that
page's published **1.45**. **The yardstick is the one the payoff curve was
priced against.**

**The complete diagram catalogue is not better than the incomplete one** — the
object `docs/cumulant_expansion.md` §5 spends nine sections building, with
every injectivity correction, both Ursell terms and `kappa_4`, verified against
Monte Carlo to one standard error, scores 1.42/1.34/1.31/1.23 against the
star's 1.41/1.36/1.32/1.24. §2 says why.

## 2. Where the error actually is, and it is not the rectifier

Decomposing the layer-`L` mean residual of the *uncorrected* chain: regress
`mu_true - mu_closure` on `A` — a rich per-neuron basis, `sigma phi(alpha)
alpha^k` for `k <= 7` plus `sigma Phi`, `sigma`, `1` — and then on `A` plus the
**true** state errors, and on `A` plus the **true** cumulants.

| `L` | rms residual | rel. `sigma` error | `R^2` on `A` | `A` + true `dsigma, dm` | `A` + true `gamma_3` | all |
|---|---|---|---|---|---|---|
| 2 | 1.65e-03 | 0.18% | 0.658 | 0.662 | **0.886** | 0.890 |
| 4 | 3.47e-03 | 0.95% | 0.613 | 0.817 | 0.674 | 0.895 |
| 8 | 4.56e-03 | 2.7% | 0.556 | **0.899** | 0.584 | 0.956 |
| 16 | 5.77e-03 | 8.1% | 0.471 | **0.961** | 0.478 | 0.986 |
| 24 | 4.59e-03 | 9.1% | 0.381 | **0.944** | 0.420 | 0.989 |
| 32 | 4.10e-03 | 11.5% | 0.589 | **0.978** | 0.593 | 0.994 |

**Past layer 4 the residual is state drift, not non-Gaussianity of the last
rectification.** Knowing the layer's true cumulants adds 1-4 points; knowing
the state's own `sigma` error adds 35-50 and takes the fit to 94-98%. This is
`docs/cumulant_expansion.md` §9.4 seen from the other side, and it is why §1's
complete catalogue is worth nothing.

It also prices the analytic-cumulant route directly. The analytic per-neuron
`gamma_3` correlates with the true one at **0.83 / 0.69 / 0.32 / 0.35 / 0.29**
at `L = 4/8/16/24/32` (star; the tree catalogue is no better) and its magnitude
is 6-13x too small — the transport deficit of `docs/cumulant_expansion.md`
§9.6, confirmed. The analytic **pair field** `kappa_iij = (W^o2)' diag(kappa_3(x)) W`
is worse: correlation **0.58 / 0.31 / 0.07 / 0.12 / 0.06** and 3-13x too small.

## 3. The trajectory-calibrated closure, probe-free

`--mode fit`. jamesrahenry's construction: per-layer linear corrections to the
post-ReLU mean (8 features), the variance diagonal (6) and the covariance
off-diagonals (2), **fitted sequentially on the chain's own rolled-forward
state** — at layer `l` every training net is advanced with the corrections of
layers `0..l-1` already applied, the design matrix is built at the state
actually visited, and one pooled least squares gives layer `l`'s coefficients.
Training distribution = deployment distribution, which is the whole of DAgger.

**One deliberate departure: no probe.** Five of his eight mean features, three
of six variance features and both off-diagonal features are built from an
`N = 4096` plain-MC probe of the target network's own per-layer cumulants. §7
prices that probe on our budget and it is fatal, so the deployable arm's
cumulants are analytic (star or tree diagrams).

**11 train MLPs, 6 held out**, `r` at L=8/16/24/32:

| arm | `r` at L=8 / 16 / 24 / 32 | min |
|---|---|---|
| `m8` — mean only, his 8 features, `kappa_3` star source | **2.07 / 2.03 / 1.95 / 2.08** | 1.95 |
| `m12` — mean only, 12 features | 2.07 / 2.02 / 1.95 / 2.10 | 1.95 |
| `m8`, complete tree catalogue instead of the star | 2.06 / 2.03 / 1.97 / 2.08 | 1.97 |
| `m8` + diagonal cumulant transport, gain 1 / 4 / 12 | 2.07 / 2.02 / 1.96 / 2.08 | 1.96 |
| `m8` + tree + transport gain 8 | 2.09 / 2.03 / 1.97 / 2.06 | 1.97 |
| `freem` — mean (8) + variance (6) | 2.06 / 1.90 / 1.77 / 1.93 | 1.77 |
| *(the `kappa_3` arm, for scale)* | 1.41 / 1.36 / 1.32 / 1.24 | 1.24 |

**`r = 2.0`, and nothing moves it.** Not the feature set (3, 8 or 12 features),
not the cumulant source (star, complete tree, or either plus the diagonal
transport of `docs/cumulant_expansion.md` §9.6 at any gain from 1 to 12), not
the train size (7 -> 11 nets). Medians track means to 5%. Adding the variance
correction makes it *worse*, which is §4's mechanism showing through. Same
plateau jamesrahenry reports ("neither more capacity nor 2.4x more data moves
it"), reached from the probe-free side.

## 4. The compensation mechanism, reproduced

`--mode ceiling`, 11 MLPs. Overwrite part of the state with Monte-Carlo truth
at every layer boundary and read the closure's own (uncorrected) mean:

| oracle | `r` at L=8 / 16 / 24 / 32 |
|---|---|
| `E[relu(z^l)] <- truth` (a perfect **mean** correction) | **2.75 / 4.59 / 5.50 / 5.96** |
| `Var(relu(z^l)) <- truth` | **0.60 / 0.42 / 0.35 / 0.29** |
| `Cov(relu(z^l)) <- truth` | 0.83 / 0.77 / 0.75 / 0.74 |
| mean + variance | 1.31 / 1.30 / 1.31 / 1.24 |
| **mean + full covariance** = the exact state, one closure step | 2.61 / 4.53 / 6.20 / **7.34** |

**Re-anchoring the covariance to truth makes the chain worse**, by 1.4-3.5x —
his sec 1(b) result reproduced on our networks with an independently written
chain (his: plain 6.0e-5, variances-only 1.8e-4, full covariance 1.0e-4). And
fixing mean *and* variance together (1.24) is nearly five times worse than
fixing the mean alone (5.96). **The covariance error is what compensates the
mean error downstream.** That is why `freem` loses to `m8` in §3, and it is the
single most important structural fact about these chains.

The last row is the **uncorrected** closure's cap: exact previous-layer moments,
one Gaussian closure step, `r = 7.34` at `L = 32` against jamesrahenry's 7.2 for
the same construction. It is not a cap on the whole programme, because a fitted
correction sits on top of it — §6 measures what that correction can add.

## 5. What the perfect probe buys, and what it costs

`--mode fit --cum oracle|probe`, 7 train / 4 held out. `oracle` feeds the exact
`kappa_3`, `kappa_4` and pair field from a 400,000-sample reference — this is
jamesrahenry's "perfect-cumulant control" at infinite probe size.

| cumulant source | probe cost, % of the `0.1B` clamp budget | `r` at L=8/16/24/32 | min `r` | verdict |
|---|---|---|---|---|
| analytic (star diagrams) | **0** | 2.07 / 2.03 / 1.95 / 2.08 | 1.95 | **FAIL** |
| MC probe, `N = 1024` | 15.8% | 1.60 / 1.52 / 1.85 / 1.48 | 1.48 | **FAIL** |
| MC probe, `N = 4096` (his setting) | **63.2%** | 2.29 / 2.43 / 3.38 / 2.95 | 2.29 | **FAIL** |
| MC probe, `N = 16384` | 252.6% | 3.53 / 4.12 / 4.57 / 3.79 | 3.53 | **FAIL** |
| **exact — infinite probe** | unaffordable | **6.72 / 6.13 / 5.75 / 4.97** | **4.97** | **PASS** |

The probe curve is the whole argument in one column: `r` reaches the bar only
between `N_p = 16384` and infinity, and `N_p = 16384` already bills 2.5x our
entire budget.

Two things.

**With exact cumulants the method clears the bar.** `r = 4.97-6.72` against a
break-even of 4.4, and it is the *covariance* features that deliver it — at
oracle cumulants, mean-only is 2.05/2.12/1.99/1.97, mean + off-diagonal is
4.68/3.30/3.13/2.57, and mean + variance + off-diagonal is 6.72/6.13/5.75/4.97.
jamesrahenry's "T1 is irreplaceable" is confirmed, and so is the reversal: the
variance correction *helps* by 1.9-2.6x once the cumulants are exact, and
*hurts* when they are analytic.

**And the probe that would supply them is unaffordable.** A plain forward probe
of `N_p` samples bills `N_p x 4.19e6` FLOPs; our whole budget at the multiplier
clamp is `0.1B = 2.72e10`, so `N_p = 4096` is **63% of it** — it would cut `N`
by 63%, i.e. 2.7x of variance, against a 1.5x gain. `N_p = 1024` fits (16%) and
is *worse than free*: probe noise in the cumulants is more damaging than the
analytic estimate's bias, which is §4's mechanism again (the chain tolerates
variance badly once it is inside a fitted correction). The probe is dead on
cost, and it is separately dead on principle: `docs/integrable_cv.md` §3.3 shows
a control-variate mean estimated from `N_p` fresh samples contributes `V/N_p` to
the variate's variance in full, and at `N_p = 4096` the plain average of the
probe's own samples (1.8e-03) is already *better* than what the chain makes of
them (2.1e-03).

## 6. So the whole thing reduces to the cumulant field

### 6.1 It is the entire remaining content

`--mode stepfit`: anchor the state to truth at every layer so the chain
contributes nothing, then fit the richest per-neuron mean correction the family
allows (7 train MLPs, 4 held out).

| basis for the per-layer correction | one-step gain at L=8 / 16 / 24 / 32 |
|---|---|
| degree-5 polynomial in `alpha` x `sigma phi`, plus `sigma Phi`, `sigma`, 1 | 1.66 / 1.45 / 1.33 / 1.14 |
| the same, with per-**MLP** coefficients (an oracle) | 2.03 / 1.69 / 1.52 / 1.34 |
| the same + **true** per-neuron `kappa_3`, `kappa_4` Edgeworth terms | **9.8 / >10 / >7.4 / >5.5** |

(`>` marks a residual that fell below the reference's own 1.4e-04 noise floor.)
Giving the correction per-MLP freedom buys 3-20%; giving it the true cumulants
buys an order of magnitude. **The per-neuron cumulant field is not a function
of the closure's own state, and it is the only thing missing.**

### 6.2 How accurately would it have to be known?

`--mode fidelity`: degrade the true `kappa_3`/`kappa_4` field to a controlled
`R^2` of its own variance and read the one-step gain back (6 train / 3 held
out).

| `R^2` of the field | gain L=8 | L=16 | L=24 | L=32 |
|---|---|---|---|---|
| 0.50 | 1.78 | 1.43 | 1.31 | 1.14 |
| 0.80 | 1.91 | 1.46 | 1.32 | 1.14 |
| 0.90 | 2.27 | 1.54 | 1.35 | 1.15 |
| 0.95 | 2.58 | 1.74 | 1.42 | 1.18 |
| **0.99** | **5.29** | **2.95** | 1.80 | 1.44 |
| 0.999 | (floor) | 8.85 | 5.12 | 2.82 |

**Flat until `R^2 = 0.95`, and it only takes off past 0.99.** Against that:

| predictor of the per-neuron `gamma_3` | `R^2` at depth |
|---|---|
| the `kappa_3` star / complete tree diagram source | **0.00 - 0.12** |
| a 3-factor model on the top eigenvectors of `Cov(z^L)` | 0.78 - 0.81 |
| a degree-5 polynomial in `alpha` | 0.84 - 0.87 |

The gap between what is predictable (0.87) and what is needed (0.99) **is** the
problem, and it is `docs/cumulant_expansion.md` §11 item 8 — cumulant
*transport* — restated as a measurement instead of a conjecture. The pair field
is harder still: analytic `R^2 <= 0.12` at depth against the same requirement.

## 7. What the numbers are worth on the score

`--mode price` reproduces `docs/integrable_cv.md` §3.1 exactly on the local
ladder (1.026x / 1.171x / 1.901x / 5.423x / 24.809x at `r = 1/2/4.4/10/30`),
then applies the same objective to the official ladder and charges the
closure's own FLOPs where they are paid: at the multiplier clamp the sample
budget is `0.1B - F_closure`, so `V/N` inflates by `1/(1 - F_closure/0.1B)`.
The analytic chain bills ~`4.0e9` FLOPs (per layer: 2 matmuls for `W' C_h W`,
1 for the `kappa_3` star, 1 for the analytic pair field; x32), i.e. **14.7% of
the clamp budget**.

| `r` | what it is | best `x ship`, free | closure charged |
|---|---|---|---|
| 1.00 | the Gaussian closure | 1.10x | 0.94x |
| 1.24 | the shipped `kappa_3` arm | 1.13x | 0.98x |
| **2.03** | **this page's probe-free trajectory-calibrated closure** | **1.27x** | **1.12x** |
| 2.95 | the same with an `N = 4096` probe (which also costs 63% of `N`) | 1.53x | ~0.6x |
| **4.4** | **break-even** | 2.15x | 1.99x |
| 4.97 | the same chain with EXACT cumulants | 2.45x | 2.28x |
| 5.96 | perfect mean re-anchoring | 3.01x | 2.85x |
| 7.34 | the exact state every layer, uncorrected closure | 3.98x | 3.79x |

All at `L = 32`, the argmax at every `r`. **The deployable arm is 1.12x and the
prize behind the cumulant field is ~2.3x.**

## 8. Where our numbers meet jamesrahenry's

| his measurement | his value | ours |
|---|---|---|
| mean-only reset, final MSE vs plain chain | 6.0e-5 -> 1.64e-6, `r = 6.05` | `r = 5.96` |
| one step with true input moments | 1.16e-6 vs 6.0e-5, `r = 7.2` | `r = 7.34` |
| re-anchoring the covariance makes it worse | 1.0e-4 vs 6.0e-5 | `r = 0.74` |
| re-anchoring the variances makes it worse | 1.8e-4 vs 6.0e-5 | `r = 0.29` |
| trajectory-fitted, `N = 4096` probe | 7.4-8.1e-6 vs 6.3-7.6e-5 baseline, 8.4-9.3x on MSE | `r = 2.29-3.38` |
| the family plateaus against capacity and data | quad head = lin head; 2.4x data does not move it | 3/8/12 features all `r = 2.0` |

Different networks, an independently written closure, and a probe-free
variant — and every one of them lands on his number.

**On the headline.** His "~8.4x better than plain kprop" is on **MSE**; `r` is
on **rms**, so it transfers as `r = 2.9`, which is exactly where
`scripts/11`'s exact-cumulant oracle already sat and below the 4.4 break-even.
Measured directly in our setting it is `r = 2.3-3.4` at his probe size.

## 9. Bars, fixed before each run

| # | bar | measured | verdict |
|---|---|---|---|
| 1 | the Gaussian closure reproduces at `r = 1` and the `kappa_3` arm at the published 1.45 | 1.000 and **1.41** at `L = 8` | **PASS** |
| 2 | **the probe-free trajectory-calibrated closure reaches `r > 4.4` at `L = 8,16,24,32`** | **2.07 / 2.03 / 1.95 / 2.08** | **FAIL** |
| 3 | ...with an affordable probe (`N_p <= 1024`, 16% of the clamp budget) | 1.60 / 1.52 / 1.85 / 1.48 — worse than free | **FAIL** |
| 4 | ...with the probe size its author used (`N = 4096`, 63% of the budget) | 2.29 / 2.43 / 3.38 / 2.95 | **FAIL** |
| 5 | ...with `N = 16384` (253% of the budget) | 3.53 / 4.12 / 4.57 / 3.79 | **FAIL** |
| 6 | ...with EXACT cumulants (infinite probe) | **6.72 / 6.13 / 5.75 / 4.97** | **PASS, not deployable** |
| 7 | the exactness boundary is asserted, not assumed | `tests/test_trajclosure.py`: `E[relu(z^1)]` and `(E[z^2], Var(z^2))` within 2 MC sigma, `E[relu(z^2)]` beyond 5; the star diagram matches `whestfloor.cumulants` to 1e-12; `ChainDriver` reproduces `chain` to 1e-12 for all three cumulant sources; the deployable pack is bitwise RNG-independent | **PASS** |
| 8 | shipped estimator untouched | `git diff submission/` empty | **PASS** |

**The answer to "is a trajectory-calibrated closure accurate enough to supply
the mean of a deep control variate" is: not from the analytic side (`r = 2.0`,
1.12x), not from an affordable probe (`r = 1.5`, worse), and yes from an
infinitely accurate one (`r = 5.0`, 2.3x).** What separates them is a single,
precisely specified object — the per-neuron `kappa_3`/`kappa_4` field and the
pair field `E[zc_i^2 zc_j]`, needed to `R^2 > 0.99`, available analytically at
`R^2 <= 0.12` at depth. Closing that is the cumulant-transport problem of
`docs/cumulant_expansion.md` §11 item 8, whose rank-32 Tucker surrogate costs
0.13 matmul units a layer — under 1% of the budget — and which nothing in this
repository or in the published prior art has yet solved. **That, and not the
control-variate basis, is where the next 2.3x is.**
