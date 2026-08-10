# Randomised quasi-Monte Carlo, and the number that decides it

**Verdict in four lines.**

| | | |
|---|---|---|
| pre-registered rate bar | `p >= 1.40` | **FAIL** — measured `p = 1.043 ± 0.023` against iid's control `0.974 ± 0.034` |
| pre-registered redundancy bar | `>= 1.20x` on top of the head | **PASS** — 1.418x, and 1.539x if the head comes off |
| headline bar | graded adjusted `< 2.4646e-07` | **PASS (projected)** — `1.727e-07`, 1.427x, from the local paired ratio |
| what it is | | a **~2x constant**, not a rate change |

So: **ship it, and stop looking for the rate here.** A randomly-shifted rank-1
lattice buys a real, cheap, unbiased 1.43x on the adjusted score — the largest
single win left in the sampler — and it does *not* explain the metered
leaderboard frontier, because on this integrand its convergence exponent is 1.04,
not 1.7. §2.1 shows why no generating vector can fix that: the obstruction is the
integrand's ANOVA spectrum, not the point set.

Two corrections to the rest of the repository fall out and are worth having on
their own: `docs/floor_theorem.md`'s 1.38x ceiling for this row is **exceeded**
(measured mean 2.00x), and `docs/cost_floor.md`'s `adjusted = v_eff·c/B` identity
is a `p = 1` statement and is now scoped to iid samplers.

Construction credit: **evaaaz** (forum 18053). The redundancy warning and the
antithetic refutation that shape how it is measured here: **radiant-allomancer**
(18085), whose "the two levers are partly redundant" is the single most useful
sentence anyone published about this method. The float64 inverse-CDF trap:
**jamesrahenry** (18097 erratum) and **mohanty** (18125).

---

## 1. The premise, re-anchored on the 2026-08-07 re-grade

The case for building this was a fitted convergence exponent. Take
`N_eq = F / 4.198656e6` (billed FLOPs in dense-forward-pass equivalents), assume
`raw = v / N_eq^p` with `v = 0.045`, and solve for `p` at each entry's single
graded point. **The board was re-graded on 2026-08-07 and the submissions the
original fit used no longer exist** — dpskv5 4.0e-10 -> 5.43e-08,
huang_chung_yi 9.0e-10 -> 1.159e-07, joe_wanza 1.0e-09 -> 4.87e-08 — so the
`p = 2.109 / 1.886 / 1.742` that opened this line are discarded. Current rows,
with the instrumented share `F/C`:

| who | `F/C` | `F` | `N_eq` | raw MSE | `p_implied` | |
|---|---|---|---|---|---|---|
| **rayan53** | **0.957** | 2.687e10 | 6,399 | **1.349e-08** | **1.714** | new #1, metered |
| ednacob | 0.930 | 1.283e11 | 30,546 | 3.633e-08 | 1.359 | metered |
| oabuod | 0.885 | 1.517e11 | 36,138 | 1.165e-07 | 1.226 | metered |
| us (shipped) | 1.000 | 6.221e10 | 14,817 | 9.285e-07 | 1.123 | metered |
| dpskv5 | 0.000 | 2.111e06 | 0.5 | 7.840e-08 | — | unmetered, under review |
| joe_wanza | 0.000 | 1.296e07 | 3.1 | 5.211e-08 | — | unmetered, under review |

`p_implied = ln(v/raw)/ln(N_eq)` is **not identifiable** — one point cannot
separate `v` from `p`, and the statistic returns a number for a deterministic
estimator just as happily as for a sampler. The bottom two rows are the
demonstration: at `N_eq = 0.5` and `3.1`, a sampler with `v = 0.045` scores 0.09
and 0.014, so those entries are 1e6x better than any sampler can be at that
compute and their error is model error, not sampling error. This page's earlier
revision flagged exactly those two from `F` alone, before the re-grade; the
re-grade then put them at `F/C = 0.000` and under organizer review. Keep the
diagnostic, discard the conclusion it was used for.

### 1.1 What a single point *can* support: is `p = 1` feasible at all?

Ask a different question, one that needs no assumption about `v`.
`F` is known, so choosing a per-sample cost `c` **forces** both the sample count
and the per-sample variance from the entry's own numbers:

    N = F / c        v_eff = raw * N

Then close the box from both sides with two figures already measured in this
repository:

* `docs/cost_floor.md` §3 — an **oracle** Gaussian surrogate replacing a prefix
  of the network has an irreducible `b^2`: 9.04e-07 at 1 layer-equivalent per
  sample, 3.25e-06 at 3, 3.77e-06 at 4. So a cheap per-sample pass carries a
  bias floor and `raw >= b^2`.
* `docs/hermite_rank_ceiling.md` — the layer-1 Hermite family caps variance
  reduction at 1.76x, and the *dictionary-free* bound (top-8 eigenfunctions of
  `Cov(y)`, which nothing realises) at 10.1x.

`scripts/52_leader_rate.py --mode feasible` runs that squeeze over the cost
ladder. For **rayan53** every rung fails:

| `c` (layer-eq) | `N` | `v_eff` forced | x reduction needed | `b^2` floor | |
|---|---|---|---|---|---|
| 32 (full pass) | 6,406 | 8.64e-05 | **519.5** | 0 | reduction 51x past the ceiling |
| 21.7 (our sparse) | 9,447 | 1.27e-04 | 352.3 | 4.81e-06 | and `b^2` 356x its raw |
| 13 | 15,769 | 2.13e-04 | 211.1 | 8.87e-06 | |
| 4 | 51,250 | 6.91e-04 | 64.9 | 3.77e-06 | |
| 1 | 205,002 | 2.77e-03 | 16.2 | 9.04e-07 | still 1.6x past, `b^2` 67x its raw |

The two constraints move in opposite directions — cheap samples buy the variance
reduction but import a bias floor far above the target raw — and they never both
hold. **So `p = 1` is refuted for rayan53, and for ednacob (40.4x needed at the
full pass) and oabuod (10.7x, marginally).** The same test says `p = 1` **is**
feasible for us, at 3.3x, which is right: we are an iid sampler.

That is the honest version of the brief's claim, and it survives. Something
structural is happening on the metered frontier that iid sampling cannot do.
What §2 settles is whether **an input-space lattice is that something.** It is
not.

## 2. The rate, measured over seven doublings: `p = 1.04`, and the bar FAILS

`scripts/50_rqmc_rate.py --mode rate`. Six official MLPs; dense forward pass, so
no mask and no pilot and the measured quantity is the sampler's own variance and
nothing else; `N` = the primes just under `2^10 … 2^17`; variance taken across
independent randomisations, which for an unbiased estimator **is** its MSE, so no
ground truth and no reference noise enter. Both arms share MLPs and rep seeds.
A fresh order-2 CBC generating vector is searched at every `N` (the criterion is
`N`-dependent), 32 to 8 randomisations per cell as `N` rises.

| `N` | iid variance | lattice variance | ratio |
|---|---|---|---|
| 1,021 | 4.13156e-05 | 2.58057e-05 | 1.601 |
| 2,039 | 1.98850e-05 | 1.10343e-05 | 1.802 |
| 4,093 | 1.03293e-05 | 5.12850e-06 | 2.014 |
| 8,191 | 4.72503e-06 | 2.73890e-06 | 1.725 |
| 16,381 | 2.61396e-06 | 1.22708e-06 | 2.130 |
| 32,749 | 1.36700e-06 | 5.82766e-07 | 2.346 |
| 65,521 | 6.20460e-07 | 3.57861e-07 | 1.734 |
| 131,071 | 3.83334e-07 | 1.44745e-07 | 2.648 |

Fitted `v = v_0 / N^p`, standard errors from 4,000 bootstrap resamples over
randomisations **and** MLPs jointly:

| arm | `p` | ± se | `v_0` | `r²` |
|---|---|---|---|---|
| iid — **the control** | **0.974** | 0.034 | 3.347e-02 | 0.9984 |
| randomly-shifted rank-1 lattice | **1.043** | 0.023 | 3.223e-02 | 0.9977 |

```
pre-registered bar        p >= 1.40
measured                  p  = 1.043 +- 0.023
                          FAIL, by 15 standard errors
```

The iid arm returning `0.974 ± 0.034` is the one number here whose value was
known in advance, and it lands on 1. So the harness is calibrated and the
lattice's 1.043 is a real reading.

The rate difference is nonzero but tiny. Fitting the **ratio** directly, which
cancels the shared MLP-to-MLP spread:

```
p_lattice - p_iid = 0.0686 +- 0.0307        (2.2 sigma -- detectable, and useless)
ratio at N = 1,021    1.671
ratio at N = 25,000   2.081
ratio at N = 131,071  2.331
```

**So the lattice is a ~2x constant with a 0.07 exponent riding on it, not a rate
change.** Extrapolating that ratio to rayan53's operating point buys 2.1x where
the gap is 291x.

### 2.1 Why 1.04 and not 1.7 — the mechanism, and why no generating vector fixes it

A rank-1 lattice with `gcd(z_j, N) = 1` has **exact** `N`-point equidistribution
in every one-dimensional projection, so the shift-averaged squared error of any
*first-order* ANOVA term is `1/(6N²)` against Monte Carlo's `1/N`. That happens
for **any** generating vector — which is why the CBC search only ever buys the
pairs, and why no better vector can rescue the exponent.

`docs/floor_theorem.md` measures the first-order share of `Var(relu z³²)` at
`f_1 = 27.6% ± 1.6%`, with the remaining 72.4% at mean ANOVA order **15.5** out
of a nominal 256. So the variance decomposes as

    v(N)  =  (1 - f_1) sigma^2 / N   +   f_1 sigma^2 c / N^2

The second term dies, the first does not, and the ratio tends to
`1/(1 - f_1) = 1.38x` with `p -> 1`. The measurement is *better* than that —
mean ratio 2.00, rising to 2.65 — so the lattice is also reaching part of the
order-2 mass, and `docs/floor_theorem.md`'s 1.38x prediction for this row is
exceeded. But it is exceeded **as a constant**. The integrand is Lipschitz with
8,192 kink surfaces, not a member of the mixed-derivative Sobolev space the
`O(N^{-2+ε})` lattice theorems need, and its effective dimension in the input
coordinates is 11.5–15.5, not 2 or 3.

**The consequence for the search.** `p = 1.7` on this problem is not reachable by
choosing a better 256-dimensional point set, because the obstruction is the
integrand's ANOVA spectrum and not the point set's quality. If the metered
frontier really is at an effective `p` of 1.7 — and §1.1 shows it is doing
*something* iid sampling cannot — then the mechanism is not an input-space
lattice. The families §1.1 leaves open are a deterministic estimator whose error
is model error (in which case `p_implied` is not a rate at all) and
`docs/floor_theorem.md`'s Route 2, the offline-learned map, which the low-order
barrier does not bound.

## 3. Cost: the lattice draw is 1.51% of the pass, and the float64 trap is shut

`scripts/51_rqmc_deploy.py --mode cost`, real `BudgetContext`, official MLP,
`tau = 2.5`, `P = 225`, through the **shipped** `corrected_sparse_kernel` with
the draw swapped by its new `x0_fn` hook:

| `N` | iid `F` | lattice `F` | delta | per element | x iid |
|---|---|---|---|---|---|
| 4,093 | 11,937,103,496 | 12,101,609,608 | 164,506,112 | 157.0 | 1.0138 |
| 8,191 | 22,859,449,622 | 23,188,662,550 | 329,212,928 | 157.0 | 1.0144 |
| 16,381 | 44,688,150,152 | 45,346,535,560 | 658,385,408 | 157.0 | 1.0147 |

```
marginal FLOPs/sample, iid     2,665,287
marginal FLOPs/sample, lattice 2,705,479
ratio                          1.0151      <- 2.00 would mean float64
draw overhead                  40,192 FLOPs/sample = 1.51% of the pass
```

**A8, and it has already cost another team 2x.** `flopscope.stats.norm.ppf`
promotes float32 to float64 to match scipy, and a single promoted array reprices
the whole 32-layer chain at the float64 rate. The ppf must run in float64 —
that is what stops a float32 uniform of exactly 1.0 from returning `inf` — and
the result must be cast back to float32 *immediately*. The 1.0151 above is the
proof that it was: the entire overhead is 157 FLOPs per element of the draw
(166 of which is the ppf itself, against 16 for `standard_normal`), and it is
proportional to the **draw**, not to the pass. `tests/test_rqmc.py::
test_billed_lattice_stays_float32_downstream` pins that per-element figure so a
future edit cannot quietly lose it.

The point set itself is data-independent, so the submission builds it once in
`setup` where it is free; billed it would be 8 FLOPs/element.

---

## 4. Unbiasedness

Let `P = {p_i} ⊂ [0,1)^d` be any fixed point set and `U ~ U[0,1)^d`. For each
fixed `i` and coordinate `j`, `t ↦ frac(p_ij + t)` is a measure-preserving
rotation of the circle, and the coordinates of `U` are independent, so

    u_i = frac(p_i + U) ~ U[0,1)^d       exactly, every i, every N

hence `x_i = Φ^{-1}(u_i) ~ N(0, I)` exactly and `E[(1/N) Σ f(x_i)] = E[f(x)]`
for any integrable `f`. No asymptotics, no lattice property used. Only the
*dependence between* points is structured, and dependence does not move a mean.
Composing with an orthogonal `Q` is also exact, because `Qx ~ N(0,I)` — which is
what makes the dimension **rotation** of §5 free and legal.

This is what keeps `docs/floor_theorem.md` applicable: the lattice estimator is
still unbiased, so its MSE is still variance and the `v̄/N` floor still binds.

Checked, not asserted (`scripts/50_rqmc_rate.py --mode unbiased`,
`tests/test_rqmc.py`): every one-dimensional projection is the exact `N`-point
grid to `< 1/N`; a fixed lattice row's marginal over shifts passes KS; the
shift-averaged estimate hits the closed form `E[relu(a·x)] = ‖a‖/√(2π)`.

---

## 5. Dimension ordering makes it WORSE, and the reason is that ANOVA is basis-dependent

The brief's expectation was that ordering would be worth more than the generating
vector — put the most important input directions in the best-searched lattice
dimensions. Two orderings were built, both free, both exactly unbiased:

* **`rownorm`** — input coordinates sorted by `‖W¹[j,:]‖²`. By Stein's lemma
  `Cov(f, x_j) = E[∂f/∂x_j]`, so a coordinate's first-order influence is its
  total drive into layer 1, and the row norm is the only predict-time-affordable
  proxy. A permutation of the **rows of `W¹`** implements it at zero per-sample
  cost, and leaves `sigma_i = ‖W¹[:,i]‖`, the Hermite Gram `rho` and every
  control variate invariant.
* **`activesub`** — the mean-field active subspace. `R = W¹ diag(Φ(α¹)) W² … W³²`
  is `E[∂z³²/∂x]` to first order in the gate fluctuations (31 matmuls, 1.0e9
  FLOPs, 0.4% of `B`); the eigenvectors of `C = R Rᵀ` ordered by eigenvalue are
  the input directions the scored row actually responds to. `Q` is orthogonal so
  `Qx ~ N(0,I)` and the estimator stays exactly unbiased, and `x → Qᵀ` folds into
  `W¹` for one `256³` matmul per MLP and nothing per sample.

`scripts/50_rqmc_rate.py --mode order`, `N = 16,381`, 8 official MLPs × 16
randomisations, all three lattice arms **paired on a common Cranley-Patterson
shift** so the comparison carries no shift-to-shift variance:

| arm | variance | x over iid | ± se |
|---|---|---|---|
| iid | 2.30530e-06 | 1.000 | — |
| **lattice, no ordering** | **1.14230e-06** | **2.018** | 0.138 |
| lattice + `rownorm` | 1.36949e-06 | 1.683 | 0.135 |
| lattice + `activesub` | 1.33718e-06 | 1.724 | 0.198 |

**Both orderings lose: 0.834x and 0.854x against leaving the dimensions alone.**
This extends radiant-allomancer's B12 (eigen/PCA ordering measured 0.95–1.04x on
a *Roberts* lattice, which is dimension-balanced so there was nothing to
exploit): on a CBC lattice, which really is dimension-graded, ordering is not
merely neutral, it is harmful.

Two mechanisms, and the second is the interesting one.

1. **There is no importance gradient to order by.** `W¹[i,j] ~ N(0, 2/256)`
   **iid**, so the row norms `‖W¹[j,:]‖²` have mean 2 and standard deviation
   `2√(2/256) = 0.177` — an 8.8% spread. Sorting 256 coordinates whose true
   importances differ by 8.8% sorts mostly noise, and the sort is a
   data-dependent choice fitted to that noise.

2. **A rotation is not ANOVA-preserving.** The active subspace maximises
   *explained variance per direction*; the lattice needs *additivity*, and those
   are different objects. The `f_1 = 27.6%` that the lattice annihilates is the
   first-order ANOVA share **in the coordinates you feed it**. Rotating mixes
   coordinates, so a function that was a sum of univariate pieces becomes a
   function of many new coordinates at once — concentrating variance into few
   directions can *raise* the effective dimension. That the variance-concentrating
   rotation lost 15% is direct evidence that it did.

So the input basis the problem arrives in is already the right one, and the
`W¹`-row permutation and the active-subspace rotation are both dead. Neither is
kept in the deployable kernel.

## 6. Redundancy with the shipped control variates — and the head has to come off

This is the number that decides whether it ships, and radiant-allomancer named
the risk exactly (18085 §2.1): *"covariance shrinkage already removes most of the
variance RQMC targets — the two levers are partly redundant"*, which is what
turned a reported 5–7x into their measured 1.40x.

It is worse than partly redundant here, and the mechanism is a theorem rather
than an accident. A rank-1 lattice **annihilates the first-order ANOVA terms**;
the shipped `k = 1` Hermite block is provably the **optimal input-linear control
variate**. Those are the same 25–28% of the variance, reached from two
directions. So the two levers are not merely overlapping, they are aimed at the
same target.

`scripts/51_rqmc_deploy.py --mode redundancy`, through the shipped
`corrected_sparse_kernel` with the draw swapped at the `x0_fn` hook. `N = 24,989`
(the prime just under the shipped 25,000), `tau = 2.5`, `P = 225`, 6 official
MLPs x 10 independent randomisations, all four cells paired on the same seeds,
MSE against the official 1e9 reference:

| arm | MSE | variance | `bias^2` | x over iid/off | ± se |
|---|---|---|---|---|---|
| iid, head off | 2.1695e-06 | 2.2994e-06 | −1.30e-07 | 1.000 | — |
| iid, head ON **(the ship)** | 1.3480e-06 | 1.3743e-06 | −2.63e-08 | 1.609 | 0.153 |
| **lattice, head off** | **8.7599e-07** | 8.8735e-07 | −1.14e-08 | **2.477** | 0.463 |
| lattice, head ON | 9.5061e-07 | 9.6349e-07 | −1.29e-08 | 2.282 | 0.388 |

```
lattice alone            2.477x
head alone               1.609x
both                     2.282x     <- LESS than the lattice alone
product if independent   3.986x
REDUNDANCY               0.573      (1.00 = no overlap)
head on top of lattice   0.922x     (against 1.609x on top of iid)
```

**With the lattice in, the shipped head is net negative.** Its coefficients were
fitted offline on iid draws, where `cv1` predicts a sampling error that is really
there; under a lattice most of that error has already been integrated away, so a
coefficient of 1 over-corrects and injects noise. Every `bias^2` column is
negative — i.e. unmeasurably small against 10 randomisations — so the lattice
imports no bias, and in particular the split-half control-variate construction
does not break when the two halves of a shifted lattice stop being independent.

The head's scale can be recovered offline for free, because `damp` enters
linearly: `P(damp) = P(0) + damp·(P(1) − P(0))` at fixed seed, so one pair of
runs prices every `damp`.

| `damp` | 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.75 | 1.0 | 1.25 |
|---|---|---|---|---|---|---|---|---|
| iid MSE (e-06) | 2.170 | 1.979 | 1.814 | 1.674 | 1.559 | 1.404 | **1.348** | 1.393 |
| lattice MSE (e-06) | 0.876 | 0.868 | **0.865** | 0.867 | 0.874 | 0.902 | 0.951 | 1.019 |

The iid arm's optimum is `damp = 1`, which is where it was trained — a good sign
for the training pipeline. The lattice arm's optimum is `damp ≈ 0.25` and it is
worth **1.013x** over switching the head off entirely, i.e. nothing: one scalar
fitted on 6 MLPs cannot claim 1.3%. **Ship the lattice with `damp = 0`.** That is
also cheaper (no feature block, no mean-field control variate, no 31 extra
matvecs) and it removes `corrector.npz` from the critical path.

Headline of this section, paired on identical seeds:

```
shipped (iid + head)   1.3480e-06
lattice, head off      8.7599e-07      1.5388x
```

The pre-registered bar was **1.20x on top of the head**, i.e.
`iid+head / lattice+head = 1.3480/0.9506 = 1.418x`: **PASS**. The configuration
actually worth shipping does better, at 1.539x, by deleting the head instead of
keeping it.

### 6.1 The obvious follow-up, not done here

The head is not intrinsically redundant — it is *mis-fitted*. Refitting `beta`
offline on **lattice** draws (`scripts/28_learned_corrector.py`, 640 generated
MLPs, disjoint seeds) would give the head coefficients appropriate to a residual
whose first-order part is already gone. `docs/learned_corrector.md`'s own
leave-one-group-out table is the tool for deciding which of the 15 columns still
carry signal under a lattice; `cv1` almost certainly does not, and `cv2` (order-2
along `W^1`) probably does. That is a training job, not a kernel job, and it is
the single highest-value thing left on this line.

## 7. `N` re-optimised from scratch — and it does not move, because `p = 1.04`

The brief expected the `N` argument to invert: with `p > 1`,
`adjusted = raw · max(0.1, C/B)` keeps falling with `N` above the clamp instead
of being flat in it, so the optimum should run to the full budget. At the
**measured** `p = 1.043` that effect is `adjusted ∝ N^{-0.043}` — 3% over a
factor of two in `N` — and it is swamped by two things that both push the other
way.

`scripts/51_rqmc_deploy.py --mode score`, official 100-MLP suite, 1e9 reference
so `raw_mse` is leaderboard-comparable, lattice at `damp = 0`, `P = 225`:

| `N` | raw MSE | `F/B` | `C/B` | adj@1x | adj@2x | worst MLP |
|---|---|---|---|---|---|---|
| 8,467 | 3.3203e-06 | 0.0938 | 0.0999 | 3.3203e-07 | 3.5239e-07 | 2.833e-05 |
| 16,993 | 1.5778e-06 | 0.1846 | 0.1931 | 3.0468e-07 | 3.1809e-07 | 1.668e-05 |
| **24,989** | 1.0530e-06 | 0.2698 | 0.2819 | **2.9686e-07** | **3.0964e-07** | 4.163e-06 |
| 34,981 | 7.4561e-07 | 0.3763 | 0.4033 | 3.0071e-07 | 3.2085e-07 | 4.380e-06 |
| **49,999** | 4.5715e-07 | 0.5363 | 0.6151 | **2.8121e-07** | 3.1725e-07 | 1.216e-06 |
| 69,997 | 4.1895e-07 | 0.7494 | 0.8659 | 3.6277e-07 | 4.1160e-07 | 2.565e-06 |
| 99,991 | — | 0.9974 | 1.0093 | **INFEASIBLE** | | 92/100 raised |

Three things to read off, and the third was not anticipated.

**1. `N ≈ 100,000` is not merely expensive, it is illegal.** `F/B = 0.997` before
the feature block, and 92 of 100 MLPs raise `BudgetExhaustedError` — *"matmul
would cost 784,896,000 FLOPs but only 424,119,825 remain"*. `F` is
machine-independent, so this is a hard cap on the grader too, not a property of
this box. The usable range ends around `N = 85,000`.

**2. The optimum is flat and it has not moved.** `2.97e-07` at 24,989 against
`2.81e-07` at 49,999 at 1x residual — 1.06x — and the ordering **reverses** at 2x
residual (`3.10e-07` vs `3.17e-07`). `C/B = 0.615` at `N = 49,999` is far off the
clamp, so any residual surprise on an unknown box is amplified there, and
`docs/cost_floor.md` §5 measured a 3x residual cliff past `N ~ 35,000` on this
shape when the sample array leaves cache. **Keep `N ≈ 25,000.** The 1.06x at
50,000 is real but it is bought with the risk the clamp exists to remove.

**3. The binding constraint at large `N` is the PILOT, not the sampler.** Raw
stops falling: 4.5715e-07 at `N = 49,999` to 4.1895e-07 at 69,997 is 1.09x for
1.4x the samples, a local exponent of 0.24. Solving `raw = b² + v/N` on that pair
gives

```
b^2  =  3.23e-07        v = 6.7e-03
```

so at `N = 49,999` the `N`-independent floor is **71% of raw**, and even at
`N = 24,989` it is 31%. The worst MLP going *up* from 1.216e-06 to 2.565e-06 as
`N` rises is the same signature. That floor is `P = 225` — the pilot sets the
mask and the frozen constants and its error scales as `1/P`, not with `N`
(`submission/estimator.py`, `N_PILOT`). **The lattice makes this worse, not
better**, because halving the sampling variance doubles the pilot's share of what
is left. `P = 600` costs 2.5e9 FLOPs = 0.9% of `B`; on this evidence it is the
next thing to re-optimise, jointly with `N`, and it is worth up to 1.4x at the
shipped `N` — more than the entire remaining `N` lever.

---

## 8. What this changes in the rest of the repository

**`docs/cost_floor.md` and `docs/graded.md`** reduce a sampler's score to
`adjusted = v_eff · c / B` by substituting `raw = v_eff/N` into
`adjusted = raw · max(0.1, C/B)`, at which point `N` cancels. That substitution
is the `p = 1` case. For general `p`, above the clamp,

    adjusted  =  v (F0 + lambda R) / (B N^p)   +   v c / (B N^(p-1))

which is flat in `N` only at `p = 1`; for `p > 1` it keeps falling and the
optimum runs to the largest affordable `N`. Both pages are now scoped to **iid**
samplers. At the measured `p = 1.043` the practical consequence is 3% over a
doubling of `N` and §7 shows the optimum does not move — but the identity was not
entitled to that, and the scoping is the honest fix.

**`docs/floor_theorem.md`**'s low-order barrier had this row at `pending | 1.38x`
and predicted "a 1.5x bar is therefore unreachable". Measured: mean 2.00x, rising
to 2.65x at `N = 131,071`. The bound's *arithmetic* is right — `f_1 = 27.6%` does
give `1/(1-f_1) = 1.38x` for a mechanism that only reaches first order — and the
lattice exceeds it because it also reaches part of the order-2 mass. What survived
is the bound's **shape**: it is a statement about a constant, and the constant is
what changed. That page now carries a scope warning saying so, and the
retrodiction row is updated with the measurement.

**Nothing in `docs/hermite_rank_ceiling.md` moves.** Its 1.76x is a ceiling on a
*dictionary's* reach, which is a different object from a point set's
equidistribution; the two multiply, and §6 measures how much of that product is
real (`redundancy = 0.573`, so much less than the naive 1.76 × 2.0).

---

## 9. What to hand over

**Kernel.** `whestfloor/rqmc.py` — the offline lattice machinery (`get_z` with
its disk cache, `cbc_order2`, `roberts_z`, `lattice_quality`) and the deployable
draw (`billed_lattice_base`, `billed_lattice_normals`, `lattice_x0_fn`).
`whestfloor/kernels.py::corrected_sparse_kernel` gained **one** optional
argument, `x0_fn`; `x0_fn=None` is bitwise and FLOP-for-FLOP the previous
program, asserted in `tests/test_rqmc.py`.

**Shipping configuration**, as measured:

```
base  = billed_lattice_base(24989, get_z(24989, 256, "cbc"))   # in setup, free
pred  = corrected_sparse_kernel(W, tau=2.5, n_samples=24989, n_pilot=225,
                                beta=None, damp=0.0,
                                x0_fn=lattice_x0_fn(base))
```

`N = 24,989` is the prime just under the shipped 25,000 — CBC needs a prime `N`,
and every `z_j` is then automatically coprime to `N`, which is what makes the
one-dimensional projections exact. The generating vector is 256 integers, data
independent, and belongs in the submission as a literal (it is `2 KiB`).

**Do not ship**: the `W¹`-row permutation, the active-subspace rotation (§5, both
lose), antithetic pairing (radiant-allomancer A7, and `scripts/27_rqmc.py`
measures 1.284x against the lattice's 1.910x), or the offline head at `damp = 1`
(§6, it is net negative under a lattice).

**Next, in value order.** (1) Re-optimise `P` jointly with `N` — §7 finds an
`N`-independent floor that is 31% of raw at the shipped `N` and 71% at 50,000,
and it is the pilot. (2) Refit the head on lattice draws (§6.1). Neither is a
kernel change.
