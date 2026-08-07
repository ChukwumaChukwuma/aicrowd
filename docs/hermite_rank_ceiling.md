# High-degree Hermite CVs on adapted directions: the blocker is rank, not `p/N`

**Result: measured and killed, with a ceiling. The whole layer-1 Hermite
family — degree <= 8 on all 256 coordinate directions, degree <= 16 on a
24-direction adapted frame with every cross product, 2,494 features in one
design — reaches held-out `R^2 = 43.2%`
against a pre-registered bar of 75%. The shipped 512-feature `k <= 2` basis
already reaches `39.2%`, so the entire remaining family is worth **4.0
points** and costs 1,982 features (`p/N = 0.23`) to get them. None of the 71
dictionaries measured beats the shipped one once `p/N` is charged; the shipped
basis is the argmax of the surface (§8 records the one derived combination
that does beat it, by 5% of the sampling residual, and why it is not cashed
here).** Reproducible from
`scripts/32_adapted_hermite.py` (`--mode dirs|span|chaos|ceiling`), with every
load-bearing identity asserted in `tests/test_adapted_hermite.py`.

The reframe that motivated this was half right and the half that was wrong is
the interesting part. `p/N` really is what killed `k = 3` in the ship. But
with `p/N` removed *entirely* — the `R^2_pop` column below is the `N -> inf`
limit — degree 3 and above are still worth about two points out of the **54
points of variance that genuinely live there**. §5 says why, exactly, and the
statement is the analogue of the raising-operator bound in `docs/stein_cv.md`.

---

## 1. The algebra, and the one identity everything turns on

`x ~ N(0, I_n)` exactly, so for any unit vector `a` the projection
`s_a = <a, x>` is exactly standard normal and, with the L2-normalised Hermite
`h_d = He_d / sqrt(d!)`,

    E[h_d(s_a)] = 0                                exactly, for every d >= 1
    Cov(h_d(s_a), h_e(s_b)) = delta_de <a, b>^d                     (Mehler)

Every feature is an exactly-mean-zero control variate and the Gram is
**analytic**, block diagonal in degree — nothing is estimated but the
covariance with the target. The shipped layer-1 family is the special case
`a = W^1[:,i] / ||W^1[:,i]||`, whose degree-`d` Gram is the Hadamard power
`rho^{o d}`. That much is the brief's premise and it is correct.

The identity that decides the question is this one. Expand `h_d(<a,x>)` in the
tensor-Hermite basis `H_alpha = He_alpha / sqrt(alpha!)` of the degree-`d`
Wiener chaos:

    h_d(<a,x>)  =  sum_{|alpha| = d} sqrt(d!/alpha!) a^alpha H_alpha(x)

(asserted numerically in `tests/test_adapted_hermite.py::
test_pure_power_is_the_rank_one_tensor`). The coefficient array is
`a^{(x)d}` and its norm is `(sum_i a_i^2)^d = 1` by the multinomial theorem.
So:

> **A one-direction Hermite feature is exactly a unit RANK-ONE symmetric
> tensor of the degree-`d` chaos, and nothing else.**

Two corollaries used throughout:

* For an **orthonormal** set `A = [a_1 ... a_m]` the products
  `H_alpha = prod_r h_{alpha_r}(<a_r, x>)` are exactly orthonormal and span
  exactly `Sym^d(span A)` at degree `d`, of dimension `C(m+d-1, d)`.
* The overlap of any such product with a pure power on an arbitrary unit `c`
  is `delta_{|alpha|,d} sqrt(d!/alpha!) prod_r (A^T c)_r^{alpha_r}`.

Those two make the Gram of the *union* of the shipped coordinate family, an
adapted family, and all their cross products **fully analytic** — which is
what lets §3 measure a 2,494-feature dictionary without ever inverting an
estimated covariance. `tests/…::test_design_gram_matches_the_sample_covariance`
checks the assembled Gram against 300,000 samples.

## 2. The directions, and how well the weights alone find them

`--mode dirs`, 3 local MLPs, 100,000 pilot samples. The mean Jacobian
`G = E[x^T ybar]` (Stein: `E[grad_x relu(z^32_j)] = E[x ybar_j]`) is measured
with the **centred** target: `E[x] = 0` only in population, and the sample term
`xbar (x) ybar` is a rank-one contaminant of norm
`sqrt(n/N_pilot) * |E y| ~ 0.9` against a signal of ~1.6, which otherwise
takes over the top singular direction outright. It cost an hour to find.

| | mlp 900000 | 900001 | 900002 |
|---|---|---|---|
| `f_1` | 0.229 | 0.281 | 0.225 |
| share of `f_1` in the top-1 jac direction | 78.6% | 90.7% | 84.8% |
| top-4 | 95.9% | 97.5% | 95.9% |
| top-16 | 99.8% | 99.9% | 99.8% |
| mean-field frame recovers top-1, `\|cos\|` | 0.935 | 0.941 | 0.921 |
| mean-field frame recovers top-8 subspace | 0.893 | 0.820 | 0.902 |

**The degree-1 content is essentially one-dimensional**, and the mean-field
path matrix `W^1 diag(Phi(alpha^1)) ... W^32` — free from the weights and the
pilot — finds that direction to `|cos| = 0.93`. This is the strongest possible
setup for the brief's construction, and §3 is what it buys.

## 3. The `R^2(p, d)` surface

`--mode span`, 4 local MLPs x 240,000 samples, adapted frame = mean-field,
`M = 24`. Coefficients are fitted on one half and the explained variance read
on the other, both ways, against the **analytic** Gram — so `R^2_pop` is an
unbiased estimate of the *population* span `c' G^-1 c` (the dictionary's
approximation power, no `p/N` inflation) and `R^2_fit` is what it achieves with
120,000 fitting samples. `eff R^2 = R^2_pop - 2p/8500` charges the split
estimator's noise at the shipped operating point; the model is calibrated —
it predicts 1.240x for `cv1` where `docs/learned_corrector.md` §3.3 measured
1.253x, and 1.374x for `cv1+cv2` where it measured 1.328x.

### 3.1 The coordinate (layer-1) family — degree buys nothing

| dictionary | p | p/N | `R^2_pop` | 1/(1-R^2) | eff `R^2` | eff x |
|---|---|---|---|---|---|---|
| `coord d<=1` (H1) | 256 | 0.030 | 25.38% | 1.340 | 19.35% | 1.240 |
| **`coord d<=2` (H1+H2, SHIPPED)** | **512** | 0.060 | **39.24%** | **1.646** | **27.20%** | **1.374** |
| `coord d<=3` | 768 | 0.090 | 40.14% | 1.671 | 22.07% | 1.283 |
| `coord d<=4` | 1024 | 0.120 | 40.76% | 1.688 | 16.66% | 1.200 |
| `coord d<=6` | 1536 | 0.181 | 41.00% | 1.695 | 4.86% | 1.051 |
| `coord d<=8` | 2048 | 0.241 | 41.10% | 1.698 | −7.08% | 0.934 |

Each degree in isolation: `d=2` **13.87%**, `d=3` **0.90%**, `d=4` 0.62%,
`d=5` 0.07%, `d=6` 0.18%, `d=7` 0.02%, `d=8` 0.09%. Degrees 3 through 8
together are worth **1.9 points**, in the `N -> inf` limit, for 1,536
features.

### 3.2 The adapted family — smaller, and it does not go further

Pure powers on the mean-field frame (`R^2_pop`, so `p/N` is not what is being
shown):

| | `d<=1` | `d<=2` | `d<=4` | `d<=8` | `d<=16` |
|---|---|---|---|---|---|
| m=1 | 19.48% | 19.87% | 19.87% | 19.87% | **19.88%** |
| m=2 | 20.97% | 21.64% | 21.64% | 21.65% | 21.65% |
| m=4 | 22.00% | 23.08% | 23.09% | 23.10% | 23.10% |
| m=8 | 22.71% | 24.64% | 24.65% | 24.65% | 24.65% |
| m=16 | 23.21% | 26.22% | 26.22% | 26.23% | 26.22% |
| m=24 | 23.53% | 26.54% | 26.55% | 26.55% | 26.55% |

**Read the rows, not the columns.** Going from degree 2 to degree 16 — eight
times the basis — moves every row by less than 0.02 points. The columns do
move, which is the brief's point and it is real: `m = 24` at degree 1 costs
24 features and recovers 92.7% of what the full 256-feature `H1` block gets.

With all cross products, and the unions:

| dictionary | p | `R^2_pop` | eff `R^2` | eff x |
|---|---|---|---|---|
| adapt tensor m=4 deg<=2 | 14 | 23.20% | 22.87% | 1.297 |
| adapt tensor m=8 deg<=2 | 44 | 25.08% | 24.05% | 1.317 |
| adapt tensor m=16 deg<=2 | 152 | 27.49% | 23.92% | 1.314 |
| adapt tensor m=8 deg<=3 | 164 | 25.15% | 21.29% | 1.270 |
| adapt ALL (deg<=4, pure to 16) | 446 | 27.89% | 17.39% | 1.211 |
| SHIP + adapt tensor m=16 deg<=2 | 664 | 41.23% | 25.61% | 1.344 |
| SHIP + adapt ALL | 958 | 41.30% | 18.76% | 1.231 |
| **EVERYTHING** | **2494** | **43.16%** | −15.52% | 0.866 |
| per-neuron dir=mf, `d<=1` | 1/neuron | 23.17% | 23.15% | 1.301 |
| per-neuron dir=mf, `d<=8` | 8/neuron | 23.73% | 23.54% | 1.308 |
| per-neuron dir=jac (oracle), `d<=1` | 1/neuron | 25.70% | 25.68% | 1.346 |

**The whole family is 43.2%.** The shipped 512 features already have 39.2% of
it. Every one of the 1,982 features that would buy the last 4.0 points costs
more in `p/N` than it returns, and the argmax of `eff R^2` over all 71
dictionaries above is the one that is already shipped.

Degrees are exactly orthogonal, so span `R^2` is *additive across degree
blocks* — `coord d<=1` 25.38% + `coord deg 2 alone` 13.87% = 39.24% = `coord
d<=2`, to the second decimal. That lets any degree-1 block be swapped for any
other without re-measuring, and §8 uses it on the one combination that beats
the ship.

## 4. The high-degree content exists — it is just not rank-one

If the family stops at 43%, the natural question is whether there is anything
above degree 2 at all. There is, and a lot of it.

`--mode chaos`, 4 local MLPs x 60,000 samples. The Ornstein-Uhlenbeck /
Mehler semigroup multiplies the degree-`d` chaos by `t^d`: with
`x_t = t x + sqrt(1-t^2) xi` and `xi` an independent standard normal,

    C(t) := sum_j Cov(y_j(x), y_j(x_t)) / sum_j Var(y_j)  =  sum_{d>=1} f_d t^d

where `f_d` is the share of `Var(y)` in the degree-`d` chaos. This is the
**degree** decomposition, not the coordinate-subset ANOVA decomposition that
`docs/floor_theorem.md` measures — `He_2(x_1)` has ANOVA order 1 and degree 2,
so `d_M = 11.5` was never the mean degree. One extra forward pass per
coupling, variance `O(Var(y)^2 / N)`: no curse of dimensionality, unlike
estimating a degree-`d` coefficient tensor.

    C(t):  0.10 -> 0.0284   0.30 -> 0.1004   0.50 -> 0.2022   0.70 -> 0.3508
           0.80 -> 0.4582   0.90 -> 0.6142   0.95 -> 0.7354   0.98 -> 0.8486

| d | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `f_d` | **26.65%** | 18.95% | 11.23% | 6.99% | 4.67% | 3.32% | 2.48% | 1.92% |
| cumulative | 26.65% | 45.60% | 56.84% | 63.83% | 68.50% | 71.82% | 74.30% | 76.22% |

`f_1 = 26.7%` (per MLP 23.1 / 31.3 / 23.1 / 28.0) reproduces the
`f_1 = 0.276 +- 0.016` of `docs/floor_theorem.md` from a **seventh independent
estimator**, so the instrument validates itself before it is used. The
**mean Hermite degree is 10.5**, and `1 - f_1 - f_2 = 54%` of the variance
sits strictly above degree 2.

`C(1) = 1` identically, which gives an assumption-free upper bound with no
model of the tail: `t^d >= t^D` for `d <= D`, so
`sum_{d<=D} f_d <= min_t C(t)/t^D` — 28.4% at `D=1`, 71.6% at `D=2`, 84.3% at
`D=3`. The `f_d` above are the non-negative least-squares inversion of the
same `C(t)`, which is an ill-posed Hausdorff moment problem and is quoted as a
smooth readout, not a hard bound; it is accurate to ~2 points of cumulative
mass on a geometric spectrum (`tests/…::test_nnls_recovers_a_smooth_spectrum`).

**So the content is there and the family cannot see it.** At degree 2 the
family gets 13.87 of the 18.95 points available (73%). At degree 3 it gets
0.90 of 11.23 (8%). At degrees 4-8 it gets 0.97 of 19.4 (5%). The barrier is
not the ANOVA order and it is not `p/N`. It is **rank**.

## 5. The ceiling, three ways

### 5.1 The family ceiling is a conditional expectation, at every degree at once

**Theorem.** For unit directions `A = [a_1 ... a_m]`, the closed linear span of
`{ prod_r He_{d_r}(<a_r, x>) }` over all multi-degrees is exactly
`L^2(sigma(A^T x))` — polynomials in `m` Gaussian variables are dense. Hence
for ANY Hermite dictionary on `A`, at any degree and with any products,

    R^2  <=  sum_j Var(E[y_j | A^T x]) / sum_j Var(y_j)

with no truncation in degree anywhere. This is the analogue of §4 of
`docs/stein_cv.md`: a bound on the whole family from one line of algebra.

`--mode ceiling`, 4 local MLPs x 120,000 samples, measured by equiprobable
binning with the per-bin means taken from **independent halves**, so the
estimate is unbiased (a same-half bin variance is inflated by `nbins/N`):

| frame | m=1 (400 bins) | m=2 (729) | m=3 (1331) |
|---|---|---|---|
| mean-field | **19.77%** | 21.32% | 21.67% |
| mean Jacobian (oracle) | 22.05% | 23.66% | 23.97% |
| random | 0.08% | 0.11% | 0.28% |

Compare the mean-field row against §3.2: the degree-free ceiling for one
direction is **19.77%** and Hermite degrees 1 and 2 alone already give
**19.87%**. *Degrees 3 through infinity along any single direction are worth
zero, measured without ever truncating a degree.* That closes the "maybe
degree 30 has something" objection completely.

### 5.2 The degree-2 reachability spectrum — an exact rank bound

At degree 2 the chaos coefficient of `ybar_j` is the symmetric matrix
`S_j = E[ybar_j (x x^T - I)]` with `||T_j^(2)||^2 = (1/2) ||S_j||_F^2`. A
dictionary on an `m`-dimensional direction subspace spans `Sym^2(A)`, so its
capture is `(1/2) sum_j ||Pi_A S_j Pi_A||_F^2 <= (1/2) sum_j ||Pi_A S_j||_F^2`,
whose maximum over **every** `m`-dimensional subspace at once is
`(1/2) sum_{k<=m} lambda_k(P)` with

    P = sum_j S_j^2   (256 x 256),   tr(P) = 2 f_2 sum_j Var(y_j)

`P` is estimated from a cross-half pair block using
`(xx^T - I)(x'x'^T - I) = <x,x'> x x'^T - x x^T - x' x'^T + I`, so it is
unbiased; `tests/…::test_degree2_rank_recovers_a_known_rank_one_tensor` pins
the estimator on a synthetic rank-one target.

| m | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| max share of `f_2` any m directions reach | 2.9% | 5.5% | 10.4% | 18.8% | **32.7%** | 54.0% | 81.4% | 100% |
| features that costs, `C(m+1,2)` | 1 | 3 | 10 | 36 | 152 | 528 | 2080 | 8256 |

The bound is tight against §3.2 and validates it: at `m = 16` it allows
`0.327 x 18.95 = 6.2` points and the measured degree-2 increment
(`adapt tensor m=16 deg<=2` minus `adapt pure m=16 d<=1`) is **4.3**; at
`m = 8` it allows 3.6 and the measurement is 2.4; at `m = 4`, 2.0 against 1.2.

Now put the shipped basis on the same axis. The 256 layer-1 coordinate
features are 256 *rank-one* tensors `w_i (x) w_i`, and they capture **13.87
points = 73% of `f_2`**. To match that with a shared-direction tensor basis
you need `m ~ 55`, i.e. **~1,500 features** — six times as many. The layer-1
basis is not an arbitrary basis at degree 2; it is very nearly the only
affordable one.

**Why.** A ReLU network is piecewise linear, so its Hessian is zero almost
everywhere and `E[grad^2 y_j] = S_j` is carried entirely by the kink surfaces.
Neuron `(l,i)`'s kink contributes `E[delta(z^l_i) (dy_j/dx^l_i) grad z^l_i (x)
grad z^l_i]` — a rank-one term along its own normal. **At `l = 1` that normal
is the constant `W^1[:,i]`; at every deeper layer it varies with `x` and the
contribution is spread.** So the degree-2 chaos has exactly 256 fixed rank-one
directions in it, they are precisely the shipped basis, and the rest is
smeared over ~128 more. The same argument at degree 3 needs *two* kinks to
coincide, which no single direction sees at all — hence `f_3 = 11.2%` and a
reachable 0.9%.

### 5.3 Degree 3 and up, and the dictionary-free bound

Two more facts, for completeness:

* **Any dictionary of `p` functions whatsoever** (Hermite or not) is bounded
  by the top-`p` eigenvalues of the 256x256 output covariance `Cov(y)`,
  because the best `p`-dimensional subspace of `L^2_0` for
  `sum_j ||Pi ybar_j||^2` is spanned by the top-`p` eigenfunctions of
  `sum_j ybar_j (x) ybar_j`, whose non-zero spectrum is that of `Cov(y)`.
  Measured: `p=1` 66.7%, `p=4` 83.1%, `p=8` 90.1%, `p=32` 98.0%. **So the
  `p/N` budget is not what closes this** — eight well-chosen features would in
  principle do the job. They are simply not rank-one Hermites.
* Evaluating a general degree-2 feature `x^T M x - tr M` costs `rank(M)` inner
  products per sample. The eigenvectors of `P` need `m ~ 128` to span, so the
  ~256 features that *would* capture `f_2` in full cost ~1.7e7 FLOPs/sample
  against 2.79e6 for the entire scored pass — **six forward passes per
  sample**. The shipped basis costs zero extra, because the forward pass has
  already computed `z^1`.

### 5.4 The statement

> **Ceiling.** Let `A` be any set of directions derivable from the network and
> let the dictionary be any set of Hermite products on `A`, at any degree.
> Because `h_d(<a,x>)` is a unit rank-one tensor of the degree-`d` chaos, the
> dictionary lives in `sum_d Sym^d(span A)`, and its `R^2` is bounded by
> `sum_j Var(E[y_j | A^T x]) / sum_j Var(y_j)` — for every degree at once.
> Measured on this suite: **43.2% (1.76x) for the largest dictionary we can
> build, of which the shipped 512-feature `k <= 2` basis already realises
> 39.2% (1.65x).** The layer-1 coordinate directions are the network's only
> *fixed* kink normals and therefore the only rank-one directions that carry
> degree-2 mass; degree 3 and above carry **54%** of the variance and the
> entire family sees 1.9 points of it.

`1/(1 - 0.432) = 1.76x` is within half a percent of the
`1/(1 - 0.43) = 1.75x` that
`docs/floor_theorem.md`'s `k <= 2` amendment quotes — the two bounds coincide
numerically, from completely different arguments, and the coincidence is not
luck: everything the Hermite family can reach at any degree is what it can
reach at degree 2.

## 6. Bars, fixed before the run

| # | bar | measured | verdict |
|---|---|---|---|
| 1 | held-out `R^2 > 0.75` (> 4x) at `p/N < 0.15` | best under the p budget is the SHIPPED `coord d<=2` at **39.2%**; the largest dictionary at any p is **43.2%** at `p/N = 0.29` | **FAIL — not integrated** |
| 2 | some dictionary in the surface beats the shipped basis on `eff R^2` | argmax over all 71 dictionaries measured is `coord d<=2`, the shipped one, at 27.20% (official 27.64%); §8 records a derived degree-block swap at 31.0% that is not one of the 71 | **FAIL** |
| 3 | the ceiling of §5 is stated quantitatively and validated against the surface | §5.2's rank bound predicts <=6.2 / <=3.6 / <=2.0 degree-2 points at m=16/8/4 and the surface measures 4.3 / 2.4 / 1.2 | **PASS** |
| 4 | shipped estimator untouched: 0 raises on 100 official MLPs, `damp=0` bitwise | see §7 | **PASS** |

## 7. The official suite

Nothing in §1-§6 opens the official suite. This section does, twice, and
changes nothing: **the shipped estimator is not modified by this round.**

`scripts/28 --mode score`, all 100 official MLPs, N=1e9 reference, both
variants interleaved in one process:

| variant | raw_mse | F/B | C/B | adj@1x | adj@2x | adj@3x | raises | worst MLP |
|---|---|---|---|---|---|---|---|---|
| **shipped (`damp=1`)** | **3.7157e-06** | **0.0919** | 0.1080 | **4.0142e-07** | 4.6140e-07 | 5.2138e-07 | **0** | 3.2338e-05 |
| `damp=0` ablation | 5.8050e-06 | 0.0915 | 0.1044 | 6.0583e-07 | 6.8023e-07 | 7.5463e-07 | 0 | 4.2164e-05 |

Both reproduce `docs/stein_cv.md` §7 to five significant figures
(3.7157e-06 and 5.8050e-06); `raw_mse` and `F/B` are machine-independent and
unchanged, and the `adj` column differs only by this box's residual drift.

`--mode span --official`, the first 4 MLPs of `official_mini.npz` by seed
protocol 3.0, run **after** every dictionary, degree and direction rule was
frozen, purely to confirm that a ceiling measured on look-alike networks
applies to the graded ones. No parameter is fitted or selected from it.

| dictionary | p | local `R^2_pop` | official `R^2_pop` |
|---|---|---|---|
| `coord d<=1` (H1) | 256 | 25.38% | 25.82% |
| **`coord d<=2` (SHIPPED)** | 512 | **39.24%** | **39.69%** |
| `coord d<=3` | 768 | 40.14% | 40.44% |
| `coord d<=8` | 2048 | 41.10% | 41.35% |
| coord degree 2 alone | 256 | 13.87% | 13.87% |
| coord degree 3 alone | 256 | 0.90% | 0.75% |
| adapt tensor m=16 deg<=2 | 152 | 27.49% | 28.36% |
| adapt ALL | 446 | 27.89% | 28.74% |
| per-neuron dir=mf `d<=1` | 1/neuron | 23.17% | 23.24% |
| **EVERYTHING** | 2494 | **43.16%** | **43.42%** |
| argmax of eff `R^2` | — | `coord d<=2`, 27.20% | `coord d<=2`, 27.64% |

Same verdict, same argmax, every entry within 0.9 points. The ceiling is a
property of the network distribution, not of the local draw.

and the degree spectrum, `--mode chaos --official`:

| | `f_1` | `f_2` | `f_3` | `d<=2` | mean degree |
|---|---|---|---|---|---|
| local (4 MLPs) | 26.65% | 18.95% | 11.23% | 45.60% | 10.5 |
| official (4 MLPs) | 26.65% | 18.66% | 11.20% | 45.32% | 10.9 |

`C(t)` itself, which is the raw measurement before any inversion, agrees to
three decimals at every coupling: `0.0284 / 0.1006 / 0.2009 / 0.3496 / 0.6097
/ 0.8439` official against `0.0284 / 0.1004 / 0.2022 / 0.3508 / 0.6142 /
0.8486` local at `t = 0.1 / 0.3 / 0.5 / 0.7 / 0.9 / 0.98`.

## 8. Honest caveats

- **The `f_d` inversion is regularised.** `C(t)` is measured directly and is
  the primary object; `f_d` is a non-negative least-squares readout of an
  ill-posed moment problem. Every load-bearing claim in §5 rests on the
  directly measured span numbers and on the exact rank bound, not on `f_d`;
  `f_d` is quoted for the picture and because `f_1` validates the instrument.
- **The rank bound of §5.2 is one-sided.** `sum_{k<=m} lambda_k(P)` bounds
  `sum_j ||Pi_A S_j||^2`, which is larger than the reachable
  `sum_j ||Pi_A S_j Pi_A||^2`. It is therefore conservative in the right
  direction — the true reachability is worse than the table says, as the
  4.3-against-6.2 comparison shows.
- **Directions were searched, not optimised.** Four families were tried
  (sampled mean Jacobian, mean-field path matrix, un-gated weight product,
  random) and the two that matter agree to `|cos| = 0.93`. No projection
  pursuit was run. §5.1's degree-free measurement and §5.2's
  every-subspace-at-once bound are what cover the directions not tried.
- **One thing did measure better than the ship, and it is small.** Degrees
  are exactly orthogonal, so span `R^2` is additive across degree blocks:
  replacing the 256-feature `H1` block with 24 mean-field directions
  (`23.53%` at `p = 24`) or with one per-neuron direction (`23.17%` at
  `p = 1`) and keeping the coordinate `d = 2` block gives `37.0-37.4%` at
  `p = 257-280`, i.e. **eff `R^2` 31.0% against the shipped 27.2%**, a 5%
  smaller sampling residual. It is not shipped here: the mean-field path
  matrix costs ~1.0e9 FLOPs and the per-sample projection ~1.1e9, together
  1.2% of the free budget, against a predicted ~3% on `raw_mse` once the
  sparse mask's `N`-independent bias is accounted for — and cashing it needs a
  full 640-MLP regeneration and a re-fit of the head. It is the one live
  follow-up, with the number attached.
- **`tau = 2.5`, `N = 8500`, `P = 150` are still inherited.** Unchanged, and
  unchanged by this round.
