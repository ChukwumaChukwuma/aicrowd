# Expressive AND integrable: the 90% is real, and its mean is 4.4x out of reach

**Result. The functions that explain 90% of `sum_j Var(y_j)` exist, are cheap,
and the forward pass already computes them — they are the network's own deep
activations. A *linear* control variate in `relu(z^L)` reaches held-out `R^2` of
46% at `L = 2`, 72% at `L = 8`, 86% at `L = 16` and **98.9% at `L = 32`**, with
256 features and no extra FLOPs. What closes the lever is `E[g]`. `z^1` is
exactly Gaussian so layer 1 is exactly integrable (measured bias 8.8e-05
against a 1.35e-04 reference noise floor: zero); every deeper layer needs the
Gaussian closure, and the closure's error enters the score as `0.1 b^2`. Scored
on the real objective, with the optimal shrinkage on the biased correction and
stacked on the exact layer-1 variate, the ENTIRE ladder from `L = 2` to
`L = 32` is worth **1.03x**, and the argmin is `L = 1` — where the shipped
basis already sits.**

**The forward-looking number is one scalar. The factor `r` by which the
analytic layer-mean would have to improve to break even is 4.2-4.6 at *every*
depth from 6 to 32; at `r = 4.4` the ladder pays 1.9x, at `r = 10` it pays
5.4x, at `r = 30` it pays 25x. So the control-variate question reduces to the
closure-accuracy question: the basis is not the lever, `E[g]` is.**

**The one thing that survives is small: `relu(z^1)` in place of `He_2` — 256
features instead of 512, an exact mean `sigma_i/sqrt(2 pi)`, and 1.014x on
unbiased true MSE over 48 generated MLPs. The one new dictionary that looked
like more — degree 2 in the layer-1 activations, 1.10x jointly fitted — is
dead at every deployable block weight (§4.4), for the same overlap reason
`cva` was.**

Reproducible from `scripts/41_integrable_cv.py`
(`--mode eig|ladder|summary|quad|anti`), 3 local MLPs (seed base 900000, the
same networks `docs/hermite_rank_ceiling.md` measured, so every number below
is directly comparable to that page). The span machinery reproduces that page's
shipped-basis figure to 0.07 points (39.31% here against 39.24% published)
before anything new is measured.

| route | exactly integrable? | held-out `R^2` | scored | verdict |
|---|---|---|---|---|
| shipped layer-1 Hermite `k <= 2` | yes | 39.3% | 1.00x | the incumbent |
| `relu(z^1)` (`h1`), 256 features (§4) | **yes** | 38.6% at **half** the features | **1.014x** end to end | small, real, free |
| degree-2 in `z^2` on the kink frame (§4) | **yes** | 46.5% pop, **1.10x jointly fitted** | **<= 1.000x** at any block weight (§4.4) | **dead — the overlap eats it** |
| linear in `relu(z^L)`, `L >= 2` (§3) | no | up to 98.9% | **1.03x** | **dead — the bias eats it** |
| antithetic `x -> -x` (§5) | yes, by symmetry | 1.077x raw | **0.958x** | **dead** |
| homogeneity Rao-Blackwell (§5) | yes, exactly | 1.038x raw | **0.975x** | **dead** |
| any degree-`<= 2` polynomial in `x` (§2) | yes | 45.6% ceiling | needs 32,896 coefficients | **closed** |

---

## 1. The objective, and why `R^2` alone is the wrong thing to report

`docs/graded.md` §3 reduced a sampler's score, at the multiplier clamp, to

```
adjusted  =  0.1 b^2  +  v_eff c / B
```

with `b` the estimator's bias. Writing the sampling term as `(1 - R^2) V0` with
`V0 = 0.1 V/N = 4.06e-07` calibrated to the ship (`R^2 = 0.392`,
adjusted 2.47e-07),

```
adjusted(g)  =  0.1 b(g)^2  +  (1 - R^2(g)) V0
```

and a control variate is a *joint* choice of a function and a claimed mean.
That is the objective every number on this page is scored against.

One refinement matters and is used throughout §3: **a biased correction is
never applied at full strength.** With a scalar shrinkage `theta` on top of the
exactly-integrable layer-1 variate (`R1 = 37.7%`, bias 0),

```
MSE(theta) = (1 - R1) V/N  -  2 theta D  +  theta^2 D  +  theta^2 b^2,
D := (R_L - R1) V/N
```

is minimised at `theta* = D / (D + b^2)` and saves `D^2 / (D + b^2)` instead of
`D`. So a wrong mean does not merely add a bias term — **it multiplicatively
discounts the whole correction by `D / (D + b^2)`.** Reporting a dictionary's
`R^2` without its `b` is therefore not conservative, it is meaningless.

## 2. The 90.1% is a tautology, and the top eigenfunctions are not low-degree

`docs/hermite_rank_ceiling.md` §5.3 bounds any `p`-function dictionary by the
top-`p` eigenvalues of `Cov(y)`, "measured: `p=8` 90.1%". The bound is correct
and it is not a construction:

> The operator `T = sum_j ybar_j (x) ybar_j` sends every function into
> `span{ybar_1 ... ybar_256}` — `T g = sum_j <ybar_j, g> ybar_j` — so **every
> eigenfunction with a non-zero eigenvalue is a linear combination of the
> centred outputs themselves**, and the non-zero spectrum of `T` is exactly
> that of the 256x256 matrix `Cov(y)`. "The top-8 eigenfunctions explain 90.1%"
> is the statement "`Cov(y)` has effective rank 8", nothing more. Their means
> are zero only because `E[y_j]` has been subtracted, which is the answer.

`--mode eig`, 3 MLPs x 100,000 samples, reproduces the spectrum (top-8
**87.8% / 93.7% / 90.3%**, mean 90.6%) and then asks the question that actually
decides the direction: *are those eigenfunctions low-degree polynomials in `x`?*
Degrees 1 and 2 are measured in closed form and cross-half unbiased —
`||P_1 Y||^2 = ||E[Y x]||^2` and `||P_2 Y||^2 = (1/2)||E[Y (xx' - I)]||_F^2` —
and the whole spectrum through the OU/Mehler semigroup.

| MLP | top-1 | top-8 | top eigenfunction: deg 1 / deg 2 / deg `<=2` | top-8, variance weighted: deg `<=2` |
|---|---|---|---|---|
| 900000 | 57.3% | 87.8% | 31.3% / 28.1% / **59.4%** | **48.4%** |
| 900001 | 77.4% | 93.7% | 33.8% / 30.6% / **64.4%** | **57.5%** |
| 900002 | 65.8% | 90.3% | 28.4% / 29.2% / **57.6%** | **48.3%** |

**Answer: no.** The top-8 eigenfunctions are 51% degree `<= 2` on average, so
the *entire* space of degree-`<=2` polynomials in `x` — all 32,896 of them,
`p/N = 1.2` at the shipped `N = 27000` — recovers barely half of the 90%. And
that is not a coincidence of the eigenbasis: an eigendecomposition of the
output covariance cannot change the degree content of the output, so the
degree-`<=2` ceiling is `f_1 + f_2 = 45.6%` however the variance is sliced. The
shipped 512-feature rank-one basis already realises 39.3% of that 45.6%. **The
low-degree-polynomial route has 6.3 points of headroom and those points cost
32,384 more fitted coefficients than the sample budget contains.**

The single-direction OU bound (`sum_{d<=D} f_d <= min_t C(t)/t^D`) on the top
eigenfunction is `<= 85.3% / 88.3% / 82.4%` at `D = 2` and `<= 95.0% / 97.7% /
94.5%` at `D = 3` — assumption-free but far too loose to contradict the exact
degree-1 and degree-2 measurements above, which is the honest way to read it.

## 3. The depth ladder: expressiveness against integrability

The exactly-integrable universe is small and its boundary is sharp:

* `z^1 = x W^1` is **exactly** `N(0, W^1' W^1)`. Every 1-D Gaussian integral of
  `z^1` is closed form, every 2-D one is the arc-cosine kernel, every
  polynomial is Wick. `E[relu(z^1_i)] = sigma_i / sqrt(2 pi)`, exactly.
* Therefore `E[z^2] = W^2' E[h^1]` and `Cov(z^2) = W^2' Cov(h^1) W^2` are
  **exact as well** — the last exact objects in the network.
* From `relu(z^2)` onwards, every mean requires the integral of a rectifier
  against a distribution that is not Gaussian, which is the whole modelling
  problem this repository has been unable to solve to better than `rms 4.8e-3`.

`--mode ladder` prices that boundary. Features: `hbar^L = relu(z^L) - mtilde^L`
with `mtilde^L` the Gaussian-closure mean (exact ReLU moments, Mehler
covariance to order 8) — 256 features per layer, all of them already computed
by the forward pass. `R^2` is held out, the ridge coefficients `c_j` are the
ones that would actually be used, the bias is `c_j'(mtilde^L - m^L)` against a
2,000,000-sample independent reference, and the objective is §1's.

3 MLPs, 120,000 span samples:

| `L` | `R2_eff` | rms `mtilde - m` | rms bias `c'e` | `0.1 b^2` | `(1-R^2)V0` | adjusted (unshrunk) | x ship |
|---|---|---|---|---|---|---|---|
| **1** | **37.73%** | 5.8e-04 | **8.8e-05** | 7.8e-10 | 2.53e-07 | **2.54e-07** | **0.97** |
| 2 | 46.15% | 1.7e-03 | 1.56e-03 | 2.43e-07 | 2.19e-07 | 4.61e-07 | 0.54 |
| 3 | 52.60% | 2.7e-03 | 2.55e-03 | 6.51e-07 | 1.93e-07 | 8.44e-07 | 0.29 |
| 4 | 58.21% | 3.6e-03 | 3.36e-03 | 1.13e-06 | 1.70e-07 | 1.30e-06 | 0.19 |
| 6 | 65.90% | 5.0e-03 | 4.40e-03 | 1.93e-06 | 1.38e-07 | 2.07e-06 | 0.12 |
| 8 | 71.64% | 5.8e-03 | 5.26e-03 | 2.77e-06 | 1.15e-07 | 2.88e-06 | 0.086 |
| 12 | 80.01% | 7.3e-03 | 5.78e-03 | 3.34e-06 | 8.1e-08 | 3.42e-06 | 0.072 |
| 16 | 85.96% | 7.1e-03 | 6.02e-03 | 3.63e-06 | 5.7e-08 | 3.68e-06 | 0.067 |
| 20 | 90.43% | 7.0e-03 | 6.07e-03 | 3.69e-06 | 3.9e-08 | 3.73e-06 | 0.066 |
| 24 | 93.83% | 7.5e-03 | 6.68e-03 | 4.47e-06 | 2.5e-08 | 4.49e-06 | 0.055 |
| 28 | 96.69% | 7.3e-03 | 6.75e-03 | 4.56e-06 | 1.3e-08 | 4.58e-06 | 0.054 |
| 32 | 98.86% | 7.0e-03 | 7.00e-03 | 4.90e-06 | 4.6e-09 | 4.90e-06 | 0.050 |

Three things to read off.

**Layer 1 is exactly integrable, confirmed to the precision available.** The
measured bias 8.8e-05 sits below the 2e6-sample reference's own noise floor of
1.35e-04, and `rms(mtilde^1 - m^1) = 5.8e-04` is exactly `sd(h^1)/sqrt(2e6)`.
The 0.97x against the ship is the agreement between this measurement and the
shipped estimator, not a loss. (The `L = 1` row *is* the `h1` block of §4,
reached from the other direction and on a different sample: 37.73% here against
37.68% there, on the same three networks. The two halves of this page agree to
0.05 points where they overlap.)

**The 90% is real, cheap and constructive.** `R^2` climbs monotonically to
98.9%, with 256 features and zero extra FLOPs, on features the network hands
over for free. Everything §2 says is unreachable by polynomials is reachable
here — because these are not polynomials, they are the network.

**`b` saturates before `R^2` does, and that decides it.** By `L = 8` the bias is
already 75% of its `L = 32` value while `R^2` has covered only 55% of the
distance from 38% to 99%. The mechanism is that the closure error is injected
once per layer and then *amplified* by the remaining nonlinear layers — the
same shape `docs/cost_floor.md` §3 found for the cheap-model floor, seen from
the other end.

### 3.1 With the optimal shrinkage — the exact objective

`--mode summary`. `theta*` is the optimal strength for the biased correction and
`adjusted` is §1's exact expression, stacked on the layer-1 variate:

| `L` | `R2_eff` | `dR^2` over `L=1` | rms `b` | `theta*` | adjusted | x ship | `r=2` | `r=4.4` | `r=10` | `r=30` |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 46.15% | 8.4% | 1.56e-03 | 0.124 | 2.486e-07 | 0.99x | 1.03x | 1.08x | 1.12x | 1.13x |
| 4 | 58.21% | 20.5% | 3.36e-03 | 0.069 | 2.471e-07 | 1.00x | 1.06x | 1.21x | 1.38x | 1.45x |
| 8 | 71.64% | 33.9% | 5.26e-03 | 0.047 | 2.463e-07 | 1.00x | 1.07x | 1.33x | 1.79x | 2.09x |
| 16 | 85.96% | 48.2% | 6.02e-03 | 0.051 | 2.428e-07 | 1.02x | 1.13x | 1.62x | 2.82x | 4.05x |
| 24 | 93.83% | 56.1% | 6.68e-03 | 0.049 | 2.418e-07 | 1.02x | 1.15x | 1.77x | 3.96x | 8.26x |
| 30 | 97.90% | 60.2% | 6.81e-03 | 0.050 | 2.406e-07 | **1.027x** | 1.17x | 1.91x | 5.20x | 18.2x |
| 32 | 98.86% | 61.1% | 7.00e-03 | 0.048 | 2.409e-07 | 1.026x | 1.17x | 1.90x | 5.43x | 24.8x |

**The whole deep ladder, done optimally, is worth 1.03x.** `theta*` is 0.05: the
optimum is to apply 5% of a correction that would otherwise remove 61 points of
variance, because the bias term dominates at any larger strength.

### 3.2 The one number that would change everything

The `r = ...` columns are the same table with the analytic layer-mean error
divided by `r`. Two facts:

* The `r` needed to break even against the ship is **4.2-4.6 at every depth
  from 6 to 32** — the trade is scale-free in depth, so there is one number to
  beat, not a curve.
* Past break-even the payoff is steep and it is the *deep* layers that carry it:
  `r = 10` gives 5.4x at `L = 32` (2.8x at `L = 16`), `r = 30` gives 25x.

Where does `r = 4.4` sit against what is known? The plain Gaussian closure
measured here is the `rms 7.0e-03` row; this repository's best analytic arm
(`kappa_3` star diagrams, `docs/state_of_play.md`) is 2.11x better on MSE, i.e.
**`r = 1.45`**; and `scripts/11`'s oracle experiment — feeding a reconstruction
the *exact* cumulants through order 6 on noise-free targets — tops out at
**`r = 2.9`** (`8.2x` on MSE). So `r = 4.4` is past even the exact-cumulant
oracle for the reconstruction step alone, and the remaining error is in the
propagated moments rather than in the final rectifier.

That is the honest statement of where the 10x went: **it is not in the
dictionary, it is in `E[g]`, and reaching it requires an analytic accuracy that
is a factor 3 beyond the exact-low-order-cumulant ceiling this repository
already measured and closed.** The control-variate framing does not evade
`docs/floor_theorem.md`'s obstruction; it relocates it.

### 3.3 The equivalent statement, in one line

A control variate on `h^L` with a claimed mean is exactly a *shrinkage between
the analytic mean and the sample mean of the layer's activations*, direction by
direction: projecting the coefficient vector orthogonal to a subspace is
algebraically identical to taking the sample mean in that subspace. So the
layer-`L` variate is worth using in direction `u` **iff the analytic error along
`u` beats the `N`-sample noise along `u`**. At `L = 1` the analytic error is
zero and the advantage is infinite. At `L = 2` it is already a wash — `c'e =
1.56e-03` against a sampling noise `sqrt(R^2 V/N) = 1.37e-03` in the same
functional. Beyond `L = 2` the analytic mean is simply worse than 27,000
samples.

## 4. The exactly-integrable frontier: degree 2 in the layer-1 *activations*

§2 closed the polynomial route at `f_1 + f_2 = 45.6%` and §3 closed the depth
route at `L = 1`. That leaves exactly one unexplored corner, and it is the one
the exactness boundary of §3 hands us: **`Cov(h^1)` is closed form, so every
quadratic form in the layer-1 activations has an exactly known mean, and
`z^2 = h^1 W^2` is already computed by the forward pass.**

```
E[relu(z^1_i) relu(z^1_j)]  =  arc-cosine kernel of  rho_ij       exact
Cov(z^2)  =  W^2' Cov(h^1) W^2                                    exact
E[ (v'z^2 - v'm^2)(w'z^2 - w'm^2) ]  =  v' Cov(z^2) w             exact
```

This block is **not** in the reach of §2: `relu(z_i) relu(z_j)` is not a
polynomial, it carries every even Hermite degree, so the family is not bounded
by `f_1 + f_2`. And it is not rank-one at degree 2, so it is not bounded by
`docs/hermite_rank_ceiling.md` §5 either. It is the first genuinely new
exactly-integrable dictionary since the shipped one.

`--mode quad --sketch kink`, 3 MLPs x 150,000 samples. Two `k`-dimensional
frames, both derived from the weights alone, both nested so one design serves
the whole sweep:

* `q1` — `u_a u_b - delta_ab` with `u = x A1`: the degree-2 Wiener chaos on an
  adapted orthonormal frame. This is §5.2 of `hermite_rank_ceiling` *built*
  rather than bounded.
* `q2` — `v_a v_b - Cov_ab` with `v = (z^2 - m^2) A2`: the same construction one
  layer later, on the layer-1 activations.

The frames are the top eigenvectors of the **kink second-moment matrix**
`Q = sum_{l,i} (phi(alpha_li)/s_li)^2 ||R^l[i,:]||^2 nhat_li nhat_li'` over all
8,192 neurons, `n_li = grad z^l_i` the mean-field normal and `R^l` the
mean-field Jacobian to the output — i.e. exactly the rank-one carriers of the
degree-2 chaos that §5.2 identified, in `x`-space for `q1` and in `z^2`-space
for `q2`.

3 MLPs, averaged (`x ship` is the ratio of residual variances, i.e. the gain on
`v_eff`):

| dictionary | `p` | `R2_pop` | `R2_eff` at `N=27000` | `1/(1-eff)` | **x ship** |
|---|---|---|---|---|---|
| SHIP `t + He_2` | 512 | 38.48% | 36.58% | 1.577 | 1.0000 |
| `h1 = relu(z^1)` | **256** | 38.63% | **37.68%** | 1.605 | **1.0177** |
| `t` (degree-1 chaos) | 256 | 24.29% | 23.34% | 1.305 | 0.8273 |
| SHIP + `h1` | 768 | 39.41% | 36.57% | 1.576 | 0.9997 |
| `q1` alone, `k=32` | 528 | 8.23% | 6.28% | 1.067 | — |
| **`q2` alone, `k=32`** | 528 | **15.67%** | 13.71% | 1.159 | — |
| SHIP + `h1` + `q1`, `k=32` | 1296 | 43.08% | 38.28% | 1.620 | 1.0275 |
| **SHIP + `h1` + `q2`, `k=32`** | **1296** | **46.46%** | **41.66%** | **1.714** | **1.0871** |
| SHIP + `h1` + `q1` + `q2`, `k=32` | 1824 | 46.50% | 39.74% | 1.659 | 1.0524 |

and the `k` sweep of `SHIP + h1 + q2` (`R2_eff`, so `p/N` is charged):

| `k` | 8 | 16 | 24 | **32** | 48 |
|---|---|---|---|---|---|
| `p` | 804 | 904 | 1068 | **1296** | 1944 |
| `q2` alone, `R2_pop` | 5.64% | 9.36% | 12.58% | 15.67% | 20.51% |
| `R2_eff` | 39.01% | 40.31% | 41.14% | **41.66%** | 41.30% |
| **x ship** | 1.040 | 1.063 | 1.077 | **1.087** | 1.080 |

**`q2` at `k = 32` is 1.087x on `v_eff` over the shipped basis, with an exactly
known mean.** Three things about it:

* **`q2` is worth roughly twice `q1` at every `k`** — 15.7% against 8.2% at
  `k=32`, and the ratio holds from `k=8` to `k=96`. Going one layer deeper in
  the *feature* while staying inside the exactly-integrable region doubles the
  reach of the same number of coefficients, and it does so precisely because
  `relu(z_i) relu(z_j)` is not a polynomial: `q1` is bounded by `f_2 = 19%` and
  `q2` is not.
* `q2`'s population reach keeps growing with `k` (27.5% at `k=96` on MLP
  900000) and `p/N` keeps eating it; the argmax is `k = 32` and the curve is
  flat from 24 to 48.
* `h1` — `relu(z^1)` itself, 256 features — matches the shipped 512-feature
  `t + He_2` basis at **half the coefficients** (38.63% against 38.48%
  population, 37.68% against 36.58% effective, **1.018x**), for free, with an
  exact mean `sigma_i/sqrt(2 pi)`. `relu` mixes degrees 1 and 2 in almost
  exactly the ratio this target wants (73.4% / 23.4% of `Var(relu)`), so one
  feature per direction does what two were doing. Adding it *on top of* the
  shipped pair is worth nothing (0.9997x) — it is a replacement, not an
  addition.

### 4.1 The adapted frame is not a refinement, it IS the mechanism

`--sketch first` repeats the whole sweep with the *unadapted* frame — the first
`k` coordinates of `z^2`, zero setup cost — on the identical samples:

| `k` | 8 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|
| `q1` `R2_pop`, **kink** frame | 2.83% | 4.84% | 6.68% | **8.23%** | 10.88% |
| `q1` `R2_pop`, unadapted frame | 0.13% | 0.33% | 0.52% | **0.78%** | 1.38% |
| `q2` `R2_pop`, **kink** frame | 5.64% | 9.36% | 12.58% | **15.67%** | 20.51% |
| `q2` `R2_pop`, unadapted frame | 1.91% | 3.68% | 5.95% | **7.82%** | 12.10% |
| `SHIP+h1+q2` x ship, kink | 1.040 | 1.063 | 1.077 | **1.087** | 1.080 |
| `SHIP+h1+q2` x ship, unadapted | 1.004 | 1.003 | 1.005 | **0.999** | 0.980 |

**Without the adapted frame the block is worth nothing** — and `q1`, the pure
degree-2 chaos, is worth *literally* nothing (0.13% at `k=8`, which is §5.2's
"any 1 direction reaches 2.9% of `f_2`" seen at the bottom of its own scale). So `q2` is available
only at the price of assembling `Q`, and `Q` cannot be assembled locally: the
kink weight is concentrated at the *deep* end — layers 21-32 carry 87% of it,
layers 28-32 carry 53% — so both the forward normal recursion (`z^2 -> z^l`) and
the backward Jacobian sweep (`z^l -> y`) have to run the full depth. That is
~60 flopscope dispatches, the same object `docs/hermite_rank_ceiling.md` §9.4
measured at ~15 ms = 5.5% of the free budget and which is exactly what turned
`cva`'s 1.032x on raw into 0.945x on the adjusted score.

### 4.2 The frame also validates §5.2's rank bound, tightly

`q1` on the kink frame is `docs/hermite_rank_ceiling.md` §5.2 *constructed*
rather than bounded, so the two can be compared directly. §5.2's
every-`m`-dimensional-subspace-at-once bound is `(1/2) sum_{k<=m} lambda_k(P)`,
i.e. that share of `f_2 = 18.95` points:

| `m` | 8 | 16 | 32 | 48 |
|---|---|---|---|---|
| §5.2 bound, points of `Var(y)` | 3.56 | 6.20 | 10.23 | ~12.8 |
| kink frame reaches (`q1` `R2_pop`) | **2.83** | **4.84** | **8.23** | **10.88** |
| fraction of the bound | 79% | 78% | 80% | 85% |

The kink frame gets 78-86% of a bound that holds over *every* subspace, at
every `m` — so it is close to the optimal shared frame, and §5.2's numbers were
not loose. (For comparison, that page's mean-field frame reached 4.3 of the 6.2
points at `m = 16`, i.e. 69%; the kink frame's 4.84 is a 13% improvement on it,
which is the only place this round improves on a previously published figure.)

### 4.3 What it would cost to ship

Per sample, given that the forward pass has already produced `z^2`:

```
v = (z^2 - m^2) A2            k n 2   = 16,384 FLOPs at k = 32
v_a v_b products              k(k+1)/2 =    528
CV accumulate (d, w, y'w)     ~3 p     =  1,840
                              ------------------
                              ~1.9e4 FLOPs = 0.65% of c = 2.79e6
```

and the Gram need not be estimated: under a Gaussian model for `z^2` — an
approximation that costs *efficiency only, never bias, because the mean is
exact independently of it* — Wick gives
`Cov(v_av_b - C_ab, v_cv_d - C_cd) = C_ac C_bd + C_ad C_bc` analytically from
the exact `Cov(z^2)`, so the `528 x 528` solve is a setup-time
`4.9e7` FLOPs and there is no `N p^2` covariance pass.

**The honest caveat is the frame, and it is the same one that killed `cva`.**
The kink weight is concentrated at the *deep* end — layers 21-32 carry 87% of
it and layers 28-32 carry 53% — so `Q` cannot be assembled from a local
computation; it needs both the forward normal recursion and the backward
Jacobian sweep, ~60 flopscope dispatches, which `docs/hermite_rank_ceiling.md`
§9.4 measured at ~15 ms, i.e. 5.5% of the free budget against this block's
8.7%. That trade is exactly the one the shipped `N` is currently being tuned
against and is left to the integrator. The zero-setup alternative is not a
partial fallback: §4.1 measures the unadapted frame at **1.004x**, so dropping
the sweep drops the block. One thing that may soften the bill and is worth a
measurement rather than an argument: the shipped path *already* runs a 31-step
mean-field propagation for `cv1mf`, at the same shapes, so the marginal
dispatch cost of a second sweep may be well below the 5.5% `cva` paid cold.

### 4.4 The end-to-end check, and what it takes back

`R^2` is a claim about a *jointly fitted* dictionary. What ships is a sum of
blocks, each with its own within-block coefficients and one scalar weight that
the offline head learns. `--mode mse` measures that object directly: 48
generated MLPs, `N = 27000`, unbiased true MSE against two independent
150,000-sample references, **no fitting anywhere** — every `theta` below is one
global constant, the same for every MLP, which is exactly what the head can
learn. Paired bootstrap over MLPs in the `+-` column.

| arm | unbiased true MSE | x plain | x `-cv1-cv2` | `+-` |
|---|---|---|---|---|
| plain | 1.6134e-06 | 1.000 | 0.562 | 0.091 |
| `-cv1` | 1.2232e-06 | 1.319 | 0.741 | 0.073 |
| **`-cv1-cv2` (the shipped mechanism)** | 9.0638e-07 | 1.780 | **1.000** | — |
| **`-h1`** | **8.9366e-07** | **1.805** | **1.014** | 0.040 |
| `-q2` alone | 1.8256e-06 | 0.884 | 0.497 | 0.115 |

and the two-dimensional block-weight sweep, `mu - cv1 - theta_2 cv2 - theta_q q2`:

| `theta_q` = | 0 | 0.15 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| `theta_2 = 1.00` | **1.000** | 0.982 | 0.944 | 0.871 | 0.647 |
| `theta_2 = 0.75` | 0.985 | 0.978 | 0.950 | 0.887 | 0.672 |
| `theta_2 = 0.50` | 0.927 | 0.931 | 0.914 | 0.867 | 0.676 |
| `-h1 - theta_q q2` | **1.014** | 1.003 | 0.970 | 0.901 | 0.674 |

**`h1` delivers what §4 predicted and `q2` does not.** `h1` is 1.014x measured
against 1.020x predicted, inside one standard error — it is a *replacement* for
`He_2`, so overlap never arises. `q2` is a loss at every one of the 15 block
weights tried; the argmax of the whole grid is `theta_q = 0`, i.e. the shipped
basis.

**Mechanism, and it is `cva`'s, exactly.** More than half of `q2`'s population
span is already inside `SHIP + h1`: 39.41 + 15.67 = 55.1 points of separate
content collapse to 46.46 jointly, so 8.6 of `q2`'s 15.7 points are duplicates.
A separately-optimal `q2` correction therefore re-removes signal `cv1 + cv2`
has already removed, and it does so carrying its own estimation error — which
is worse than the layer-1 blocks' because the features are *products*, so their
fourth moments are far from Gaussian and the coefficient noise is heavier at
equal `p`. Shrinking by `theta` scales the duplicated signal and the noise
together, so there is no `theta > 0` that wins.

The joint 1.10x is therefore a **ceiling that needs joint coefficients**, not a
drop-in. Realising it means residualising `q2`'s features against the layer-1
blocks before forming its coefficients, which needs the cross-block Gram:
`N p_1 p_2 = 27000 x 768 x 528 = 1.1e10` FLOPs, **4% of `B` and +44% of the
shipped `F`**. At the clamp that costs 44% of `N`, i.e. 1.44x of variance,
against a 1.10x gain. It does not close.

> **A held-out `R^2` measured with a joint fit is an upper bound on a shipped
> sum of blocks, and the gap is exactly the overlap.** `docs/hermite_rank_ceiling.md`
> §9.3 found the same thing for `cva` from the other side (the head had already
> bought it); here the block is genuinely new content and the overlap still
> eats it. Two rounds, one lesson: **report the increment jointly fitted AND at
> deployable block weights, or the number does not mean anything.**


## 5. Symmetry-derived variates: exact, and worth less than nothing

Two exact symmetries need no model at all.

**Antithetic.** `x -> -x` preserves `N(0, I)` exactly, so `(y(x) + y(-x))/2` has
the same mean as `y` and — because `y(-x)`'s degree-`d` chaos is `(-1)^d` times
`y(x)`'s — contains **no odd Hermite degree whatsoever**. Measured: the pair
mean has **46.4%** of a single draw's variance (`f_even`), i.e. **1.077x** on raw
variance per unit cost.

**Homogeneity.** The networks have no biases
(`whestbench.generation.sample_mlp`), so `y` is *exactly* positively
homogeneous of degree 1: `y(x) = ||x|| Y(x/||x||)` with `||x||` and `x/||x||`
independent. `E||x||` is `sqrt(2) Gamma((n+1)/2)/Gamma(n/2) = 15.9843826666`
in closed form, so `mu_hat = E||x|| * mean_s y(x_s)/||x_s||` is an exactly
unbiased Rao-Blackwellisation of the radius. Measured: **1.038x** on raw
variance.

Both are free, both are exact, and both **lose** once composed with the shipped
control variate (`--mode anti`, 3 MLPs x 150,000 samples, residual variance per
unit of forward-pass cost, relative to plain MC + the shipped basis):

| scheme | var/cost | `R^2` of the shipped basis on it | residual/cost | x ship |
|---|---|---|---|---|
| plain MC + `t`,`He_2` | 1.000 | 32.9 / 43.1 / 33.8% | 1.000 | **1.000** |
| antithetic pair | 0.928 | 24.4 / 35.0 / 26.9% | 1.043 | **0.958** |
| homogeneity RB | 0.963 | 29.6 / 38.3 / 29.8% | 1.026 | **0.975** |
| both | 0.857 | 16.5 / 23.2 / 17.5% | 1.096 | **0.913** |

**Mechanism, and it is the same one twice.** Antithetic pairing removes the odd
degrees — but 26.7 of the 45 points of odd-degree variance are degree 1, which
the control variate already removes *exactly and at half the cost*. So the
pairing spends a factor of two in samples to buy the odd degrees `>= 3` (which
the CV cannot reach) and simultaneously destroys the channel the CV was living
on: `R^2` falls from 32.9% to 24.4%. The radial variate is the same story —
`||x||^2 - n` is `sum_i He_2` on any orthonormal frame, so the degree-2 block
already contains it, and dividing by `||x||` removes 3% of the variance while
costing 3.4 points of `R^2`.

> An exactly-mean-zero variate is worth nothing if the dictionary you already
> ship spans it. Both of these are inside the shipped span, and both are
> strictly more expensive per unit of it.

## 6. The official suite

Nothing in §1-§5 opens the official suite. This section does, once, after every
frame rule, `k` and dictionary was frozen, purely to confirm that a conclusion
measured on look-alike networks applies to the graded ones. No parameter is
fitted or selected from it, and the shipped estimator is not modified.

`--mode quad --official`, the first 4 MLPs of `official_mini.npz` by seed
protocol 3.0:

| dictionary | local (3 MLPs) `R2_pop` | official (4) `R2_pop` | local x ship | official x ship |
|---|---|---|---|---|
| SHIP `t + He_2` | 38.48% | **39.69%** | 1.000 | 1.000 |
| `t` (degree-1) | 24.29% | 25.96% | 0.827 | 0.830 |
| `h1 = relu(z^1)` | 38.63% | 40.07% | **1.018** | **1.022** |
| `q2` alone, `k=32` | 15.67% | 14.09% | — | — |
| `SHIP+h1+q2`, `k=16` | 43.66% | 44.54% | 1.063 | 1.058 |
| **`SHIP+h1+q2`, `k=32`** | 46.46% | 46.98% | **1.087** | **1.076** |
| `SHIP+h1+q2`, `k=48` | 48.50% | 48.84% | 1.080 | 1.066 |

Same argmax (`k = 32`), same ordering, every gain within 1.1% of the local
figure. The `SHIP` row's 39.69% is *bit-for-bit* the official number
`docs/hermite_rank_ceiling.md` §7 published for `coord d<=2`, from an
independently written span estimator — which is the tightest cross-check
available that this page's machinery measures the same object as that one.

`--mode ladder --official`, 2 MLPs, `L` in `{1,2,4,8,16,24,32}`, same
2e6-sample reference protocol:

| `L` | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| `R2_eff`, official | 41.36% | 50.22% | 59.62% | 69.66% | 83.60% | 92.95% | 98.87% |
| `R2_eff`, local | 37.73% | 46.15% | 58.21% | 71.64% | 85.96% | 93.83% | 98.86% |
| rms `b`, official | **5.9e-05** | 1.42e-03 | 3.16e-03 | 4.22e-03 | 5.33e-03 | 5.91e-03 | 6.36e-03 |
| rms `b`, local | **8.8e-05** | 1.56e-03 | 3.36e-03 | 5.26e-03 | 6.02e-03 | 6.68e-03 | 7.00e-03 |
| x ship, official | **1.036** | 0.611 | 0.212 | 0.129 | 0.085 | 0.070 | 0.061 |
| x ship, local | **0.974** | 0.536 | 0.190 | 0.086 | 0.067 | 0.055 | 0.050 |

Same argmin, same monotone collapse, and the layer-1 bias is again at the
reference's noise floor — layer 1 is exactly integrable on the graded networks
too. The official closure errors are 6-20% *smaller* than the local ones at
every depth, which moves nothing: the break-even `r` is 4.0-4.3 instead of
4.2-4.6.

## 7. Bars, fixed before each run

| # | bar | measured | verdict |
|---|---|---|---|
| 1 | the top-8 eigenfunctions are `> 75%` degree `<= 2` in `x` (which would make their means exact by Wick) | **51.4%**, and the full degree-`<=2` space is 45.6% of `Var(y)` for 32,896 coefficients (`p/N = 1.22`) | **FAIL** |
| 2 | some layer of the depth ladder beats the shipped adjusted 2.47e-07 | argmin is `L = 1` at 2.536e-07 (0.97x); every `L >= 2` is worse; with optimal shrinkage the whole ladder is 1.03x | **FAIL** |
| 3 | a symmetry-derived variate is `> 1.05x` on residual per unit cost | antithetic **0.958x**, homogeneity **0.975x**, both **0.913x** | **FAIL** |
| 4 | a new **exactly integrable** dictionary reaches held-out `R^2 > 60%` net of `p/N` | best is SHIP + `h1` + `q2(k=32)` at **41.7%** jointly fitted (1.71x against 1.58x) | **FAIL** |
| 4b | ...and that gain survives end to end at deployable block weights | `h1` **1.014x** (predicted 1.020x); `q2` **<= 1.000x** at all 15 block weights, argmax is `theta_q = 0` | **`h1` PASS, `q2` FAIL** |
| 5 | the conclusion transfers to the official suite | §6: same argmax, every gain within 1.1%, and the `SHIP` row reproduces `hermite_rank_ceiling` §7's official 39.69% exactly | **PASS** |
| 6 | the exactness claims are asserted, not assumed | `tests/test_integrable_cv.py`: layer-1 Mehler == arc-cosine to 1e-12, `E[h^1]`/`Cov(h^1)`/`E[z^2]`/`Cov(z^2)` within 6 sigma of 2e6 samples, `E[z^3]` NOT (>20 sigma), antithetic kills degree 1, `y(3x) = 3y(x)` to 1e-10 | **PASS** |
| 7 | shipped estimator untouched | `git diff submission/` empty | **PASS** |

Bar 4's 60% was the brief's, and it was the right bar: it is the level at which
a new basis would be worth more than the whole cost frontier. Nothing exactly
integrable reaches it, and §3 says why — 60% starts at `L = 4`, and `L = 4`
costs `rms b = 3.4e-3`.

## 8. What this leaves

**The lever is real and it is not the dictionary.** Stack the three closures:

* `docs/cost_floor.md`: `c` is closed at 1.13x.
* `docs/hermite_rank_ceiling.md`: the rank-one Hermite family is closed at
  1.76x on `v_eff`, of which 1.65x is realised.
* this page: the *exactly integrable* extension of that family has a joint-fit
  ceiling of 1.71x (SHIP + `h1` + `q2(k=32)` against the shipped 1.58x) and a
  **deployable** value of 1.014x, because the only genuinely new block overlaps
  the shipped ones and the cross-block Gram that would separate them costs 44%
  of `N`. And the 90%-explaining functions that would give 10x exist, are free
  to evaluate, and are unusable **only** because `E[g]` is not known to better
  than `rms 7e-3`.

Everything now points at one scalar. Writing `r` for the factor by which the
analytic layer-mean accuracy would have to improve:

| `r` | 1 | 2 | **4.4** | 10 | 30 |
|---|---|---|---|---|---|
| best layer-`L` control variate | 1.03x | 1.17x | **1.91x** | 5.4x | 25x |
| what that `r` corresponds to | today's Gaussian closure | ~this repo's `kappa_3` arm (1.45) | past `scripts/11`'s exact-cumulant oracle (2.9) | — | — |

So the honest next question is not "which basis" but **"can the mean of a deep
layer's activation be computed to 4.4x better than the Gaussian closure"** — and
`scripts/11`'s oracle experiment says that even *exact* cumulants through order
6 only buy 2.9x on the reconstruction step, so the remaining factor has to come
from the propagated moments, not from a better rectifier integral. That is a
different programme from control variates, it is the one
`docs/floor_theorem.md` and `docs/cumulant_expansion.md` already went at, and
this page's contribution is to price exactly what it would be worth: **1.9x at
`r = 4.4`, 5.4x at `r = 10`, and the whole 25x at `r = 30`.**

**One thing is live**, and it is small: swap the 256-feature `relu(z^1)` block
in for the 256-feature `He_2(t)` block. Same cost (the forward pass already has
`relu(z^1)`), exact mean `sigma_i/sqrt(2 pi)`, analytic arc-cosine Gram, half
the coefficients, and **1.014x** on unbiased true MSE over 48 generated MLPs
against a 1.020x prediction. `whestfloor.corrector.relu1_cv`. It is a
replacement, not an addition — stacking it on top of `t + He_2` is 0.9997x.

**And one methodological correction worth carrying forward.** Rounds 9 and 12
both produced a real held-out `R^2` gain that evaporated on the way to the
score, for two different reasons — `cva` because the fitted head had already
bought it, `q2` because the block overlaps what is already being subtracted and
a per-block scalar cannot undo an overlap. The fix is procedural: **quote every
new dictionary twice, jointly fitted AND at deployable block weights on
unbiased true MSE.** `--mode quad` and `--mode mse` are those two halves and
should be run together from now on.
