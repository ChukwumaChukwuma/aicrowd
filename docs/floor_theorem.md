# The floor is a theorem, and it bounds everyone

> **Scope.** The `vbar/N` floor in §1 binds *every* estimator, iid or not — it
> is a statement about the reference's own noise and uses nothing about how the
> estimator samples. Everything downstream of it that mentions a **rate** does
> not. In particular the sentence "its asymptotic variance is
> `(1 - f_1) sigma^2 / N`" in the low-order-barrier section assumes the
> order-`>= 2` remainder converges at the **iid** Monte Carlo rate, and the
> retrodiction table compares mechanisms at a fixed `N`. Measured
> (`docs/rqmc.md`): a randomly-shifted rank-1 lattice runs at `p = 1.078` in
> `v = v_0/N^p` against iid's `p = 1.002`, and beats iid by 1.6–2.1x at matched
> `N` — i.e. the assumption is nearly right about the exponent and wrong about
> the constant. See the amendment at the end of the barrier section.

## Statement

Let `p` be any estimator measurable with respect to the information available at
grade time — the weights and `mlp.seed`. The baked reference is

    mu_ref = mu_true + eps ,    eps independent of p ,    Var(eps_j) = v_j / N

Then

    E[mse] = (1/n) sum_j E[(p_j - mu_true,j - eps_j)^2]
           = (1/n) sum_j (p_j - mu_true,j)^2  +  vbar/N
           >= vbar/N

The cross term vanishes because `eps` is independent of `p`. Equality holds iff
`p = mu_true` exactly.

**The only way below `vbar/N` is to predict `eps`** — a deterministic function of
the grader's private sampling stream (`SeedSequence(input_seed).spawn(3)[1]`,
consumed by a torch generator during the bake). Not available at grade time, and
not inferable from the weights: the weight stream is `spawn(3)[0]`, a
sibling, and SeedSequence spawning is not invertible.

## Numbers

Measured on the official 100-MLP suite (`scripts/19_fetch_official_suite.py`):

    vbar = 0.049491 +- 0.003336   (CV 0.674, median 0.03655)
    N    = 1e9

    raw floor       = vbar/N            = 4.949e-11
    adjusted floor  = 0.1 * vbar/N      = 4.949e-12    (multiplier is max(0.1, C/B))
    per-neuron RMS a perfect estimator still shows = 7.03e-6

## Consequences

| | adjusted | x floor |
|---|---|---|
| noise floor (nobody can go below) | 4.949e-12 | 1.0 |
| leaderboard #1 (dpskv5) | 3.63e-10 | 73.3 |
| grader's plain-MC constant `sampling_mse` | 6.4695e-7 | 130,700 |
| this repo, shipped | 7.7791e-7 | 157,200 |

**The entire remaining headroom in this benchmark is 73.3x.** A request to beat
the leader by 10,000x asks for 3.63e-14, which is 136x below the floor. It is not
difficult; it is impossible, and the impossibility is a two-line consequence of
independence.

## What the floor demands of a method

To reach it from a 2.72e10 free budget (6479 forward passes):

    required per-sample variance = vbar^2/(N * vbar) ... = 0.049491/1e9 * 6479
                                 = 3.2e-7
    natural per-sample variance  = 0.0495
    required variance reduction  = 154,000x
    equivalently, a control variate with rho = 0.9999968

The leader's 3.63e-9 raw corresponds to an effective 1.36e7 samples, i.e. a
~2100x variance reduction. So the state of the art is 2100x and the ceiling is
154,000x.

## Two structural results that constrain the search

**1. The exact tail decomposition, and why it is only usable at the end.**
Since `relu(z) = z - z*1{z<0}`,

    mu^l = mu^{l-1} W^l - T^l ,   T^l_j = E[z^l_j 1{z^l_j < 0}]

which unrolls (using mu^0 = 0) to the exact identity

    mu^32 = - sum_{l=1..32} T^l P^l ,    P^l = W^{l+1} ... W^32

`T` is a tail functional: for |alpha| ~ 4.4, Var(T) ~ sigma^2 phi(alpha)/alpha^3
~ 5.8e-7 against 0.05 for the rectifier — a 1e5 variance gap. But
`E||P^l v||^2 = 2^{32-l} ||v||^2`, so a level-l error is amplified 2^{32-l};
at l=1 that is 2.1e9 and the naive estimator is far worse than plain MC. The
identity is exact but ill-conditioned, and is only worth using over the last
few layers, where the amplification is small and the variance gap survives.

**2. Input-anchored control variates are bounded at 1.33x -- but the effective
dimension is LOW, and an earlier version of this section got that wrong.**

CORRECTION. This section previously claimed "only ~6% of Var(z^32_j) is
explained by the best linear function of the input", and concluded from it that
the integrand has *high* effective dimension so input-space RQMC cannot work.
The 6% was wrong. It was inferred from a forum-reported `rho = 0.25`, which is
the correlation of a particular *Gaussian-affine surrogate chain*, not of the
L2-optimal linear functional. Measured directly (12 official MLPs, 163,840
samples, `scripts/27_anova.py`), by three mutually validating estimators
(Hermite split-half projection, Jansen one-coordinate resample, and
pick-and-freeze on Bernoulli(p) coordinate subsets recovering the whole order-
generating function):

    linear (Hermite k=1) share of Var    24.6%      (not 6%)
    first-order ANOVA share f_1          27.6% +- 1.6%   (range 20.3-37.5%)
    mean ANOVA dimension d_M             11.5 +- 0.9     (out of 256)

Two consequences, in opposite directions:

* The *conclusion* about control variates survives, with a corrected constant:
  a perfect linear control variate caps at `1/(1 - 0.246) = 1.33x`, nowhere near
  the 154,000x required. Input-anchored control variates remain dead.
* The *conclusion* about RQMC does not survive. A mean ANOVA dimension of 11.5
  out of 256 is LOW, which is precisely the regime where randomised QMC
  outperforms plain Monte Carlo. The dismissal was based on a wrong input
  number, and the reasoning step it fed ("low linear R^2 implies high
  superposition dimension") is roughly sound here but was applied to a figure
  that was 4x too small in R^2.

The remaining 72.4% of the variance sits in interactions of order >= 2, which
bounds how much RQMC can buy: the first-order part can be integrated at a much
faster rate, the remainder converges at the Monte Carlo rate.

## Theorem (low-order barrier) — why every surrogate family failed

Measured on 12 official MLPs (`scripts/27_anova.py`, three mutually validating
estimators): first-order ANOVA share `f_1 = 0.276 +- 0.016`, mean ANOVA
dimension `d_M = sum_d d f_d = 11.5 +- 0.9`.

Therefore the non-additive mass sits at average ANOVA order

    dbar_{>=2} = (d_M - f_1) / (1 - f_1) = (11.5 - 0.276) / 0.724 = 15.5

**Bound.** Let `h` be any surrogate whose ANOVA content is concentrated in
orders `<= k`. Its maximal correlation with the integrand is
`rho = sqrt(sum_{d<=k} f_d)`, so the variance reduction from using it as a
control variate is at most `1 / (1 - sum_{d<=k} f_d)`. For `k = 1`:

    max variance reduction = 1 / (1 - 0.276) = 1.38x

**This retrodicts every measurement in the ledger:**

| mechanism | measured | bound |
|---|---|---|
| optimal linear control variate | 1.33x | 1.38x |
| MLMC over rank-truncated nets | 0.94x | 1.38x |
| Rao-Blackwell on decided neurons | 1.001x | 1.38x |
| RQMC (first-order accelerated, rest at MC rate) | **1.60–2.13x** | 1.38x, and it is **exceeded** |

RQMC deserves a word because it is not a control variate: it integrates the
first-order part at a much faster rate and leaves the remainder at the Monte
Carlo rate, so its asymptotic variance is `(1 - f_1) sigma^2 / N` — the SAME
1.38x ceiling, reached from a different direction.

**That last row is now measured, and the prediction was wrong in the
interesting direction.** `docs/rqmc.md` §2: a randomly-shifted rank-1 lattice,
six official MLPs, `N` = the primes just under `2^10 … 2^17`, variance across
independent randomisations, iid arm as the control. The gain over iid at
matched `N` is 1.60x / 1.80x / 2.01x / 1.73x / 2.13x at
`N = 1021 / 2039 / 4093 / 8191 / 16381` — **above** the 1.38x this table
predicted, because the lattice also reaches part of the order-2 mass, not only
`f_1`.

What survived is the *shape* of the prediction, which is the part that matters:
it is a **constant, not a rate**. Fitted exponents in `v = v_0/N^p` are
`p = 1.002 ± 0.065` for iid — the control, and it lands on 1 — against
`p = 1.078 ± 0.059` for the lattice. `1 - f_1` still governs the asymptotics,
so the ratio saturates rather than growing; it just saturates somewhat higher
than `1/(1 - f_1)`.

**Scope warning: every bound on this page is a bound on `1/Var` at a fixed `N`,
not on the exponent.** The retrodiction table compares mechanisms at one sample
count. A mechanism that changed `p` would not be bounded by anything here, and
the sentence "its asymptotic variance is `(1-f_1) sigma^2/N`" quietly assumed
that the order-`>= 2` remainder converges at exactly the Monte Carlo rate — an
assumption, not a theorem, and the one place this section could have failed
badly. It happens to be nearly right: the measured `p` is 1.08, not 2. But the
argument was not entitled to it.

**Constructive content.** A surrogate that helps must itself carry ANOVA mass at
order ~15. Exactly two objects do:

1. **The network itself.** Measured: coupling `rho >= 0.975` needs `r = 251` of
   256, at 1.96x the dense cost. Self-defeating (`docs/mlmc.md`).
2. **A learned map fitted to the weight -> mean relation.** Not built from
   low-order structure, so not subject to the bound. Training and offline
   precomputation are unbounded and load at zero cost at grade time, and the
   submission may carry 50 MiB (~13M float32 parameters).

Route 2 is the only family the barrier does not close, and the competition
forum independently reports an offline corrector among the mechanisms that work.

### Amendment: the bound was only ever *computed* at k = 1

The theorem above is stated for general `k` and then instantiated at `k = 1`,
because `f_1` was the only order share that had been measured. `f_2` had not
been, and it is not small.

Measured on 4 local MLPs x 400,000 samples
(`scripts/28_learned_corrector.py --mode anova`), by exact projection onto the
**layer-1 Hermite family** — the statistics `He_k(z^1_i / ||W^1[:,i]||)`, whose
population Gram is analytic by Mehler and whose expectations are exactly zero:

| cumulative share of `Var(relu(z^32_j))` | mlp 0 | mlp 1 | mlp 2 | mlp 3 | bound |
|---|---|---|---|---|---|
| `k = 1` | 23.4% | 29.4% | 23.5% | 29.5% | 1.31-1.42x |
| `k <= 2` | 37.8% | 47.9% | 38.5% | 44.7% | **1.61-1.92x** |
| `k <= 3` | 39.6% | 50.3% | 40.7% | 46.8% | 1.66-2.01x |
| `k <= 6` | 43.4% | 55.9% | 45.0% | 50.9% | 1.77-2.27x |

The `k = 1` row reproduces the 1.38x bound and its `f_1 = 0.276`, from a fifth
independent estimator — the bound is correct. But **`f_2` is about 0.16 along
the network's own first-layer directions**, so the `k = 2` instance of the same
theorem is `1/(1 - 0.43) = 1.75x`, not 1.38x, and it is *reachable*: those
statistics are already computed by the forward pass, their expectations are
known in closed form, and contracting the correction costs two length-N
matvecs (0.4% of the scored pass). `docs/learned_corrector.md` builds it.

Two things this does *not* change:

* Every mechanism in the retrodiction table stays refuted. They were all `k = 1`
  objects — a Jacobian response *is* the linear response — and 1.38x is still
  their ceiling. (Measured: mean-field propagation of the layer-1 mean gap
  lands at 1.33x with pilot gates, 1.37x with scored gates, and no higher.)
* The bound still bites hard above `k = 2`. `k = 3` adds ~2% of explained
  variance against `p/N = 256/8500 = 3.0%` of estimation noise per block, so it
  loses; and the full order-2 basis (all 32,896 quadratics in the input, rather
  than the 256 along `W^1`) has `p > N` and is unusable at this sample count
  whatever it costs.

The honest summary is that the barrier is a statement about **which orders you
can reach**, and the layer-1 Hermite family reaches order 2 for free because
the first layer of a ReLU MLP is exactly Gaussian in the input. Nothing
comparable is available at layer 2 or beyond, which is where this line stops.

### Second amendment: ANOVA order is not Hermite degree, and the family is now closed

`docs/hermite_rank_ceiling.md` measures the **Hermite degree** spectrum
directly, by the Ornstein-Uhlenbeck semigroup (`Cov(y(x), y(x_t)) = Σ_d f_d
t^d` for the Mehler coupling `x_t = t x + √(1−t²) ξ`). Two corrections to how
this page has been read:

* **`d_M = 11.5` is the mean ANOVA (superposition) dimension, not the mean
  degree.** `He_2(x_1)` involves one coordinate and has degree 2. The mean
  *Hermite degree* is **10.5**, measured separately; the two agree in spirit
  but they are different objects and only the degree one bounds a Hermite
  dictionary.
* **The degree spectrum is** `f = (26.7, 19.0, 11.2, 7.0, 4.7, 3.3, …)%`, and
  the `f_1 = 26.7%` entry reproduces the `f_1 = 0.276 ± 0.016` above from a
  seventh independent estimator. So 54% of `Var(y)` really does sit above
  degree 2 — the barrier's `k ≤ 2` instance is *not* an approximation-power
  statement about the target.

It is a statement about what a Hermite dictionary can *reach*.
`h_d(⟨a,x⟩) = Σ_{|α|=d} √(d!/α!) a^α H_α` is exactly a unit **rank-one**
symmetric tensor of the degree-`d` chaos, so a dictionary on `m` directions
lives in `Sym^d(span A)` and is capped by `Σ_j Var(E[y_j | Aᵀx])` at every
degree at once. Measured: 2,494 features spanning degree ≤ 16 on every
direction basis derivable from the network reach **43.2%**, against the
shipped `k ≤ 2` basis's **39.2%** — i.e. `1.76×` against `1.65×`. The
`1/(1 − 0.43) = 1.75×` quoted above is therefore not just the `k ≤ 2` ceiling;
**it is the ceiling for the whole layer-1 Hermite family at any degree.**
