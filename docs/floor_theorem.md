# The floor is a theorem, and it bounds everyone

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

**2. Input-anchored control variates cannot work, and neither can input-space RQMC.**
Only ~6% of Var(z^32_j) is explained by the best linear function of the input
(forum-reported rho = 0.25 at depth 32, consistent with our own measurement that
the fluctuation has participation ratio 2.2 but is *not* linear in x). So the
integrand has high effective dimension in input coordinates. Any surrogate
correlated ~0.9998+ with the pathwise output must therefore be *another network*,
not a functional of the input — which is what makes multilevel Monte Carlo over
rank-truncated weights the natural candidate: Var(f_full - f_r) is controllable
by r, and level r costs n/(2r) less per sample.
