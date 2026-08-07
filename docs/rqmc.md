# Randomised quasi-Monte Carlo, and the number that decides it

**Headline.** The lattice is real and it is a **constant**, not a rate. Measured
on the official MLPs over seven doublings of `N`, the randomly-shifted rank-1
lattice converges at `p = 1.08`, against iid Monte Carlo's `p = 1.00`, and buys
a **1.6–2.1x variance reduction** at the sample counts we ship at. And the
premise that the top of the leaderboard is doing this — that its `p ~ 2` is a
lattice rate and ours is not — is **refuted by the leaderboard's own
telemetry**: within an entry, raw MSE is flat in `N`.

Construction credit: **evaaaz** (forum 18053). The redundancy warning and the
antithetic refutation that shape how it is measured here: **radiant-allomancer**
(18085). The float64 inverse-CDF trap: **jamesrahenry** (18097 erratum) and
**mohanty** (18125).

---

## 1. The premise, and why the telemetry refuses it

The case for building this was a fitted convergence exponent. Take
`N_eq = F / 4.198656e6` (billed FLOPs in dense-forward-pass equivalents),
assume `raw = v / N_eq^p` with `v = 0.045`, and solve for `p` at each entry's
single graded point:

| entry | `N_eq` | raw MSE | `p_implied` |
|---|---|---|---|
| huang_chung_yi | 2,262 | 2.99e-8 | 1.84 |
| dpskv5 | 5,792 | 3.63e-9 | 1.89 |
| joe_wanza | 11,163 | 3.95e-9 | 1.74 |
| ednacob | 30,524 | 9.31e-8 | 1.27 |
| us (shipped) | 14,817 | 1.08e-6 | 1.11 |

`p = 1` is iid Monte Carlo, `p ~ 2` is the classical lattice rate, so this reads
as though the top of the board found the lattice and we did not.

**One point cannot identify two parameters.** `p_implied = ln(v/raw)/ln(N_eq)`
is a monotone recoding of `raw` at fixed `N_eq`; it returns a number for a
deterministic estimator, for a biased estimator and for an estimator whose
per-sample cost is not one dense pass, and in none of those cases is it a
convergence exponent. Two things in the telemetry settle it.

**(a) Four of dpskv5's graded submissions ran at `N_eq = 0.5`** — half a dense
forward pass of billed compute for the entire 100-MLP prediction — and scored
raw 5.0e-8 to 7.9e-8. A sampler with `v = 0.045` at `N_eq = 0.5` scores 0.09.
Those entries are 1e6x better than any sampler can be at that compute, so their
error is model error. dstepanov (`N_eq = 3.0`, raw 1.05e-7) and ely2sh
(`N_eq = 3.5`, raw 5.6e-8) are the same. `p_implied` for them is 10.9–11.6,
which is not a rate; it is a ratio with a vanishing denominator.

**(b) The identifiable fit — regress `ln raw` on `ln N_eq` *within* an entry,
which eliminates `v` — says the rate is zero.** Every top entry that varied its
own `N`:

| entry | subs | `N_eq` range | raw range | **fitted `p`** |
|---|---|---|---|---|
| dpskv5 | 4 | 152 – 42,028 | 3.56e-9 – 2.88e-8 | **0.310** |
| huang_chung_yi | 2 | 202 – 2,262 | 2.99e-8 – 2.14e-7 | **0.814** |
| joe_wanza | 2 | 11,163 – 21,456 | 3.95e-9 – 4.36e-9 | **−0.151** |

And the sharpest pair in the whole dataset, because it is one team 3.5 hours
apart and it is the pair that took them to rank 1:

```
dpskv5 324846  2026-08-06 09:05  N_eq 42,028  raw 3.5550e-09  adjusted 5.8272e-09
dpskv5 324969  2026-08-06 12:33  N_eq  5,792  raw 3.6284e-09  adjusted 3.6297e-10
```

They cut compute **7.26x** and raw moved **+2.1%**. Fitted `p = 0.010`. Their
16.1x jump in adjusted score came entirely from the multiplier — from landing on
the `max(0.1, C/B)` clamp — and none of it from raw. Had `p` been 1.9, that cut
would have multiplied raw by 43x, to 1.5e-7.

**So the top of the board is bias-limited, not sampling-limited**, and it ranks
by driving `C` to the clamp. That is `docs/graded.md` §5's conclusion, now
measured from the telemetry rather than inferred from an inequality.
`scripts/52_leader_rate.py` is the whole analysis and prints the table above.

None of this makes the lattice worthless. It makes the lattice a
**variance-reduction constant** to be priced like any other, rather than a rate
change that would reorganise the whole cost argument. §2 measures the rate on
our own estimator, where the same question *is* identifiable because we control
`N`.

---

## 2. The rate, measured over seven doublings

`scripts/50_rqmc_rate.py --mode rate`. Six official MLPs, dense forward pass
(no mask, no pilot, so the measured quantity is the sampler's own variance and
nothing else), `N` = the primes just under `2^10 … 2^17`, and the variance taken
across independent randomisations — which for an unbiased estimator *is* its
MSE, so no ground truth and no reference noise enter. Both arms share seeds
(common random numbers) and MLPs.

*(table filled by the run; see `$WHEST_ARTIFACTS/rqmc/rate.log`)*

The iid arm returning `p = 1.00` is the control: it is the one number in this
document whose value is known in advance, and it comes back at
`1.002 ± 0.065`.

### Why the rate is ~1 and not ~2, in one paragraph

A rank-1 lattice with `gcd(z_j, N) = 1` has **exact** `N`-point equidistribution
in every one-dimensional projection, so the shift-averaged squared error of any
*first-order* ANOVA term is `1/(6N²)` against Monte Carlo's `1/N` — annihilated,
for any generating vector, which is why the CBC search only ever buys the pairs.
`docs/floor_theorem.md` measures the first-order share of `Var(relu z³²)` at
`f_1 = 27.6% ± 1.6%`. The other 72.4% sits in interactions of mean ANOVA order
15.5, and a 256-dimensional point set does not integrate those at `O(N^-2)` —
the integrand is Lipschitz but kinked, not in the mixed-derivative Sobolev space
the `O(N^{-2+ε})` theorems need. So the variance is
`(1-f_1)σ²/N + f_1 σ² c/N²`: the second term dies, the first does not, and the
asymptotic ratio tends to `1/(1-f_1) = 1.38x` with `p → 1`.

The measured ratio is **above** 1.38, so the lattice is also reaching some of
the order-2 mass — but it is reaching it as a constant, not as a rate.

---

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

## 5. Dimension ordering

*(pending)*

---

## 6. Redundancy with the shipped control variates

*(pending — this is the number that decides whether it ships)*

---

## 7. What this changes in the rest of the repository

`docs/cost_floor.md` and `docs/graded.md` reduce a sampler's score to
`adjusted = v_eff · c / B` by substituting `raw = v_eff/N` into
`adjusted = raw · max(0.1, C/B)`, at which point `N` cancels. That substitution
assumes `p = 1`, i.e. **iid** sampling. For general `p`, above the clamp,

    adjusted  =  v (F₀ + λR)/(B N^p)  +  v c / (B N^{p-1})

which is flat in `N` only at `p = 1`; for `p > 1` it keeps falling and the
optimum runs to the largest affordable `N`. Both documents have been scoped
accordingly. The correction matters independently of this page's verdict, and
at the measured `p = 1.08` the practical consequence is small — a 1.08 exponent
leaves `adjusted ∝ N^{-0.08}` above the clamp, which is a few per cent over the
usable range of `N`, not a reorganisation.
