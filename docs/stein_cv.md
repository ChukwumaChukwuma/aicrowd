# Stein control variates from the network's own gradient

**Result: measured and killed. The identity is exact — `mean(h)` lands at
0.65–1.30× the Monte-Carlo standard error over 3,072 independent tests — and
the family is worthless: the whole Stein dictionary reaches `R² = 11.5%`
against `relu(z³²_j)` and adds `+0.44` percentage points on top of the shipped
layer-1 Hermite family, against a pre-registered bar of `R² > 0.75`. There is
also an exact, Jacobian-free ceiling of `R² ≤ 1.8%` for the entire
256-direction family built on the network's own output, over *every* choice of
direction at once.** Everything here is reproducible from
`scripts/30_stein_cv.py` (`--mode verify|span`).

The trip was not wasted: the same algebra that kills it says exactly *why*,
and the reason is a property of the Hermite ladder rather than of this
network. Section 5 states it.

> **Sequel.** `docs/hermite_rank_ceiling.md` closes the *other* half of the
> board with the same kind of argument. Where §4 here bounds the Stein family
> by the raising operator, that page bounds the whole **layer-1 Hermite**
> family — every degree, every direction basis, every cross product — by the
> fact that `h_d(⟨a,x⟩)` is exactly a unit **rank-one** tensor of the
> degree-`d` chaos, hence `R² ≤ Σ_j Var(E[y_j | Aᵀx])`. Measured: 43.2%
> (1.76×) for a 2,494-feature dictionary, of which the shipped `H1+H2` at
> 39.03–39.24% already has almost all. The `39.03%` in §3 below is not a way
> station; it is the family's answer.

---

## 1. The construction, and why it looked like the way out

`docs/floor_theorem.md` bounds any surrogate whose ANOVA content sits at order
`≤ k` at `1/(1 − Σ_{d≤k} f_d)`, measured `1.38×` at `k = 1` and `1.75×` at
`k = 2`. `docs/learned_corrector.md` ships the `k ≤ 2` instance and says
plainly that it is close to its ceiling, because layer 1 is the only place in
the network where a moment is known in closed form.

Stein's identity needs no closed-form moment anywhere inside the network. For
`x ~ N(0, I_n)` and any weakly differentiable `phi : R^n -> R^n` of at most
polynomial growth,

    E[ div phi(x) ] = E[ x . phi(x) ]

so with `phi(x) = psi(x) c` for a fixed direction `c` and any scalar field
`psi`,

    h(x) = c . grad psi(x) − (c . x) psi(x)          E[h] = 0  EXACTLY

**for any `psi` whatsoever** — in particular for the network's own internals,
whose gradients cost one extra pass. That is genuinely a high-order object:
`grad psi` is piecewise constant and switches on the same order-15 sign
structure as the network, so the barrier's `k ≤ 2` cap does not apply to it,
and it is exactly unbiased by construction, which the floor argument requires.

**Forward mode, not reverse.** A control variate is needed *per output
neuron*. Reverse mode gives the whole input gradient of one scalar; forward
mode gives one directional derivative of all 256 outputs, from the recursion

    v⁰ = c ,   v^l = (v^{l−1} W^l) ⊙ 1{z^l > 0}

which is one matmul chain — the same cost as the forward pass, ten lines,
no autodiff library. So `m` directions cost `m` extra forward passes, and the
whole 256-wide dictionary `{h_i = c.grad x^l_i − (c.x)(x^l_i − E x^l_i)}` at
*every* depth comes out of a single tangent pass. By linearity of the Stein
operator in `psi`, that 256-feature block already contains `sum_i w_i x^l_i`
for every weighting `w`, random or Jacobian-aligned, so nothing is lost by not
searching over `w`.

## 2. Step 1 — the identity holds, to Monte-Carlo error

Before anything else (`--mode verify`), because a wrong tangent pass or a
violated growth condition would make every number downstream meaningless.
4 local MLPs × 200,000 samples × 3 random unit directions × 256 neurons =
**3,072 independent tests per `psi` family**:

| `psi` | rms `mean(h)` | rms MC s.e. | ratio | rms z | max&#124;z&#124; |
|---|---|---|---|---|---|
| `relu(z³²_j)` | 1.18–3.34e-3 | 1.72–3.19e-3 | 0.65–1.05 | 0.86–1.21 | 3.07 |
| `z³²_j` | 1.71–4.69e-3 | 2.22–3.61e-3 | 0.76–1.30 | 0.79–1.28 | 2.82 |
| `x¹⁶_i` | 1.57–2.44e-3 | 1.52–2.54e-3 | 0.69–1.20 | 0.84–1.13 | 3.10 |
| `x¹_i` | 2.00–2.46e-3 | 2.23–2.25e-3 | 0.90–1.10 | 0.90–1.10 | 3.83 |

Worst rms z-score over all four families and all four MLPs: **1.278**; the
largest single |z| is 3.83 over 12,288 tests, which is what the null predicts.
`mean(h)` is zero to Monte-Carlo error and nothing else. The ReLU kink set has
measure zero and the network is Lipschitz, so `relu` costs the identity
nothing — the trap is elsewhere (§5).

## 3. Step 3 — the span, which is the whole answer

`--mode span`, 4 local MLPs × 240,000 samples, two directions (the top left
singular vector of the mean Jacobian `E[x^T y]`, and one random unit vector).
Coefficients are fitted on one half of the sample and the explained variance is
read on the other, both ways, best over a ridge grid — so the numbers carry no
`p/N` in-sample inflation, which matters at `p = 2048`. Total-variance
weighted, exactly as `docs/floor_theorem.md` weights `f_d`:

| dictionary | p | held-out R² | ⇒ variance reduction |
|---|---|---|---|
| `H1` — the input-linear CV (Hermite k=1) | 256 | 25.20% | 1.337× |
| `H1+H2` — **the shipped layer-1 family** | 512 | **39.03%** | **1.640×** |
| Stein `psi = x¹_i`, dir = mean-Jacobian | 256 | 0.63% | 1.006× |
| Stein `psi = x¹⁶_i`, dir = mean-Jacobian | 256 | 0.58% | 1.006× |
| Stein `psi = x³²_i`, dir = mean-Jacobian | 256 | 2.47% | 1.025× |
| Stein `psi = z³²_i`, dir = mean-Jacobian | 256 | 11.02% | 1.124× |
| the same four, random direction | 256 ea | 0.008–0.09% | 1.000× |
| **Stein, all blocks × all directions** | 2048 | **11.53%** | **1.130×** |
| **`H1+H2` + Stein, everything** | 2560 | **39.47%** | **1.652×** |
| **CEILING, `psi = y_j`, all 256 directions** | — | **1.84%** | **1.019×** |

`H1` reproduces `f_1 = 0.276 ± 0.016` and `H1+H2` reproduces the 38–48% of
`docs/floor_theorem.md`'s amendment, from a sixth independent estimator, so the
measurement validates itself before it condemns anything.

**Read off the two numbers that decide it.** The Stein dictionary alone is
`R² = 11.5%`, against a bar of 75%. Stacked on the shipped family it adds
**+0.44 points** — 39.03% → 39.47%, i.e. 1.640× → 1.652×, for **twice the
per-sample cost**. That is a factor of 6.5 short of the bar and it is not a
tuning problem.

The one block that looks alive, `psi = z³²_i` at 11.0%, is the linear control
variate wearing a hat: `psi` there is *uncentred*, and
`−a_c^†(psi) = −a_c^†(psi − E psi) − E[psi] (c.x)`, so it carries the k=1
Hermite block as a summand. That is exactly why it collapses to nothing once
`H1` is already present.

## 4. The exact ceiling, and it costs no Jacobian at all

Write `a_i = ∂_i` and `a_i^† = x_i − ∂_i` for the annihilation and creation
operators of the Hermite basis in `L²(gamma)`, so that
`a_i^† H̃_alpha = sqrt(alpha_i + 1) H̃_{alpha + e_i}` and `[a_i, a_j^†] = δ_ij`.
The Stein CV is nothing but the raising operator:

    h = c.grad psi − (c.x) psi = − a_c^† psi ,      a_c^† = Σ_i c_i a_i^†

Three consequences, all exact.

**(i) Unbiasedness is the adjointness.** `E[h] = −<1, a_c^† psi> =
−<a_c 1, psi> = 0`, because `a_c` annihilates constants. Stein's identity *is*
the statement that `a_c^†` is the adjoint of `c.grad`. §2 is a check on the
tangent pass, not on the mathematics.

**(ii) The variance cost is the full norm of `psi`.**

    Var(h) = <psi, a_c a_c^† psi> = ‖a_c psi‖² + |c|² ‖psi‖²  ≥  |c|² Var(psi)

**(iii) The covariance is only ever an adjacent-degree pairing.**

    Cov(h, ȳ_j) = −<ȳ_j, a_c^† psi> = −<a_c ȳ_j, psi> = −E[(c.grad y_j) psi]

Now take the obvious `psi = ȳ_j`. Then `Cov = −c·k_j` with
`k_j = E[ȳ_j grad y_j]`, and one more application of Stein's identity removes
the Jacobian entirely:

    k_j = ½ E[ grad (ȳ_j²) ] = ½ E[ x ȳ_j² ]

— two `(width, width)` cross-moments, no tangent pass, no backward pass.
Combining with (ii), **for every direction `c` at once**, including the mean
Jacobian, `W³²[:,j]`, and any per-neuron choice we could not afford anyway:

    R²_j  ≤  |k_j|² / Var(y_j)²

Measured (cross-half product, so `|k_j|²` is unbiased rather than inflated by
the noise of 256 estimated components): **1.84%**, i.e. **1.019×**. Per MLP:
1.88%, 3.44%, 0.63%, 1.39%.

`k_j` is exactly one half of the order-1 Hermite coefficient vector of `ȳ_j²`.
A symmetric fluctuation has none: the square of a Gaussian is an even
function. The ledger already measured `relu(z³²)` to be nearly Gaussian at
depth — excess kurtosis 0.40, `gamma_1 ≈ 0.043` at layer 32 — so the ceiling
being ~2% is the *same* fact, seen from a new direction.

## 5. Why it fails, stated once

The brief's premise is correct and it is not enough. `a_c^†` raises the Hermite
degree by one, so a Stein CV built from an order-15 object *is* an order-16
object and the low-order barrier genuinely does not bound it. But raising the
degree is precisely what makes it useless as a control variate:

> **The raising operator moves the surrogate's mass one rung up the ladder, so
> its covariance with the target is a purely off-diagonal, adjacent-degree
> pairing, while its variance cost is the full diagonal.**

Escaping the barrier is necessary, not sufficient. The quantity that has to be
large is the *alignment* `E[(c.grad y_j) psi]`, and the function that would
maximise it is `psi ∝ c.grad y_j` itself — the tangent output, which the
forward pass already has in hand. **That one function is exactly the function
Stein's identity forbids.** For a ReLU network `c.grad y_j` is piecewise
constant and *discontinuous* across the kink hypersurfaces, hence not weakly
differentiable, hence `phi = (c.grad y_j) c'` has no valid divergence and
`E[div phi] ≠ E[x.phi]`. The construction excludes its own optimum. Nothing
between the two — activations at any depth, pre-activations, any linear
combination of them — correlates with the directional derivative well enough
to matter, and §3 measures how little: under one percentage point per block.

This is a statement about ReLU networks and the Gaussian Stein operator, not
about this suite. A smoothed gate would restore differentiability, but then
`div phi` needs second derivatives of the smoothed network, which is
`O(n³)` per sample.

## 6. What did work, and it was free

`docs/learned_corrector.md` §8 flagged that the Rao-Blackwell/Edgeworth feature
group measures at *exactly* 1.000× and costs about 1.3% of the free budget in
residual, and left it in so its own ablation table described the shipped code.
Cashing that in, with the same discipline applied to every other group that
measured 1.000×:

| group | leave-one-out (validation) | cost |
|---|---|---|
| `cv1` (Hermite k=1) | 1.480× | 2 length-N matvecs |
| `cv2` (Hermite k=2) | 1.437× | `z1²` + 2 matvecs + one 256² solve |
| `cv1mf` (mean-field) | 1.404× | 31 matvecs, 4.1e6 FLOPs |
| shape (`s, Phi, phi, a, sd_mc, dpilot`) | 1.464× | free except `sd_mc` |
| `cv3` (Hermite k=3) | 1.504× | zero at `CV_KMAX = 2` |
| **RB gap + Edgeworth** | **1.504×** | **`d³`, `d⁴`: five passes** |
| **shrink (`mu`, `mu*Phi`)** | **1.504×** | free |
| **weights + suite (`wn, w4, vbar, arms`)** | **1.504×** | `vbar` needs `mean(x²)` |
| full design | 1.504× | — |

Thirteen of the 28 columns are dropped: the whole RB/Edgeworth group, the
shrink group, the weight/suite group, `cv3`, and `sd_mc` (the last member of
the shape group that needs `vh = Var(relu z³²)`). Between them they cost
**eight passes over the (8500, 256) sample array** — `d³` and `d⁴` for the
sample-Edgeworth skew and kurtosis, and `mean(x*x)` for `Var(relu z)`. One
more pass goes because `v` is now taken as `mean(z²) − m²` rather than from a
centred copy of the array; at `rms|alpha| = 3.44` that costs 1e-6 relative on
`v`, six orders under the ~1.7e-3 residual the head predicts, and the pilot
already used exactly this form.

The head is **re-fitted** on the 15 remaining columns rather than masked, so
the ridge shrinkage is the right one for the columns that survive. On the
identical three-way split by MLP (384 train / 128 validation / 128 test):

| design | validation | TEST (read once) |
|---|---|---|
| 28 columns (previous ship) | 1.504× | 1.354× |
| **15 columns (shipped)** | **1.504×** | **1.353×** |

Same accuracy to the third decimal, eight fewer passes over the sample array.
§7 is what that is worth on the official suite.

## 7. End to end on the official suite

`scripts/30_stein_cv.py` never opens it; this table is the only place the
official suite is read, and both variants run **in the same process,
interleaved per MLP**, so the residual wall time — the only machine-dependent
term — is measured under identical load. The previous kernel is taken straight
out of `git show HEAD:whestfloor/kernels.py`, so it cannot have drifted from
what was published.

| variant | raw_mse | F/B | C/B | adj@1x | adj@2x | adj@3x | raises | worst MLP |
|---|---|---|---|---|---|---|---|---|
| previous ship (28 features) | 3.7194e-06 | 0.0920 | 0.1079 | 4.0149e-07 | 4.6096e-07 | 5.2043e-07 | 0 | 3.2805e-05 |
| `damp=0` ablation (uncorrected) | 5.8050e-06 | 0.0915 | 0.1029 | 5.9749e-07 | 6.6355e-07 | 7.2961e-07 | 0 | 4.2164e-05 |
| **LEAN ship (15 features)** | **3.7157e-06** | **0.0919** | **0.1064** | **3.9532e-07** | 4.4919e-07 | 5.0307e-07 | **0** | 3.2338e-05 |
| LEAN, N = 9000 | 3.5842e-06 | 0.0971 | 0.1125 | 4.0323e-07 | 4.5829e-07 | 5.1334e-07 | 0 | 3.4766e-05 |
| LEAN, N = 9500 | 3.5337e-06 | 0.1024 | 0.1178 | 4.1637e-07 | 4.7093e-07 | 5.2548e-07 | 0 | 3.9121e-05 |

| ratio against the previous ship, same run | adj@1x | adj@2x | adj@3x |
|---|---|---|---|
| `damp=0` ablation | 0.672× | 0.695× | 0.713× |
| **LEAN ship** | **1.0156×** | **1.0262×** | **1.0345×** |
| LEAN, N = 9000 | 0.996× | 1.006× | 1.014× |
| LEAN, N = 9500 | 0.964× | 0.979× | 0.990× |

Three things to read off.

**The saving is real and it is entirely in `C/B`.** Raw MSE is 3.7157e-6
against 3.7194e-6 — the same number, as §6's test split predicted. `C/B` falls
from 0.1079 to 0.1064, so the adjusted score improves by **1.56%** at this
box's residual, **2.62%** at twice it and **3.45%** at three times it. The gain
grows with residual, which is the direction the risk points: `F` is
machine-independent and the grader's single core may be slower than this one.

**The `damp=0` ablation is unchanged, to five significant figures.** It reads
5.8050e-6, which is exactly what `docs/learned_corrector.md` §6 published for
the previous ship's ablation and what `docs/sparse_sign_stable.md` published
for the ship before that. The feature block was deleted around the early
return, not through it, so the ablation is still the identical code path with
the head switched off — and `tests/test_submission_parity.py::
test_corrector_damp_zero_is_exactly_uncorrected` pins it bitwise *and*
FLOP-for-FLOP.

**The freed budget does not buy samples, and this is the measurement that
settles it.** `N = 9000` and `N = 9500` both lower the raw MSE (3.584e-6,
3.534e-6) and both *raise* the adjusted score, to 0.996× and 0.964×. Above the
0.1 multiplier floor the adjusted score is flat in `N` only to first order; the
sparse mask's closure bias is `N`-independent, so the second-order term is
adverse, and the worst single MLP degrades from 3.23e-5 to 3.91e-5. `N = 8500`
stays.

**The shipped file itself** (`scripts/28 --mode ship`, real protocol-3.0
`mlp.seed` per MLP, run alone on this box): raw **3.7157e-6**, `F/B`
**0.0919**, adjusted **3.9998e-7** at 1× residual / 4.5851e-7 at 2× /
5.1705e-7 at 3×, **0 raises in 100**, worst single MLP 3.2338e-5 (down from
3.2805e-5), max `C` over the suite 0.1313 of `B`, setup 0.000 s against a 5 s
window. The defensive path was probed by forcing `_sparse` to raise: it returns
a finite `(32, 256)` prediction at raw MSE 1.0873e-5 and `C/B` 0.0948. Against
the previous ship's published figure in the same mode (4.039e-7) that is
1.0098×; the interleaved A/B above is the number to trust for the ratio,
because the two shipped runs were taken minutes apart on a box whose residual
wanders by about 0.6% between runs. **Raw MSE and `F/B` are
machine-independent and both moved the right way.**

`submission/corrector.npz` is 322 bytes and regenerates **byte-identically**
from `scripts/28_learned_corrector.py --mode fit --val-frac 0.2 --install`
(sha256 `eff7515db0c1b0a2…`, checked by running it twice).

## 8. Bars, fixed before the run

| # | bar | measured | verdict |
|---|---|---|---|
| 1 | `mean(h)` is zero to Monte-Carlo error on a real 256×32 MLP | rms z-score 0.79–1.28 over 3,072 tests per family; worst 1.278 | **PASS** |
| 2 | the Stein dictionary reaches **R² > 0.75** (> 4×), materially above the k≤2 ceiling of 1.75× | **11.5%** alone, **+0.44 points** on top of the shipped family; exact ceiling 1.8% for `psi = y_j` over all 256 directions | **FAIL — not integrated** |
| 3 | dropping the measured-1.000× feature groups costs ≤ 0.5% of accuracy on the untouched test split | 1.353× against 1.354× | **PASS** |
| 4 | 0 raises on all 100 official MLPs, and a `damp=0` ablation recovering the uncorrected estimator through the identical code path, bitwise | see §7; `tests/test_submission_parity.py::test_corrector_damp_zero_is_exactly_uncorrected` | **PASS** |

## 9. Honest caveats

- **Two directions, not 256.** §3 measures the span for the mean-Jacobian
  direction and one random one. The 256-direction question is answered
  *exactly* rather than empirically, but only for `psi = y_j` (§4). For hidden
  activations at other depths the evidence is the two directions and the
  monotone pattern across them, not a proof.
- **The ceiling in §4 is for centred `psi`.** Uncentred `psi` adds
  `−E[psi](c.x)`, which is the k=1 Hermite block and is already in the shipped
  dictionary; the joint measurement in §3 is what covers that case, and it is
  the `+0.44` points.
- **`R² > 0.75` was a hard bar, and the result is not marginal.** At 11.5% the
  question is not whether a better direction or a cleverer `psi` closes it —
  the gap is 6.5×, and §4 and §5 say where the ceiling comes from.
- **The lean head is a cost win, not an accuracy win.** Raw MSE is unchanged
  to within the run-to-run noise of the estimator; everything gained is in
  `C/B`, which is the machine-dependent half of the score. Hence adj@2× and
  adj@3× are quoted in §7.
- **`tau = 2.5`, `N = 8500`, `P = 150` are still inherited, not re-swept.**
  §7 quotes `N = 9000` and `N = 9500` on the freed budget; the contract test
  pins `F/B ≤ 0.10`, which is the binding constraint on `N`.
