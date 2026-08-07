# State of play

Everything below is measured, with the producing script named. The ledger
(`ledger/experiments.jsonl`) carries the acceptance bar that was fixed *before*
each run, and the verdict against it.

## The target, corrected

| quantity | challenge docs | measured here | script |
|---|---|---|---|
| avg final-layer activation variance `v` | 0.18 | **0.0551 ± 0.0023** | `02_adv_variance.py` |
| raw MSE noise floor `v/1e9` | ~2e-10 | **5.51e-11** | — |
| adjusted floor (×0.1 multiplier) | ~1.8e-11 | **5.51e-12** | — |
| per-neuron RMS target | 1.34e-5 | **7.42e-6** | — |
| per-layer modelling budget | — | **1.31e-6** | — |

`0.18` is the **depth-8 warm-up value**, inherited across the change to depth 32
(measured: depth 4/8/16/32 → 0.329 / 0.189 / 0.100 / 0.055, and every
dataset-bake example in the whestbench docs still passes `--depth 8`). An
independent analytic route via the ReLU arc-cosine overlap map agrees. `v` also
has CV 0.48 across MLPs, so the floor is a per-suite random variable.

Free compute is `C ≤ 2.72e10` (the multiplier clamps at 0.1 below that). The
shipped estimator sits at `F/B = 0.0920` — `F` is machine-independent, `C`
is not, and the grader runs participant code on one physical core, so the
margin that matters is quoted in `F`.

**The shipped estimator is sampling with the always-off neurons pruned out of
every matmul** (`docs/sparse_sign_stable.md`), **plus layer-1 Hermite control
variates and a 15-feature offline-trained residual head**
(`docs/learned_corrector.md`, trimmed in `docs/stein_cv.md` §6): raw
**3.7157e-6**, adjusted **3.95e-7**, `F/B` **0.0919**, 0 raises in 100 — 1.56x
raw / 1.51x adjusted over the same code path with the head switched off, and
1.64x below the grader's own plain-MC constant `sampling_mse = 6.4695e-7`.

The whole analytic programme below is retained as research, not as ship:
blending it back in *worsens* the score, because its 2.7e9 FLOPs push `C/B`
from 0.100 to 0.124 and the 1.24x multiplier penalty exceeds what it buys.

## Where the error is

`scripts/04` isolates the error the Gaussian assumption injects in **one** layer
given exact inputs. Layer 1 reads 3.3e-4, which is the paired-measurement noise
floor rather than an error — `z¹` is exactly Gaussian, so the diagnostic
validates itself. Layers 2–32 read **1.0–2.4e-3**, against a 1.31e-6 budget.
Errors add incoherently across layers (√32 × 2.0e-3 = 1.13e-2 ≈ the measured
9.2e-3 end-to-end), so no rearrangement of the chain rescues a closed-form
Gaussian scheme.

At depth the network is nearly input-independent: `mean(m²)/mean(z²) → 0.95` by
layer 32, so most rectifiers are almost always on or off.

**Corrected by direct measurement** (`scripts/25 --mode alpha`, 4 official MLPs
× 200k samples): `rms|α|` at layer 32 is **3.44, not 4.4**, and it is exactly 0
at layer 1 (`E[z¹] = E[x] W¹ = 0` — the mean is *generated* by rectification and
accumulates with depth). `α` is close to Gaussian across neurons, so the
fraction with small `|α|` falls only linearly near zero. Consequently **987 of
8192 neurons flip sign per sample (12.05%)**, not the ~5% the "nearly decided"
picture suggests, and a scheme that must *evaluate* every neuron that might
flip has to touch 71.4% of them at `|α| < 3`. That single fact is what bounds
every sparse-correction scheme — see `docs/sparse_sign_stable.md`.

## Mechanisms tried

| mechanism | result | verdict | script |
|---|---|---|---|
| exact Mehler post-ReLU covariance | 6.82e-5 → 6.32e-5 (8%), saturates at k=4 | kept, marginal | `12` |
| low-rank conditional (top-k directions exact) | 2.9× vs a 100× bar | **dead** | `06` |
| Edgeworth w/ **oracle** cumulants, order ≤ 4 | 10–22× per layer vs 550× needed | ceiling | `07` |
| any reconstruction from exact κ₁..κ₆ | **8.2×** on noise-free targets | **dead** | `11` |
| analytic κ₃ via star diagrams | 84–88% of true κ₃ at O(n³) | **works** | `13` |
| κ₃ Edgeworth correction, end to end | 6.82e-5 → **3.23e-5** (2.11×) | **shipped** | `12` |
| MLMC over rank-truncated networks | 0.94× vs a 20× bar | **dead** | `22`, `23` |
| sign-stable **modal fusion** + kink correction | 4.95× MORE expensive than dense | **dead** | `25` |
| sign-stable **Rao-Blackwell** on decided neurons | 1.001× at τ=2 | **dead** | `25` |
| sign-stable **dead-neuron pruning** | 1.51×/sample → **1.44× on score** | **shipped** | `25`, `26` |
| final-layer Rao-Blackwell (Gaussian closure) | 0.81×; +Edgeworth 0.99×; 1.000× inside the fitted head | **dead** | `28` |
| **layer-1 mean gap through the mean-field Jacobian** | **1.43×** at unit coefficient | **shipped** | `28` |
| **layer-1 Hermite control variates, k ≤ 2** | **1.33×** at unit coefficient; k=3 is 1.20× | **shipped** | `28` |
| **offline-trained ridge head** over both | **1.52× adjusted on the official suite** | **shipped** | `28` |
| offline-trained **MLP** head, same features | 0.785× vs ridge (bar 1.5×) | **dead** | `28` |
| **Stein CVs from the network's own gradient** | R² = 11.5% vs a 0.75 bar; +0.44 points on the shipped family; exact ceiling 1.8% | **dead** | `30` |
| **dropping the 13 measured-1.000× head columns** | **1.016× adjusted** (1.035× at 3× residual), free | **shipped** | `28`, `30` |

### Why sign-stable sparsity mostly does not work (`docs/sparse_sign_stable.md`)

The 4× per-sample cost bar **failed at 1.51×**, and 4× is the mechanism's
*ceiling*: pruning costs `(|ON|/n)²` and `ON` must contain every neuron with
positive mean, ~50% by construction, so 4× is reachable only at `τ = 0` where
the sign error is 4.5e-2 — 7,700× the whole score. Two richer versions are dead
outright. The modal-fusion identity `z³² = xA + Σ εˡ Rˡ` is exact but `E[x] = 0`,
so `A` predicts *nothing* and the two arms cancel 15.3× (`Var 1.968` and `1.921`
summing to `0.128`), leaving the correction arm with **31× more variance than
plain MC**; billed, it costs 4.95× more per sample plus 51% of the free budget
in setup. And the decided neurons carry **0.10% of the estimator variance at
τ = 2**, so there is no Rao-Blackwellisation to collect. What survives is the
dumbest version — drop the rows and columns of neurons that never fire — worth
**1.44× on the score, replicated across four independent seeds**, which shipped
anyway despite the failed cost bar.

### Why Stein control variates are dead (`docs/stein_cv.md`)

The one construction that *provably* escapes the low-order barrier, and it is
still worth nothing. `h = c·∇ψ − (c·x)ψ` is exactly mean-zero for any Lipschitz
`ψ` and any fixed `c` — verified to Monte-Carlo error, worst rms z-score 1.278
over 3,072 tests per `ψ` family — and `∇ψ` for a network internal is as
high-order as the network. Measured span against `relu(z³²_j)`, held out:
**11.5%** for the whole dictionary (four `ψ` families × two directions × 256
source neurons), **+0.44 points** on top of the shipped layer-1 Hermite family,
against a pre-registered bar of `R² > 0.75`.

The reason is one line of operator algebra. `h = −a_c^†ψ` where `a_c^†` is the
Hermite *raising* operator, so `Var(h) ≥ |c|²‖ψ‖²` while
`Cov(h, ȳ) = −E[(c·∇y)ψ]` — a purely adjacent-degree pairing. Escaping the
barrier is necessary and not sufficient. Worse, the `ψ` that would maximise the
alignment is `c·∇y_j` itself, which for a ReLU net is discontinuous across the
kinks, hence not weakly differentiable, hence exactly the function Stein's
identity forbids: **the construction excludes its own optimum.** A second
application of Stein's identity gives an exact, Jacobian-free ceiling for the
whole 256-direction family on `ψ = y_j` — `R² ≤ |kⱼ|²/Var(yⱼ)²` with
`kⱼ = ½E[x ȳⱼ²]` — measured at **1.8%**, i.e. 1.019×.

### Why multilevel Monte Carlo is dead (`docs/mlmc.md`)

Two independent obstructions, either fatal, both measured. **Cost:** flopscope
charges a factored layer `4nr` against `2n²` dense, so `c(r) = 2r/n` and the
level-0 discount alone caps the gain at `n/(2r₀) = 128/r₀` — a 20× bar needs
`r₀ ≤ 6`, and `r = 128` already costs exactly what the dense layer does.
**Coupling:** a
norm-preserving relative weight perturbation `d` gives a clean quadratic law
`Var(f − f̃)/V = 393 d²` over two decades, so the top level needs `d ≤ 2.4e-3`,
i.e. `r = 251` of 256 at 1.96× the dense cost. The best coupling measured
anywhere is `ρ = 0.875` at `r = 224` (1.75× dense); at every rank that saves
FLOPs the best is `ρ = 0.229`, against a necessary `ρ ≥ 0.975`.
Same wall as the input-anchored control variates in `floor_theorem.md`, from a
third direction.

### Why the cumulant route is dead

`scripts/11` removes Monte Carlo from the question entirely: test distributions
are built as sums of independent rectified Gaussians, their densities computed
by FFT convolution (exact to ~1e-12), so both `E[relu]` and the cumulants are
known exactly. Feeding a reconstruction the *exact* cumulants through order 6:

```
gauss    9.00e-3     edge4    1.10e-3  (8.2x)
edge3    1.85e-3     edge6    1.82e-3  <- WORSE than edge4
gamma_g  1.34e-3     (moment-matched valid density, not a series)
```

Edgeworth-6 being worse than Edgeworth-4 shows the series diverges rather than
converges — the effective expansion parameter is `γ₁ ≈ 0.44`, not `n^{-1/2} =
0.06`. And a genuine density matched to the same moments does no better, so
this is not a bad-reconstruction problem: **the information required is not
present in the low-order cumulants.**

### The one mechanism that worked

`κ₃(S_j) = Σ_{i₁i₂i₃} W W W · κ(x_{i₁},x_{i₂},x_{i₃})` is n⁴ overall — five
times the entire free budget. Expanding each rectifier in Hermite polynomials
of its own standardised pre-activation turns the joint cumulant into a sum over
triangles with edge multiplicities `(p,q,r)`, and **every diagram with a zero
multiplicity factorises**, because `R⁽⁰⁾` is the rank-one all-ones matrix:

```
κ₃^star_j = 3 Σ_{u,v≥1} colsum_j[ G⁽ᵘ⁺ᵛ⁾ ∘ (R⁽ᵘ⁾G⁽ᵘ⁾) ∘ (R⁽ᵛ⁾G⁽ᵛ⁾) ] / (u!v!),
G⁽ᵐ⁾ = diag(a_m) W
```

— `umax` matmuls for all 256 outputs at once. Validated against brute force on
real layers: 84–88% captured. Higher `umax` degrades, so the omitted triangle
diagrams are not negligible and the series should not be pushed further.

## Honest position

Shipped: **3.7194e-6 raw / 3.99e-7 adjusted = 80,600× the floor**, 0 raises in
100, ablation-clean (`damp=0` reproduces the previous ship bitwise *and*
FLOP-for-FLOP) and parity-tested against the research kernel. That is 1.62×
below the grader's own plain-Monte-Carlo constant and 2.5× behind the best
sign-stable entry on the public board.

The barrier theorem still stands, with one amendment: it was only ever
*computed* at `k = 1`. Its `k = 2` instance is 1.75×, not 1.38×, and the
layer-1 Hermite family reaches order 2 for free because `z¹ = x W¹` is exactly
Gaussian — the one place in the network where a moment is known in closed
form. Nothing comparable exists at layer 2 or beyond, which is where this line
stops.

## Open lines

1. **Chaos-2 / generalized chi-square.** Truncating the Wiener-chaos expansion
   at order 2 gives an *exact distribution* (a generalized chi-square), not a
   moment truncation, and chaos 3+ is genuinely near-Gaussian because
   `Cov(He_k(t_i),He_k(t_j)) = k! ρ_ij^k` is strongly suppressed for `k ≥ 3`.
   This sidesteps the reason the cumulant route died. Cost is the obstacle: a
   per-output eigendecomposition is 1.4× the free budget for a single layer, so
   it needs a `log det` / saddlepoint shortcut.
2. **Fourth-cumulant diagrams**, same factorisation trick. Bounded upside
   (~2–4×) given the oracle-cumulant ceiling, but cheap.
3. ~~**Offline-calibrated residual correction.**~~ **Done** — see
   `docs/learned_corrector.md`. The Edgeworth form with learned coefficients
   turned out to be worth nothing (the final-layer Gaussian closure is *worse*
   than plain averaging at depth 32); what the learned head is actually
   weighting is a **layer-1 Hermite control variate**, which is free because
   `z¹` is exactly Gaussian and its Mehler Gram is analytic. The label-noise
   argument held: 640 MLPs at N_gt = 2×10⁵ each was enough.

## Where this leaves the board

| | adjusted | × floor |
|---|---|---|
| proven noise floor (`docs/floor_theorem.md`) | 4.949e-12 | 1.0 |
| leaderboard #1 (dpskv5) | 3.63e-10 | 73.3 |
| the sign-stable family as a competitor ships it | 1.60e-7 | 32,300 |
| **this repo, now** | **3.95e-7** | **79,900** |
| this repo, previous ship (28-feature head) | 4.01e-7 | 81,100 |
| this repo, two ships ago (sparse MC) | 5.87e-7 | 118,600 |
| grader's plain-MC constant `sampling_mse` | 6.4695e-7 | 130,700 |
| this repo, three ships ago (analytic blend) | 7.7791e-7 | 157,200 |

The top two rows of this repo's block come from **one interleaved run** in
which both estimators were scored per MLP in the same process, so the
machine-dependent residual is measured under identical load; the previous
ship's own published figure was 3.99e-7, and the 0.6% spread between the two
is what run-to-run residual noise on this box looks like. Raw MSE, which is
machine-independent, went 3.7194e-6 → 3.7157e-6.

The entire remaining headroom in the benchmark is 73.3× and we are 1,100× behind
the leader, so the gap is not a modelling gap — it is that nothing here yet
beats sampling by more than a constant factor.

## Reproducing

`scripts/00_bootstrap_env.sh` rebuilds NumPy/OpenBLAS/flopscope from Git
sources — this sandbox could reach `github.com` and nothing else (PyPI, the
Ubuntu archive, huggingface.co and aicrowd.com all return 403 at the egress
gateway), and shipped no NumPy.
