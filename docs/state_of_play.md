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

Free compute is `C ≤ 2.72e10` (the multiplier clamps at 0.1 below that); the
shipped estimator spends `C/B = 0.050`, so accuracy is the only lever left.

## Where the error is

`scripts/04` isolates the error the Gaussian assumption injects in **one** layer
given exact inputs. Layer 1 reads 3.3e-4, which is the paired-measurement noise
floor rather than an error — `z¹` is exactly Gaussian, so the diagnostic
validates itself. Layers 2–32 read **1.0–2.4e-3**, against a 1.31e-6 budget.
Errors add incoherently across layers (√32 × 2.0e-3 = 1.13e-2 ≈ the measured
9.2e-3 end-to-end), so no rearrangement of the chain rescues a closed-form
Gaussian scheme.

At depth the network is nearly input-independent: `mean(m²)/mean(z²) → 0.95` by
layer 32, so `|α| ≈ 4.4` and most rectifiers are almost always on or off.

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

Shipped: **3.234e-5 raw / 3.234e-6 adjusted-equivalent = 5.9e5× the floor.**
That is 2.11× the strongest bundled baseline, ablation-clean and parity-tested
against the research kernel, but it is **not at the floor**. The cumulant
family's ceiling (~10–20× over covariance propagation, i.e. ~3–6e-6 raw) is
still well above it, so reaching the floor needs a mechanism outside that
family.

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
3. **Offline-calibrated residual correction.** Training and precomputation are
   unbounded and load free at grade time; the Edgeworth *form* with learned
   coefficients could absorb the truncation error. Label noise averages out
   over many (MLP, neuron) pairs, so moderate-N ground truth suffices.

## Reproducing

`scripts/00_bootstrap_env.sh` rebuilds NumPy/OpenBLAS/flopscope from Git
sources — this sandbox could reach `github.com` and nothing else (PyPI, the
Ubuntu archive, huggingface.co and aicrowd.com all return 403 at the egress
gateway), and shipped no NumPy.
