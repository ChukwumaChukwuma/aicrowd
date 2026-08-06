# Multilevel Monte Carlo over network surrogates: measured, and dead

**Verdict: FAIL.** Pre-registered bar was a 20x improvement in cost-to-accuracy
`(Σ_k √(V_k C_k))²` over plain Monte Carlo's `V·C_full`. The best configuration
found over four surrogate families, seven ranks and every contiguous ladder is
**0.94x** — i.e. every genuine multilevel scheme is *worse* than plain sampling,
and the allocator's own optimum is to put one sample on level 0 and degenerate
back to plain MC.

Two independent obstructions, either of which alone is fatal. They are
quantified below and they generalise past rank truncation.

Scripts: `scripts/22_mlmc_variance_decay.py` (level variances),
`scripts/22b_mlmc_decorrelation.py` (the perturbation law),
`scripts/22c_mlmc_ceiling.py` (per-layer compressibility and the ceiling),
`scripts/23_mlmc_endtoend.py` (flopscope cost audit, parity, official suite).
Kernel: `whestfloor.kernels.mlmc_kernel`. Ledger row `mlmc_rank_levels`.

---

## 1. The construction, and what it needs

    E[f_full] = E[f_{r_0}] + Σ_k E[f_{r_k} − f_{r_{k−1}}]

with each difference estimated from a *coupled* pair (one input draw pushed
through both networks) and `N_k ∝ √(V_k/C_k)`. Total cost-to-accuracy is
`(Σ_k √(V_k C_k))²`, so with `v_k = V_k/V` and `c_k = C_k/C_full`

    gain = 1 / (Σ_k √(v_k c_k))²  ,   and a 20x bar needs  Σ_k √(v_k c_k) ≤ 0.2236.

Every level is a real forward pass, so this is exactly two requirements:

* **level 0 must be very cheap** — `√(v_0 c_0) ≤ 0.2236`, and any surrogate that
  reproduces the output fluctuation at all has `v_0 ≈ 1`, so `c_0 ≤ 0.05`;
* **the top level must be very well coupled** — it runs the full network, so
  `c_K ≥ 1`. Even granting a *free* level 0 this forces `v_K ≤ 0.05`, i.e.
  `ρ ≥ 0.975`; with a realistic rank-4 level 0 (`√(v_0 c_0) = 0.177`) it
  tightens to `v_K ≤ 2.2e-3`, i.e. `ρ ≥ 0.9989`.

`ρ ≥ 0.975` is therefore a **necessary** condition for the bar, and it is the
one used below, because nothing measured comes close to it.

## 2. Obstruction A — a factored layer is only cheaper below half rank

flopscope charges a dense layer `n(2n−1) + n ≈ 2n²` and a factored layer
`x @ A_r @ B_r` at `r(2n−1) + n(2r−1) + n ≈ 4nr`. So

    c(r) = 2r/n ,   and the level-0 discount alone caps the gain at n/(2 r_0) = 128/r_0.

A 20x bar therefore needs `r_0 ≤ 6`; `r = 128` costs exactly as much as the
dense layer (`4·256·128 = 2·256²`) and anything above it costs more. Measured
ceilings (a perfect surrogate, `v_K = 0`):

| r | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| ceiling | 15.8x | 7.9x | 4.0x | 2.0x | 1.0x |

**Even a perfect rank-8 surrogate cannot reach 20x.**

## 3. Obstruction B — the network decorrelates from any perturbed copy

`scripts/22b` perturbs every weight matrix by a relative amount `d`, preserving
the norm exactly (`W̃ = √(1−d²) W + d G`, `G ~ iid N(0, 2/n)`), so there is no
gain-collapse confound, and drives both networks from the same input. 3 official
MLPs, 8192 coupled samples, `V = 0.0615`:

| d | Var(f−f̃)/V | ρ |
|---|---|---|
| 1e-4 | 3.93e-6 | 0.6836 |
| 3e-4 | 3.33e-5 | 0.6835 |
| 1e-3 | 3.60e-4 | 0.6830 |
| 3e-3 | 3.25e-3 | 0.6779 |
| 1e-2 | 3.03e-2 | 0.6396 |
| 3e-2 | 1.97e-1 | 0.5060 |
| 1e-1 | 7.02e-1 | 0.2613 |
| 3e-1 | 1.18e+0 | 0.1018 |

Clean quadratic over two decades:

    Var(f − f̃)/V  =  K d²  ,   K = 393

So the network amplifies a relative weight perturbation into an output-variance
perturbation by **393x**. Inverting:

* the *necessary* condition for 20x, `v_K ≤ 0.05` (free level 0), needs
  **d ≤ 1.1e-2**;
* the realistic one, `v_K ≤ 2.2e-3` (rank-4 level 0), needs **d ≤ 2.4e-3**.

Perturbations at different layers add in quadrature, sub-linearly weighted
toward the *early* layers (perturbing only the last m layers at `d = 1e-2` gives
`V/V` of 3.19e-2 / 1.24e-2 / 3.74e-3 / 1.36e-3 for m = 32/16/8/4, against a
uniform prediction of 3.19e-2 × m/32). Early layers matter most — and they are
the least compressible.

## 4. What rank actually buys what perturbation

Two spectra decide it. `d(r) = √(1 − captured fraction)`:

| r | W singular values | d_W | act. spectrum L16 | d_act | act. spectrum L32 | d_act |
|---|---|---|---|---|---|---|
| 4 | 0.0586 | 0.970 | 0.94534 | 0.234 | 0.99111 | 0.094 |
| 16 | 0.2102 | 0.889 | 0.97892 | 0.145 | 0.99747 | 0.050 |
| 32 | 0.3762 | 0.790 | 0.98984 | 0.101 | 0.99896 | 0.032 |
| 64 | 0.6232 | 0.614 | 0.99592 | 0.064 | 0.99968 | 0.018 |
| 128 | 0.8941 | 0.326 | 0.99925 | 0.027 | 0.99997 | 0.005 |
| 192 | 0.9872 | 0.113 | 0.99997 | 0.005 | 1.00000 | 0.000 |
| 240 | 0.9998 | 0.014 | 1.00000 | 0.000 | 1.00000 | 0.000 |

* **Plain SVD truncation is hopeless.** A random square matrix has a
  quarter-circle spectrum, so meeting even the necessary condition (`d ≤ 1.1e-2`)
  takes `r = 243` of 256, and the realistic one (`d ≤ 2.4e-3`) takes `r = 251`.
  Those cost 1.90x and 1.96x the dense layer.
* **Input-adapted truncation is far better but still loses.** Projecting onto
  the top-r eigenspace of the *measured* `E[h h^T]` (an oracle: a real estimator
  could only approximate it) exploits the fact that deep activations are almost
  rank-one — at layer 32 one direction carries 98.3% of `E‖h‖²`. But layer 1's
  activation second moment is the identity (the input is isotropic), so the
  population capture is `cap_1(r) = r/n` **exactly** — 0.5 at r = 128 — and the
  early layers are *incompressible*. (The table's `d` at layer 1 is computed from
  a 4096-sample estimate of `E[x x^T]`, whose Marchenko-Pastur spread makes it
  look 0.606 rather than 0.500; the measurement is *optimistic* there, and the
  real early-layer situation is worse than shown.) Solving for the per-layer
  ranks that hit even the break-even target gives `[256, 256, 254, 249, …, 96,
  92]` — a surrogate costing **0.97x** the dense network. For the 20x target it
  costs **1.00x**. There is no saving at all.

An unrescaled SVD truncation additionally outputs literal zeros: keeping a
fraction `κ` of `‖W‖_F` gives a per-layer gain `√κ`, and `0.894^16 = 0.16`,
`0.376^16 = 6e-8`. Rescaling to fix that is measured too (`svd_rescale`) and is
*worse*, because matching the gain to better than 1% per layer is needed for
`g^32` not to blow up or collapse.

## 5. The measured level table

3 official MLPs, 8192 coupled samples, `V_full = 0.063084`. `ρ` is the coupled
correlation with the full network; the bar needs `ρ ≥ 0.975` at minimum.

| r | c(r) | svd `V(f−f_r)/V` | svd ρ | covproj (oracle) `V/V` | covproj ρ |
|---|---|---|---|---|---|
| 8 | 0.06 | 1.0000 | n/a | 1.0006 | 0.051 |
| 16 | 0.12 | 1.0000 | n/a | 1.0045 | 0.061 |
| 32 | 0.25 | 1.0000 | n/a | 0.9973 | 0.115 |
| 64 | 0.50 | 0.9998 | 0.140 | 0.9637 | 0.229 |
| 128 | 1.00 | 0.9297 | 0.298 | 0.8996 | 0.390 |
| 192 | 1.50 | 0.6370 | 0.615 | 0.7514 | 0.568 |
| 224 | 1.75 | 0.2437 | 0.875 | 0.5610 | 0.699 |

`n/a` where the unrescaled surrogate's output has collapsed to a constant, so
the correlation is undefined (and its usefulness is zero: `V(f−f_r) = V`).

The best coupling anywhere in the study is `ρ = 0.875` at `r = 224`, which costs
**1.75x the dense network**. At every rank that actually saves FLOPs
(`r < 128`) the best ρ anywhere is **0.229**. The necessary `ρ ≥ 0.975` is not
approached by anything, at any cost.

Best 2-level ladder `[r, 256]`, all four families, correct coupled variances:

| family | best r | gain |
|---|---|---|
| svd | 8 | **0.941x** |
| covproj (oracle) | 8 | 0.893x |
| sketch (random `V V^T`) | 8 | 0.622x |
| svd_rescale | 8 | 0.551x |

All below 1.0: every ladder loses to plain Monte Carlo, and the ones closest to
1.0 are the ones whose level 0 contributes nothing.

## 6. End to end on the official suite

`scripts/23`. Cost model audited against a real `flops.BudgetContext` (analytic
price within **0.2–0.4%** of the billed FLOPs, the excess being `mean`/`stack`
overhead), and `mlmc_kernel(levels=((256, N),))` is **bitwise identical** to
`mc_kernel(n_samples=N)` with identical FLOP counts — so the plain-MC arm is a
true ablation through the same code path, not a second program.

100 official MLPs, N=1e9 reference, every variant sized to 2.2e10 FLOPs with the
MLMC-optimal allocation, `C/B ≈ 0.086` so all sit at the 0.1 multiplier floor:

| variant | levels (rank, N) | C/B | pred MSE | raw MSE | adjusted | x plain (meas / pred) |
|---|---|---|---|---|---|---|
| mc_plain | (256, 5240) | 0.0829 | 9.4449e-06 | 1.0994e-05 | 1.0994e-06 | 1.000 / 1.000 |
| mlmc_8_256 | (8,1), (256,4917) | 0.0868 | 1.0065e-05 | 9.1010e-06 | 9.1010e-07 | 1.208 / 0.938 |
| mlmc_16_256 | (16,1), (256,4630) | 0.0902 | 1.0689e-05 | 9.6144e-06 | 9.6144e-07 | 1.144 / 0.884 |
| mlmc_128_256 | (128,529), (256,2229) | 0.0865 | 2.3097e-05 | 2.4393e-05 | 2.4393e-06 | 0.451 / 0.409 |
| mlmc_224_256 | (224,1672), (256,681) | 0.0864 | 4.5385e-05 | 4.9742e-05 | 4.9742e-06 | 0.221 / 0.208 |
| mlmc_192_224_256 | (192,860),(224,600),(256,426) | 0.0897 | 1.0646e-04 | 1.3808e-04 | 1.3808e-05 | 0.080 / 0.089 |

Measured and predicted agree except within the ~15% noise on a raw MSE estimated
over 100 MLPs whose 256 neurons are strongly correlated (participation ratio
~2, so ~200 effective d.o.f.). The two top rows are ties with plain MC *because
the allocator gives level 0 a single sample* — the optimum inside the MLMC
family is "do not use MLMC".

## 7. What this rules out, and what it does not

The kill is not specific to SVD truncation. Section 3 is a statement about any
surrogate obtained by perturbing the weights: `Var(f − f̃)/V = 393 · d²`, so a
usable coupling needs the surrogate's weights within `2.4e-3` relative of the
real ones, and no such surrogate is cheaper to evaluate. Combined with §2 — a
factored layer only saves below half rank — the rank-based MLMC family is
closed.

This is the same wall, reached from a third direction, that
`docs/floor_theorem.md` records for input-anchored control variates (ρ = 0.25)
and input-space RQMC. Depth-32 ReLU mixing decorrelates *anything* that is not
the network itself: an affine function of the input gets ρ = 0.25, a half-rank
copy of the network gets ρ = 0.39, and a copy costing 1.75x the original gets
ρ = 0.875.

Not ruled out, and the numbers above bear on it: the activation distribution is
extremely low-rank at depth (one direction carries 98.3% of `E‖h‖²` at layer 32,
99.9% at rank 128 by layer 16). That compressibility is real; it just cannot be
converted into a *cheaper forward pass*, because the cost of a layer is set by
its widest point and the early layers are isotropic. Any mechanism that exploits
the low-rank structure has to do so without paying `4nr` per layer.
