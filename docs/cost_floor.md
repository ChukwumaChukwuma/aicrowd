# The cost lever is closed: 1.13x, not 6-9x, and here is the bound

`docs/graded.md` reduced a sampler's score to one product,

```
adjusted  =  v_eff * c / B          (B = 2.72e11)
```

with `v_eff` the residual per-sample variance and `c` the **billed FLOPs per
scored sample**. We sit at `v_eff = 0.0245`, `c = 2.79e6`, product 68,400.
`docs/hermite_rank_ceiling.md` caps `v_eff` gains at 1.76x and 1.66x is
already realised, so the plan was to take 6-9x out of `c`: `2.79e6 -> ~4.8e5`.

**That is not available, and this page is the bound rather than a list of
failed attempts.** Every route was measured; the important measurement is §3,
because it prices the entire family of cheap estimators at once instead of one
construction, and it uses ORACLE statistics so nothing practical beats it.

Reproducible from `scripts/37_cost_floor.py` (`--mode linfold|gauss|mask`) and
`scripts/38_strassen.py` (`--mode price|score`), on LOCAL MLPs (seeds
700000+) disjoint from the official suite.

| route | best `c` it reaches | what it costs | verdict |
|---|---|---|---|
| linearise the decided neurons and fold them out (§2) | 456,638 (9.2x) | `b^2 = 1.4e-05` | **dead — 1,400x the budget** |
| replace a prefix of the net by a Gaussian (§3) | 524,288 (8.0x) | `b^2 = 3.8e-06` | **dead — worse than the ship at any N** |
| rank the mask by damage instead of `alpha` (§4) | 2.73e6 (1.02-1.04x) | ~0 | real, too small to ship alone |
| structured / blocked weight surrogates (§4.2) | — | — | **dead by construction** |
| **Strassen in the scored pass (§5)** | **2,515,251 (1.1296x)** | **3.0e-07 rms** | **real; residual-limited** |

---

## 1. What `c` can even be made of

The dense per-sample pass is 32 layers of `256 x 256 x 2 = 131,072` FLOPs.
Call that one **layer-equivalent**; dense is 32 of them, the shipped sparse
pass is 21.7, and the 4.8e5 target is **3.6**. Only three things can move it.

1. **Evaluate fewer neurons.** That is the shipped `alpha < -tau` mask. Its own
   optimum is `tau = 2.5` (`scripts/35`), it is worth 1.49x, and pushing it
   further injects variance faster than it saves compute.
2. **Take neurons out of the matmuls without removing them from the answer** —
   make them affine, so consecutive affine blocks compose into one precomputed
   matrix. §2.
3. **Stop doing a forward pass.** Replace a prefix of the network by a
   distribution and sample from that. §3.

Anything cheaper than ~3.6 layer-equivalents is in category 3, because
categories 1 and 2 still touch every layer.

## 2. Linearise and fold — the trade curve has the wrong shape

### 2.1 The construction

A neuron is affine to first order, not only when `|alpha|` is large:

```
relu(z)  =  a + b z + delta,     b = Phi(alpha)   (Stein: Cov(relu z, z) = Var(z) Phi)
                                 a chosen so E[delta] = 0
Var(delta) = Var(relu z) - Phi(alpha)^2 Var(z)
```

`Var(delta)/Var(z)` is **symmetric in `alpha`** — 0.0908 at `alpha = 0`, 0.0433
at `|alpha| = 1`, 0.0054 at `|alpha| = 2`. So this subsumes the shipped mask:
freezing a dead neuron at a constant is the `b = 0` special case, and the
positive side (`alpha > +tau`, exactly linear) is its mirror image. Measured
over all 32 layers, `sum Var(delta) / sum Var(relu z) = 0.118` — **linearising
a neuron injects 8.5x less variance than freezing it at its mean**, and unlike
freezing it can be folded, because the affine parts compose.

Costed properly, a fold of `k` layers with state `d` and `m` exact neurons a
layer bills

```
2 d^2  +  4 (k-1) d m  +  (k-1)(k-2) m^2          (per fold, k layers)
```

`k = 1` gives `2 d^2`, the unfolded cost, so the formula is its own ablation.
Differentiating: **the fold pays only while `m < d/2`**, i.e. while the exact
set is under half the state. That single inequality is what decides everything
below, and it is also why the naive "prune the always-ON neurons too" version
does nothing: at `tau = 2.5` the undecided set is 62.4% of the layer against
18.7% always-on, so `m/d = 0.77` and `k = 1` is optimal — no fold exists.

### 2.2 The measurement that kills it

Rank every one of the 7,936 linearisable neurons by the bias it puts on the
scored mean, `damage(l,i) = 1/2 Var(delta) sum_j (phi_j/s_j) R_l[i,j]^2` with
`R_l` the mean-field Jacobian from layer `l` to `z^32`. The question is whether
the damage is CONCENTRATED. It is not:

| exact set `\|E\|` | 8 | 32 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|
| share of damage captured | 1.9% | 5.9% | 16.8% | 27.3% | 42.8% | 62.7% | **84.4%** | **99.2%** |
| implied `b^2` | 2.6e-05 | 2.4e-05 | 1.8e-05 | 1.4e-05 | 8.7e-06 | 3.7e-06 | 6.5e-07 | **1.6e-09** |
| best fold `c`/sample | 139,326 | 164,831 | 277,999 | **456,638** | 873,112 | 1,644,988 | 2,881,675 | **4,194,304** |
| x dense | 30.1 | 25.5 | 15.1 | **9.19** | 4.80 | 2.55 | 1.46 | **1.00** |

Read the two bold columns. At `|E| = 256` the fold delivers **9.19x**, which is
better than the target — and `b^2 = 1.4e-05`, which is 1,400x the entire error
budget. At `|E| = 4096` the bias is finally affordable (`1.6e-09`) and
`|E|/31 = 132 >= d/2 = 128`, so the fold delivers **exactly 1.00x**. The curve
crosses from "unusable" to "free" without ever passing through "useful".

**Mechanism.** The linearisation residual is spread almost uniformly: 7,936
neurons, and the worst 256 of them carry 27% of the damage. That is the same
rank obstruction `docs/hermite_rank_ceiling.md` §5.2 found in the Hermite
basis, seen in a different coordinate — a ReLU network's nonlinearity lives on
8,192 independent kink surfaces and no small subset of them dominates.

## 3. The cheap-model floor — a bound on the whole family, not one scheme

To cost less than a forward pass an estimator must replace a prefix of the
network by a model. Give that model every possible advantage: let it be an
exact Gaussian with the **oracle** mean and full covariance of `z^j`, measured
from a 400,000-sample Monte-Carlo pass, then run layers `j+1..32` exactly. Cost
is `33-j` layer-equivalents a sample (one for the correlated draw). `b^2` is
measured paired, `mean_j e_1j e_2j` over two independent (surrogate, reference)
streams, which is unbiased.

3 local MLPs, 400,000 surrogate samples:

| scheme | `c` (layer-eq) | `c`/sample | x dense | `b^2` | rms `b` |
|---|---|---|---|---|---|
| closed form `m Phi + s phi` from the **exact** `(m,s)` of `z^32` | 1 | 131,072 | 32.0 | **9.04e-07** | 9.51e-04 |
| gaussian at `z^30`, 2 exact layers after | 3 | 393,216 | 10.7 | 3.25e-06 | 1.80e-03 |
| **gaussian at `z^29`, 3 exact layers after** | **4** | **524,288** | **8.0** | **3.77e-06** | 1.94e-03 |
| gaussian at `z^28`, 4 exact layers after | 5 | 655,360 | 6.4 | 4.08e-06 | 2.02e-03 |
| gaussian at `z^26`, 6 exact layers after | 7 | 917,504 | 4.6 | 5.66e-06 | 2.38e-03 |
| gaussian at `z^24`, 8 exact layers after | 9 | 1,179,648 | 3.6 | 6.87e-06 | 2.62e-03 |
| gaussian at `z^20`, 12 exact layers after | 13 | 1,703,936 | 2.5 | 8.87e-06 | 2.98e-03 |
| the full pass | 32 | 4,194,304 | 1.0 | 0 | 0 |

Three things to read off.

**Every row is worse than what we already ship, at any `N`.** `adjusted >= b^2
* max(0.1, C/B) >= 0.1 b^2`, so the `z^29` row cannot score better than
`3.8e-07` however many samples it draws, against the graded **2.6082e-07**.
Even the 1-layer-equivalent endpoint — no truncation at all, just the Gaussian
closure of the final rectifier with oracle moments — floors at `9.0e-08`
adjusted, which is only 2.9x under the ship and buys nothing, because that
scheme still has to compute `z^32` per sample. (This independently reproduces
the ledger's `Gaussian closure / Rao-Blackwell the last layer = 0.809x`.)

**The floor gets WORSE the earlier you truncate**, i.e. exactly where the
savings are: 3.25e-06 at 3 layer-equivalents rising to 8.87e-06 at 13. The
model error is injected once and then amplified by the remaining nonlinear
layers, so buying more cheapness costs more than linearly.

**This refutes the 478,000 back-solve.** Graded submission 323861 sits at raw
`4.24e-8` with `C/B = 1.12`; assuming our `v_eff` that back-solves to ~478,000
billed FLOPs/sample, i.e. 3.6 layer-equivalents. The table says a
3.6-layer-equivalent sampler has an irreducible `b^2 >= 3.8e-06` — **89,000x
its measured raw**. So 323861 is not a cheap-per-sample sampler. It is either
(a) a sampler at ordinary per-sample cost with a much better `v_eff` (raw
`4.24e-8` at `c = 2.5e6` needs `N = 1.2e5` and `v_eff = 0.005`, i.e. 5x better
than ours and past the Hermite cap, so a different CV family), or (b) not a
sampler at all — a deterministic estimator whose `4.24e-8` is model error, at
an RMS of `2.1e-4`. `docs/graded.md` §5 already argued (b) for ranks 1-3 on
independent grounds. **Either way the lever is `v_eff` or a different kind of
estimator, not `c`.**

## 4. Two smaller routes, also measured

### 4.1 A better-chosen mask is worth 1.02-1.04x

`alpha` ranks neurons by `Var(relu z)` alone; the damage a frozen neuron does
is that times its downstream sensitivity `sum_j (phi_j/s_j) R_l[i,j]^2`, and
the mask ought to be ranked by the product. Measured, the sensitivity spread
*within* a layer is small — p90/p10 is 5.4 at layer 1 falling to **1.5 at layer
31** — while `Var(relu z)` spans five orders of magnitude. So `alpha` is
already nearly the right order:

| `tau` | 3.0 | 2.5 | 2.0 | 1.5 |
|---|---|---|---|---|
| damage-ranked freezes this much more, at equal damage | 1.09-1.10x | 1.05-1.08x | 1.05-1.06x | 1.05x |
| which is worth, on `c` | 1.030-1.035x | 1.022-1.037x | 1.029-1.038x | 1.039-1.040x |

Real but ~3%, and it needs `R_l` (31 extra `n^3` matmuls, 1.0e9 = 0.4% of `B`)
plus the pilot covariance to do properly. Not shipped: the gain is inside the
run-to-run noise of the grader (~2%).

### 4.2 Structured weight surrogates are dead by construction

`W[i,j] ~ N(0, 2/width)` **iid** (`whestbench.generation.sample_mlp`). A random
Gaussian matrix is maximally incompressible: any surrogate with `p` free
parameters — low rank, block-diagonal, butterfly, sparse — captures in
expectation `p/65536` of `||W||_F^2` and loses the rest. There is no structure
in the weights to exploit; all the exploitable structure is in the
*activations*, which is what §2 and the shipped mask already address. This is
one line of algebra and it closes item 4 of the "structured/blocked sparsity"
brief without an experiment.

## 5. What IS available: Strassen, 1.1296x billed

`docs/graded.md` established that every **equivalent contraction** bills
exactly `n w (2w-1)` — `matmul`, `dot`, `inner`, `einsum ij,jk->ik`,
`tensordot`, `multi_dot` — so there is no mispriced op to arbitrage. Strassen
is not an equivalent contraction. It is a different algorithm that returns the
same product from 7 half-size multiplications instead of 8. The bill is honest
and the arithmetic really is cheaper:

```
7 x (N/2 x 128) @ (128 x 128)  +  13 elementwise (N/2 x 128) adds
   = 115,072 FLOPs/sample   against   130,816 direct    = 1.137x
```

Measured on the real kernel inside a real `BudgetContext`, over the whole
sparse pass at `tau = 2.5`:

| | `dF/dN` | `F/B` at N=22000 | rms move in `mu` |
|---|---|---|---|
| ship (direct) | 2,841,232 | 0.2324 | — |
| kept sets rounded up to even | 2,851,028 | 0.2332 | 5.4e-05 |
| **+ Strassen depth 1** | **2,515,251** | **0.2061** | 5.4e-05 |
| Strassen's own round-off | | | **3.0e-07** |

Two implementation points make it work. The activation is carried as **four
quadrant blocks** and never reassembled between layers — Strassen's outputs are
exactly those blocks and the next layer's inputs are exactly those blocks — so
no concatenate is ever billed (512 FLOPs/sample saved). And the seven
weight-side combinations are per-MLP, so they are priced in the plan and not in
`dF/dN`. Kept sets must be even to split the contraction; they are rounded
**up** (the least-dead pruned neuron goes back in), which is a strictly weaker
approximation than the shipped mask and costs 0.34% of `c`.

**What decides it is residual, not FLOPs.** Depth 1 issues 7 matmuls + 13 adds
+ 4 relu + 4 bias-adds a layer instead of 3 ops, and a flopscope dispatch costs
~26 us (elementwise) to ~106 us (matmul) of BILLED residual — `wall - backend -
overhead`, charged at 1e11 FLOP/s. Depth 2 bills 1.2699x but adds ~216 ms of
residual and is a net **1.00x**: measured and rejected.

Because the extra residual is a fixed per-predict cost while the 1.1296x is per
sample, **the trade improves with `N`** and tends to the full 1.1296x in the
large-`N` limit — which is the direction `scripts/35 --mode head` is pushing
the operating point.

## 6. Where this leaves the effort

`v_eff * c` is 68,400 and rank 4 needs 12,974. This page removes `c` from the
list of ways to get there: the best honest cut is 1.13x, the next 6x does not
exist, and the belief that a competitor found it rests on a back-solve that §3
refutes. What is left is `v_eff` — where the *Hermite* family is capped at
1.76x but `docs/hermite_rank_ceiling.md` §5.3's dictionary-free bound is not
(top-8 eigenfunctions of `Cov(y)` reach 90.1%, i.e. 10x) — and the
deterministic arm, where the leader is.
