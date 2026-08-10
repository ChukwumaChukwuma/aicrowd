# The cost lever is closed: 1.13x, not 6-9x, and here is the bound

> **Scope: this page is about IID samplers.** The identity below is obtained by
> substituting `raw = v_eff/N` into `adjusted = raw * max(0.1, C/B)`, at which
> point `N` cancels. `raw = v_eff/N` is the `p = 1` case. For a sampler with
> `raw = v/N^p`, above the clamp,
>
> ```
> adjusted  =  v (F0 + lambda R) / (B N^p)   +   v c / (B N^(p-1))
> ```
>
> which is flat in `N` only at `p = 1`; at `p > 1` it keeps falling and the
> optimum runs to the largest affordable `N` rather than sitting anywhere
> interior. Every "no sampler can reach it" conclusion on this page is
> therefore a statement about **iid** samplers.
>
> How much that matters is now measured rather than assumed. `docs/rqmc.md` §2
> sweeps `N` over seven doublings on the official MLPs and fits
> `p = 0.974 +- 0.034` for iid (the control) and `p = 1.043 +- 0.023` for a
> randomly-shifted rank-1 lattice. So the exponent is 1.04, not 2: above the
> clamp `adjusted ~ N^-0.04`, 3% over a doubling of `N`, and the identity below
> survives as an excellent approximation with `v_eff` read at the operating `N`.
> `docs/rqmc.md` §7 re-optimises `N` from scratch on the official 100-MLP suite
> and confirms the optimum does not move — and finds, separately, that
> `N ~ 100,000` raises `BudgetExhaustedError` on 92 of 100 MLPs, so `N` is hard
> capped near 85,000 whatever the exponent turns out to be.
>
> What the lattice does buy is a **constant** — 1.60-2.65x (mean 2.00x) on the
> bare sampler's variance at matched `N`, and 1.448x on the shipped estimator's
> raw MSE end to end — and a constant on `v_eff` is exactly what this page's
> product is made of. It is priced in `docs/rqmc.md` §6, not here.
>
> And one calibration from that sweep which bears on every A/B on this page:
> three runs of the identical estimator at identical `N`, differing only in pilot
> size, spread over raw MSE with **CV 8.8%**, and the `raw·N` scatter across the
> whole `N` sweep is 7.7% — the same number independently. **A single-seed
> 100-MLP local raw carries ~8% of realisation noise.** So the 1.02-1.04x of §4.1
> and the 1.0121x of §5.1 are inside it and are correctly described here as
> unshippable; a `b^2` fitted from two points of an `N` sweep is inside it too
> (`docs/rqmc.md` §7.1 retracts one).

`docs/graded.md` reduced a sampler's score to one product,

```
adjusted  =  v_eff * c / B          (B = 2.72e11)     [p = 1 only]
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
sample, **the trade improves with `N`**. Medians of 5 interleaved repeats, one
thread, `tau = 2.5`, whole `predict`:

| `N` | ship `C/B` | ship + chunk16k | **strassen `C/B`** | x C vs ship | x C vs best direct |
|---|---|---|---|---|---|
| 8,500 | 0.0985 | 0.0985 | 0.1017 | **0.968** | 0.968 |
| 22,000 | 0.2431 | 0.2447 | 0.2317 | **1.049** | 1.049 |
| 45,000 | 0.5295 | 0.4925 | 0.4751 | **1.115** | 1.037 |
| 90,000 | 1.0639 | 0.9802 | 0.9401 | **1.132** | 1.043 |

Read two things off it. Strassen **loses at `N = 8500`** — 800 extra dispatches
against too few samples to amortise them — and the gain rises monotonically to
the full billed 1.13x by `N = 90000`. And past `N ~ 35000` the *direct* path
hits its own cache cliff, which chunking fixes for free (`ship+chunk16k`,
1.075x at 45,000); against that repaired baseline Strassen is 1.037-1.043x. So
the two devices overlap and should be combined, not compared: chunking is
wired into the Strassen path too (answer-invariant to 3e-6, because chunking
changes which rows are paired in `(A11 + A22)` and therefore the rounding, not
the arithmetic), and pricing that combination is the open follow-up.

The shipped operating point is `N = 27000`, where this table puts Strassen at
~1.06x, and `scripts/35 --mode head` is pushing `N` up — which is the direction
that makes it worth more.

### 5.1 The even-mask tax, and how to remove it

`--mode score`, official 100-MLP suite, `N = 27000`, `P = 600`, `N=1e9`
reference so `raw_mse` is leaderboard-comparable:

| variant | `raw_mse` | `F/B` | `C/B` | adj@1x | adj@2x | adj@3x | raises |
|---|---|---|---|---|---|---|---|
| ship: direct, mask as shipped | 1.1810e-06 | 0.2936 | 0.3078 | 3.6345e-07 | 3.8019e-07 | 3.9693e-07 | 0 |
| direct, kept sets rounded to even | 1.1873e-06 | 0.2948 | 0.3095 | 3.6745e-07 | 3.8486e-07 | 4.0227e-07 | 0 |
| **STRASSEN depth 1 (+ even sets)** | 1.1873e-06 | **0.2612** | 0.3025 | **3.5909e-07** | 4.0803e-07 | 4.5697e-07 | 0 |
| gain over the ship | | **1.124x** | | **1.0121x** | 0.932x | 0.868x | |

Paired on the identical stream, the answer moves by **2.6e-07 rms** from
Strassen and **4.2e-05 rms** from the even mask.

So the machine-independent 1.124x on `F/B` survives intact and everything else
eats it. **Rounding the kept sets up to even is 0.989x on its own**: +0.34% of
`F` and +0.5% of raw. The +0.5% is not a real accuracy loss — un-pruning the
least-dead neuron is a strictly weaker approximation — it is realisation noise
of exactly the size the change makes (4.2e-05 rms is 1.8e-09 of MSE against a
raw of 1.18e-06, 0.15%). But it is a real 1.1% off the top, and it is
**avoidable**: instead of un-pruning a neuron, pad the sliced weight with a
zero column and the next layer's slice with a zero row. That gives an even
contraction at the same FLOP cost with no mask change at all, so the only
remaining perturbation is Strassen's own 2.6e-07 rms.

The rest is residual, and the adj@2x/adj@3x columns are the honest warning:
this box's residual is what turns 1.124x of `F` into 1.0121x of `C`, and at
twice it Strassen is a **loss**. That is why it is not shipped yet. Both fixes
are known and both point the same way — remove the even-mask tax, and run at
the larger `N` the pilot work has unlocked, where the fixed dispatch cost
amortises (1.132x at `N = 90000`).

Until then Strassen is **default-off in the research kernel** and the shipped
estimator is untouched: `corrected_sparse_kernel(strassen=False)` is bitwise
and FLOP-for-FLOP the previous ship (asserted against the untouched
`sparse_mc_kernel`).

## 6. Where this leaves the effort

`v_eff * c` is 68,400 and rank 4 needs 12,974. This page removes `c` from the
list of ways to get there: the best honest cut is 1.13x, the next 6x does not
exist, and the belief that a competitor found it rests on a back-solve that §3
refutes. What is left is `v_eff` — where the *Hermite* family is capped at
1.76x but `docs/hermite_rank_ceiling.md` §5.3's dictionary-free bound is not
(top-8 eigenfunctions of `Cov(y)` reach 90.1%, i.e. 10x) — and the
deterministic arm, where the leader is.

### 6.1 And the leader really is deterministic — measured, not inferred

§3 argued from an irreducible-`b^2` table that submission 323861 "is not a
cheap-per-sample sampler". The leaderboard telemetry now says it directly, for
the whole top of the board (`scripts/52_leader_rate.py`, `docs/rqmc.md` §1).
Four of dpskv5's graded submissions ran at `N_eq = F/4.198656e6 = 0.5` — half a
dense forward pass of billed compute for the entire 100-MLP prediction — and
scored raw `5.0e-8` to `7.9e-8`, where a sampler with `v = 0.045` scores 0.09.
And regressing `ln raw` on `ln N_eq` *within* an entry, which eliminates `v`,
gives `p = 0.310` (dpskv5, 4 submissions over `N_eq` 152 to 42,028),
`p = 0.814` (huang_chung_yi) and `p = -0.151` (joe_wanza). The sharpest pair is
dpskv5 cutting compute 7.26x in 3.5 hours and moving raw by +2.1%
(`p = 0.010`), which took their adjusted score from 5.83e-9 to 3.63e-10 —
16.1x, all of it from the multiplier reaching the clamp.

`raw` flat in `N` is a bias floor. So the top of the board is spending its
whole optimisation on `C`, exactly as this page says the endgame must be for
anyone whose `raw` has stopped moving — and its `raw` is model error, which is
the arm this page does not bound.
