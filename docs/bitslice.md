# Bit-slicing: the lane is real, the ceiling is 22.4x, and the family still loses

Forum 18125 settles the legitimacy question. @thylinao measured the bit-packing
discount and @dipam replied for the AIcrowd team:

> "You're right about the bitwise operation optimizations, we reproduced it.
> **We are keeping the current billing and treating bit-packing as a legitimate
> optimization.** The reason is that we cannot distinguish it. A packed word
> and an ordinary integer array are byte-for-byte identical with the same
> dtype, so nothing tells the meter whether a uint32 element holds 32 boolean
> lanes or one integer value."

So this is metered, priced, blessed compute, and the discount is algorithmic.
This page prices it end to end and then bounds it. **The bound is negative, on
two independent counts, and the second one is the interesting one.**

Reproduce with `scripts/44_bitslice.py --mode probe|ranges|price|sweep|bias`,
on LOCAL MLPs (seeds 700000+) disjoint from the official suite.
`tests/test_bitslice.py` proves the packed product is an identity.

---

## 0. The verdict in one table

Two pre-registered bars, both fixed before the runs and neither re-rolled:
`v_eff · c < 59,439` (the shipped 68,400 beaten by 1.15x), and 16 bools per
billed FLOP for a packed dot. **Both FAIL**, and the two optima are in
different places:

| | shipped float32 sparse | best on `v_eff·c` | best on `adjusted` |
|---|---|---|---|
| configuration | — | `b_a=2, b_w=4, k=1.5` | `b_a=7, b_w=6, k=3.0` |
| billed FLOPs a sample (`dF/dN`, measured) | 2,847,132 | **1,421,780 (2.00x)** | 6,894,048 (0.40x) |
| `v_eff = 0.0245 + v_q` | 0.0245 | 0.0828 | 0.0302 |
| `v_eff · c` | **68,400** | 116,358 (**0.587x**) | 208,279 (0.328x) |
| rms bias | ~6e-5 | 8.2e-02 | 4.0e-03 |
| `0.1 b²` | 3.6e-10 | 6.8e-04 | 1.6e-06 |
| `adjusted = 0.1b² + v_eff c/B` | 2.4646e-07 | 6.79e-04 (**2,755x worse**) | 2.38e-06 (**9.6x worse**) |

The cost saving is real and it is bigger than any other lever in this
repository — **2.00x, machine-independent, verified in a real
`BudgetContext`**, against Strassen's 1.13x. It is also not enough, and what
kills it is not what the brief expected: the estimator that has the cost saving
is not unbiased, even though its rounding is.

---

## 1. The ceiling is 22.4x, not 32x — the popcount reduction is a third of the bill

`--mode probe`, real `BudgetContext`, a 64x7 uint32 array = 14,336 boolean lanes:

| op | billed | bools / FLOP |
|---|---|---|
| `bitwise_and(u32, u32)` | 448 | 32.0 |
| `bitwise_count(u32)` | 448 | 32.0 |
| `and` + `count` | 896 | 16.0 |
| + `sum(axis=-1)` **default accumulator** | 1,664 | 8.6 |
| + `sum(axis=-1, dtype=int32)` | 1,280 | **11.2** |

A packed dot needs three passes over the packed array, not two: AND, popcount,
and a reduction over the `w = n/32` words. On the layer shape (224 lanes, 256
outputs) one `(p,q)` bit-plane pair bills **5,120 a sample** against float32's
**114,432**, so

```
ceiling  =  22.35 / (b_a b_w)        NOT   32 / (b_a b_w)
```

and break-even against float32 is at `b_a·b_w = 22`, i.e. **b ≈ 4.7 symmetric,
not 6**. Two smaller traps on the way: the default `sum` accumulator is uint64,
billed at rate 2.0, and declaring `dtype=int32` is worth 1.30x on its own;
`fnp.view` does not exist, so uint8 → uint32 has to be gathered by hand.

**The whole pipeline runs in the reduced sandbox.** `--mode sandbox` spawns a
fresh interpreter with the repo off `sys.path` and runs quantise → pack →
AND/popcount/reduce using *nothing but* `flopscope.numpy` — no `np.`, no
`.base` (which `RemoteArray` does not have), no `.view` (which `fnp` does not
have). Every primitive is present, and the packed result matches a float32
matmul of the same codes to **0.0**, checked inside flopscope so even the
assertion needs no host numpy.

## 2. The kernel, and its price

`whestfloor/bitslice.py::bitsliced_sparse_kernel` is a complete billed forward
pass — quantise, pack, AND/popcount/reduce, dequantise — inside the existing
`tau = 2.5` sparse mask. `--mode price`, two-point `dF/dN` (which cancels the
whole per-MLP plan: pilot, masks, weight packing):

| variant | `dF/dN` | model | model/meas | x ship | resid ms |
|---|---|---|---|---|---|
| ship: float32 sparse | 2,847,132 | | | 1.000 | 42 |
| **packed a2 w4** | **1,421,780** | 1,417,428 | 0.997 | **2.003** | 185 |
| packed a3 w5 | 2,561,342 | 2,556,990 | 0.998 | 1.112 | 237 |
| packed a4 w6 | 4,021,736 | 4,017,384 | 0.999 | 0.708 | 278 |
| packed a5 w6 | 5,000,882 | 4,996,530 | 0.999 | 0.569 | 325 |
| packed a6 w6 | 5,980,028 | 5,975,676 | 0.999 | 0.476 | 390 |

`packed_layer_cost` is accurate to **0.3%**, so schedules can be costed without
re-billing each one.

Three things make the kernel work, each worth a real factor:

* **Offset-binary weights.** Carrying `q_w + 2^(b_w-1)` rather than two's
  complement makes every plane coefficient a positive `2^(p+q)` — no signed
  plane to special-case — and the offset comes back out through one per-sample
  scalar `rowsum = Σ_i q_a[i]`, a `K`-FLOP reduction *independent of the number
  of outputs*.
* **`packbits` reads its input as boolean**, so extracting bit-plane `p` is a
  single `bitwise_and(q, 1<<p)`: 1 FLOP an element instead of the 3 that
  shift+and+cast costs.
* **Kept sets round UP to a multiple of 32** so the contraction axis packs into
  whole words — the same strictly-weaker-mask device `cost_floor.md` §5.1 uses
  for Strassen.

RESIDUAL is the caveat: the packed path issues ~9 dispatches a layer for the
quantise/pack plus `b_a·b_w` for the planes against 3 for the direct path, and
its `(N, w, K_out)` intermediates are large. 185 ms against 42 ms on this box
turns the machine-independent 2.003x on `F` into ~1.34x on `C` at `N = 22000`.

## 3. Per-neuron scaling: the constraint, and what it is actually worth

A bit-sliced product returns `Σ_i q_a[i] q_w[i,j]`. **A per-input scale cannot
be pulled out of that sum**, so per-neuron scaling is not free the way it is in
a float kernel. It becomes implementable if the scales are restricted to
OCTAVES of a common base: the inputs partition into `G` octave groups, each
contracted separately and combined with one power-of-two-weighted add per group
(2 FLOPs an output per group, against `b_a b_w (3w-1)` for the core).

`--mode ranges` says the windows span p90 = 3.1 octaves and max 5.6 after the
`tau` mask, and that a single per-layer scale would carry `1/0.207 = 4.84x`
more injected variance than ideal per-neuron scaling. **That 4.84x is an
overestimate of what it is worth**: measured end to end, `G = 1 → 2 → 6` moves
`v_q` only `0.0335 → 0.0261 → 0.0249`, i.e. **1.35x, not 4.84x** — because the
downstream sensitivity is not uniform across neurons, so the wide-window
neurons that dominate `Σ r²` are also the ones that matter, and normalising
them away buys less than the range spread suggests. `G = 2` captures nearly all
of it.

## 4. The levers, measured, and what each one is worth

Starting from a naive symmetric quantiser at `kappa = 3` and going to the best
configuration, `v_q` at `b_a = 2` falls **10x**, from 0.60 to 0.058. The
ordering matters:

| lever | worth | why |
|---|---|---|
| pre-activation window `[max(0,m-ks), max(0,m+ks)]` | large | by layer 32 `rms\|alpha\| = 3.44`, so the naive `[0, m+ks]` window is `(alpha+k)s` wide where this one is `2ks` |
| **`kappa` co-optimised with `b_a`** | up to 2x | injected variance is `step²/6 + clipped tail`; only the first term falls with `b`, so the argmin moves 1.5 → 2.8 as `b_a` goes 2 → 6 |
| weight batch-mean fix `hbar @ (W - What)` | decisive for `b_w` | see §5 |
| antithetic rounding | 1.2x | see below |
| octave groups (`G=6`) | 1.35x | §3 |
| depth-graded `b_a` | ~1.05x on the product | §6 |

**Antithetic rounding, and its exact ceiling.** Draw `u` for half the batch and
`1-u` for the other. The rounding error is `e = 1{u ≥ 1-f} - f`; decomposing,
`E_f[e | u] = u - 1/2`, so `Var_u(E_f[e|u]) = 1/12` of a total `Var(e) = 1/6`.
**Exactly half the rounding variance is explained by `u`, so exactly half is
what any `u`-side variance reduction can remove** — antithetic pairing,
stratification, RQMC on the rounding uniforms, all of them. The remaining
`1/12` is conditional on the data and no choice of randomness touches it. That
is a clean ceiling and antithetic attains it for free (it also halves the
number of uniforms drawn). End to end it measures 1.2x rather than 2x, because
part of `v_q` is the weight term, which is not a function of `u` at all.

## 5. Activation error is variance; weight error is bias — and the fix

Stochastic rounding of the ACTIVATIONS is redrawn every sample, so it is
zero-mean and divides by `N`. A quantised WEIGHT is fixed for the whole run:
its error does not divide by `N` at all. Left alone, `b_w = 4` would need ~10
bits to get the bias under the budget.

The fix costs nothing. Write `h W = ĥ Ŵ + ĥ(W - Ŵ) + (h - ĥ)W`. The middle
term's batch mean is `h̄ (W - Ŵ)`, one `(1,K) @ (K,K)` product a layer a batch
— `2K²/N ≈ 4` FLOPs a sample at `N = 27000` — and folding it into the layer
bias converts the leading weight-quantisation term from bias into a per-sample
fluctuation. With it on, `b_w` is nearly free in variance terms:

| `b_w` (at `b_a = 4`) | 3 | 4 | 5 | 6 | 32 (exact) |
|---|---|---|---|---|---|
| `v_q` | 0.0974 | 0.0565 | 0.0393 | 0.0355 | 0.0345 |

`b_w = 5` is within 14% of exact weights. So the weight side is solved, and the
family's fate rests entirely on `b_a`.

## 6. Where the noise comes from: sub-additive, and front-loaded

`--mode layers` quantises one layer at a time. The per-layer contributions
decay smoothly, 11.3% at layer 1 to 1.2% at layer 32 — a 9.4x spread, so a
depth-graded schedule exists. But water-filling over a spread of 9.4x buys only
`arith/geom = 1.25x` on the variance at fixed total bits, and because cost is
linear in `b` while variance is exponential, that is **~1.05x on the product**.
Not decisive.

The per-layer contributions are also strongly SUB-ADDITIVE: they sum to 0.0536
where the joint measurement is 0.0220. Injecting noise everywhere is 2.4x
cheaper than the sum of injecting it anywhere. That is good news for the
family, and it is why the naive "32 layers x per-layer relative variance" model
over-predicts by 2.4x — but it is also the tell for §7, because the reason the
linearised model fails is that the perturbation is not small.

## 7. The obstruction: relu is convex, so variance becomes bias

**This is the result.** The brief's premise was: stochastic rounding is
unbiased, so quantisation error is variance, and variance divides by `N` like
any other Monte-Carlo variance. The first half is true. The second does not
follow.

`--mode bias`, PART A — the SAME quantiser through ONE contraction and no relu,
16 independent rounding draws on a fixed input stream of 8,192:

| `b_a` | `b_w` | rms bias | rms se | ratio | verdict |
|---|---|---|---|---|---|
| 2 | 4 | 1.908e-03 | 1.975e-03 | 0.97 | **UNBIASED** |
| 4 | 6 | 4.196e-04 | 3.785e-04 | 1.11 | **UNBIASED** |
| 6 | 8 | 8.538e-05 | 9.370e-05 | 0.91 | **UNBIASED** |

The rounding is unbiased, exactly as claimed, at every precision.

PART B — the same quantiser through `d` relu layers:

| config | d=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| a2w4, rms bias | 1.46e-02 | 4.54e-02 | 4.12e-02 | 4.52e-02 | 8.78e-02 | 6.92e-02 |
| — in sigma | 16.7 | 48.6 | 36.2 | 44.2 | 123.2 | 173.8 |
| — `0.1b²` / shipped score | 85.6 | 835 | 689 | 829 | **3,129** | **1,941** |
| a4w6, `0.1b²` / score | 38.2 | 118.7 | 170.3 | 170.8 | 61.5 | **66.3** |
| a6w6, `0.1b²` / score | 2.1 | 4.9 | 6.4 | 7.1 | 4.6 | **5.5** |

**One relu is enough.** `relu` is convex, so for `eps` with any variance at all

```
E[relu(z + eps)] - E[relu(z)]  =  (1/2) v phi(m/s) / s  +  O(v²)   >  0
```

An unbiased perturbation of the pre-activation is a BIASED perturbation of the
activation, at every one of 32 layers, each shift then amplified by everything
downstream. Unbiasedness of the ROUNDING is not unbiasedness of the ESTIMATOR,
and `0.1 b²` is charged on the second.

The size is what decides it. At `a6w6` — where the packed pass already costs
**1.76x MORE** than the float32 sparse pass — the bias term alone is still
**5.5x the entire shipped adjusted score**. There is no `(b_a, b_w)` at which
both the variance and the bias are affordable.

### 7.1 Three ways to correct it, all measured

**(a) Linearised analytic.** Propagate `v` by
`v^{l+1}_j = Σ_i [Phi_i v^l_i + fresh_i] W_ij²` and subtract
`(1/2) v phi(alpha)/s`. **Over-corrects by 7x** and leaves a bias *larger* than
the one it removes. Diagnosed: the propagation model is 0.52x at layer 1 rising
to 3.03x at layer 32, because *the perturbation is not small* — at `b_a = 4`
its sd is **half** the pre-activation sd at every layer, so nothing linearised
in it converges. (`np_noise_plan`, kept in the module as the measured negative.)

**(b) Variance matching**, `dh = relu_mean(m, s_q) - relu_mean(m, s)` with
`s_q` read off the scored batch. Needs no exact reference and is exact in `v`
rather than linearised. **It does not work either**: `v = max(0, s_q² - s²)`
must be clamped, and clamping a noisy zero-mean quantity has positive mean, so
the estimation noise becomes a systematic over-correction. Un-clamped, it needs
the pilot's `(m, s)` accurate to ~1e-4, which costs more than the pass.

**(c) Direct calibration** — run the exact and the quantised pass on the same
`n_cal` inputs and subtract the paired per-layer mean shift, correcting as you
go. **This works**, and it prices the obstruction:

| `n_cal` | rms bias | `0.1b²` / shipped score |
|---|---|---|
| 0 | 1.62e-02 | 106.6 |
| 2,000 | 4.35e-03 | 7.5 |
| 20,000 | 1.69e-03 | 1.0 |

The residual falls as `0.195 / sqrt(n_cal)`. To get the bias term to 0.1 of the
score needs `rms b ≤ 5e-4`, i.e. **`n_cal ≈ 1.5e5` EXACT forward passes = 6.4e11
FLOPs = 2.3x the entire budget `B`.** The calibration costs 7x more than the
thing it is calibrating, and it needs the very dense pass the packing was
supposed to replace.

A fourth route is identified and not built: run the mean-field relu closure
TWICE, once at `s²` and once at `s² + v`, so the closure's model error cancels
in the difference and no Monte Carlo is needed. It is the mechanistically right
answer and it is free. It is not built because §8 shows it cannot rescue the
family anyway.

## 8. The bound

Two curves, both measured, crossing in the wrong place.

```
c(b_a, b_w)  =  c_0 · b_a b_w / 22.4        (+ overheads; 0.3% accurate)
v_q(b_a)     ~  A · 4^(-b_a)                 (A ~ 5.4 at the best kappa, G, anti)
bias(b_a)    ~  0.6 · v_q                    (measured, 0.60-0.61 over b_a = 4,5,6)
```

so `adjusted = 0.1 (0.6 v_q)² + (v_eff0 + v_q) c / B`, and

* the **cost** term needs `b_a b_w < 22`, i.e. `b ≲ 4.7`;
* the **variance** term needs `v_q ≲ 0.02132 g - 0.0245`, which is already
  negative at `g = 1.1`, so it needs `b_a b_w` well under 20 AND `v_q` small —
  incompatible demands;
* the **bias** term reaches the shipped score's size at `v_q ≈ 2.6e-3`, which
  needs `b_a ≈ 7`, where `b_a b_w ≥ 42` and the packed pass costs **2x more**
  than float32.

The family is squeezed from both ends and the squeeze has no interior. The best
`v_eff · c` over `(b_a, b_w, kappa, G, antithetic, depth schedule)` is
**116,358 at `b_a = 2, b_w = 4, kappa = 1.5`, i.e. 1.70x worse than the shipped
68,400** — and that point carries a bias worth 2,755x the whole score.
Including the bias term honestly, the best adjusted score anywhere in the
family is `2.38e-06` at `b_a = 7, b_w = 6`, **9.6x worse than the ship**, and
that configuration bills **2.42x MORE per sample than the float32 sparse
pass**.

The full 18-point grid (2 local MLPs, N=8192, 6 rounding replicates, kappa at
its per-`b_a` argmin, G=6, antithetic on, weight batch-mean fix on):

| | `b_w`=4 | 5 | 6 |
|---|---|---|---|
| **`v_eff·c` / 68,400** | | | |
| `b_a=2` | **0.587** | 0.488 | 0.414 |
| `b_a=3` | 0.451 | 0.400 | 0.341 |
| `b_a=4` | 0.371 | 0.382 | 0.343 |
| `b_a=5` | 0.301 | 0.367 | 0.358 |
| `b_a=6` | 0.246 | 0.337 | 0.354 |
| `b_a=7` | 0.208 | 0.299 | 0.328 |
| **`adjusted` / 2.4646e-07** | | | |
| `b_a=2` | 0.0004 | 0.0004 | 0.0004 |
| `b_a=4` | 0.0012 | 0.0073 | 0.0105 |
| `b_a=6` | 0.0008 | 0.0142 | 0.0736 |
| `b_a=7` | 0.0007 | 0.0126 | **0.1037** |

Read the two blocks against each other. The variance objective is maximised at
the cheap corner and the score objective at the expensive one, and they never
meet: every cell is below 1.

**The mechanism that limits it, stated once.** Bit-packing buys 22.4x per
bit-pair, so it only pays below ~4.7 bits. Below ~5 bits the injected
perturbation is an O(1) fraction of the signal — at `b_a = 4` its sd is half
the pre-activation sd — and a deep ReLU network is not a linear map on
perturbations of that size. Its convexity rectifies the injected variance into
a mean shift that does not divide by `N`, and measuring that shift well enough
to subtract it costs more than a dense forward pass. **The 32x lane is real,
and it opens exactly where the network stops being locally linear.**

This is not an argument against quantisation in general; it is an argument
against quantisation as a *cost* lever in a scoring rule that charges
`0.1 b² + v_eff c / B`. If the estimator's output were a linear functional of
the activations, §7 would not exist and `b_a = 2` would win by 1.7x. The
scored functional is `E[relu(z^32)]`, and the relu is the whole problem.
