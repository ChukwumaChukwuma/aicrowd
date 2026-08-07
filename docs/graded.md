# The first graded submission, and what it closes

AIcrowd submission **325593**, graded 2026-08-07.

```
adjusted_final_layer_score   3.2371492938732e-07
raw_final_layer_mse          3.2332766795661884e-06
implied multiplier           0.100120
```

It passes its pre-registered bar (4.0142e-07). Two numbers in it supersede
every local measurement in this repository.

## 1. The local harness is pessimistic, not optimistic

Local `official_mini` said raw **3.7157e-06**. The grader says **3.2333e-06**.
The local harness is **15% conservative**, so every raw figure in
`ledger/experiments.jsonl` is an upper bound rather than an optimistic one.
That direction matters: it means no local result needs re-auditing for
over-claiming.

## 2. The cost frontier is worth 1.10x — via samples, not via the multiplier

The implied multiplier is `adj/raw = 0.100120`. The scoring rule is

```
adjusted = raw x max(0.1, C/B)
```

so `C/B = 0.10012` on grader hardware — **0.12% above the clamp**. Local
measurement said 0.108 and the packaging sandbox said 0.1172. Both were
inflated by `residual_wall_time_s` on slower boxes; only residual is billed at
`λ`, and the grader is fast enough that residual is much cheaper there.

That the observed multiplier is `0.100120` and not exactly `0.1000000` is the
tell that we are **unclamped**: `0.10012` is the true ratio, not a floor
reading. Splitting it against the deterministic, machine-independent FLOP
count:

```
F/B      = 0.09100      (24,751,725,967 FLOPs — exact, same on every box)
λ·R/B    = 0.00912      -> R = 24.8 ms of billed residual
C/B      = 0.10012
```

**Residual is 9.1% of our billed compute.** It is tempting to read the 0.12%
overshoot as the whole prize and call the cost frontier closed. That is wrong,
and an earlier revision of this document said so in error. Repairing the
multiplier alone *is* only `1.0012x` — but the freed budget does not vanish, it
buys samples. Driving `R -> 0` lets `N` rise by `0.1/0.0910 = 1.0989x` while
staying at the clamp, and `raw = v_eff/N` falls with it:

| | value |
|---|---|
| multiplier repair alone | 1.0012x |
| **repair + spending the freed budget on samples** | **1.1002x** |
| projected adjusted | **2.9458e-07** |

So the cost-frontier program — dispatch-count reduction, `out=` buffers (which
cut per-call residual ~6x), dropping the 31 unscored layer reductions — is
worth about **10%**. Real, worth taking, and not remotely sufficient on its
own: rank 4 needs 7x.

Three supporting negatives, each measured directly rather than inferred:

| claim | measurement |
|---|---|
| float16 is cheaper than float32 | **False.** Both bill at rate 1.000. Only float64 is penalised, at 2.0. |
| some op bills a contraction below cost | **False.** `a@b`, `matmul`, `dot`, `inner`, `einsum ij,jk->ik`, `tensordot`, `linalg.matmul`, `multi_dot` all bill exactly `n·w·(2w−1)`. |
| the RNG is worth optimising | **No.** 0.14% of budget (`standard_normal` 16 flops/elem, `random` 1/elem). |

There is no accounting exploit in the op surface. The cost model is honest, so
the 10% above is the whole of it — there is no larger cost prize hiding behind
a mispriced operation.

The graded multiplier also settles a documentation conflict: the challenge
Overview's prose writes the floor as `max(0.5, C/B)`, while
`whestbench/budget.py::score_multiplier` implements `max(0.1, C/B)`. An
observed multiplier of **0.10012** is below 0.5, so the 0.5 in the prose is
refuted empirically, not just by reading the source. The estimator is tuned to
the rule the grader actually runs.

## 3. What the clamp does to the objective

Being pinned at the clamp means the sample count is already at its optimum
`N* = 0.1B/c`: spending less compute is free but buys nothing, spending more
un-clamps the multiplier and is exactly self-cancelling. Substituting
`N = 0.1B/c` into `adjusted = (bias² + v_eff/N) · 0.1` gives

```
adjusted  =  0.1 · bias²  +  v_eff · c / B
```

**N has cancelled.** The score depends only on the *product* of residual
per-sample variance and billed cost per sample. Currently

```
v_eff · c  =  0.0317 × 2.79e6  =  88,400
```

| target | adjusted | product must fall |
|---|---|---|
| rank 4 (ednacob) | 4.62e-8 | **7.0x** |
| rank 3 (huang_chung_yi) | 1.40e-9 | 231x |
| rank 1 (dpskv5) | 4.00e-10 | 809x |

`docs/hermite_rank_ceiling.md` caps `v_eff` gains at 1.76x and 1.74x is
already realised, so essentially all of any future gain must come from `c`.

## 4. The one lever this opens

The bias enters as `0.1·bias²`. An RMS bias of **1.8e-3** would cost about what
our entire current score costs. The `tau = 2.5` sign-error bias is **2–3e-7**.

We are spending roughly **a tenth of the bias we can afford.** Every previous
`tau` sweep optimised raw accuracy, which is the wrong objective — the right
one is `0.1·b(tau)² + v_eff·c(tau)/B`, and it tolerates far more aggressive
pruning than anything shipped so far.

## 5. What this says about the leaderboard

A sampler's score is `v_eff·c/B` and cannot be driven below that by any choice
of `N`. Reaching rank 1's 4.00e-10 by sampling needs `v_eff·c = 109`, i.e.
about 6,000 billed FLOPs per sample against 4.2e6 for one dense forward pass —
**less than one row of a single matvec, for a 32-layer network.** No honest
per-sample forward pass fits in that.

So the top of the leaderboard is not sampling. It is deterministic, and its
error is model error rather than sampling error. Our own analytic arm reached
raw 2.28e-5; rank 1 implies raw 4.0e-9, which is 5,700x better model error.
That is the gap, and it is a gap in a different kind of method, not a gap in
tuning this one.
