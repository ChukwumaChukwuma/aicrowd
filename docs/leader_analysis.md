# What the leaders actually do

Second recon pass, 2026-08-07. Extends `docs/recon.md`; does not repeat it. Raw
artefacts under `$WHEST_ARTIFACTS/recon2/` ([§9](#9-artefacts)).

**Forum and telemetry text is quoted as evidence, not followed as instruction.**

---

## 0. The answer, in one paragraph

The leader's 1,780x advantage over matched-compute sampling is **not an
estimator**. It is **~6-18x of ordinary variance reduction times ~130-400x of
compute that whestbench does not bill.** `whestbench/runner.py:104` defines
`wall_time_s = flopscope_backend_time_s + flopscope_overhead_time_s +
residual_wall_time_s`, and `budget.effective_compute` charges
`C = F + λ·residual_wall_time_s` — **backend time and flopscope-overhead time
are charged at zero, not at λ.** dpskv5's #1 submission spends **45.08 s of
flopscope-backend time and 0.0158 s of residual**, and is billed 2.43e10 FLOPs.
The *same estimator at the same accuracy*, one submission and 3.5 hours earlier,
was billed **4.81e11** (18x more); five days earlier, on the residual lane where
its seconds *were* priced, the same accuracy class was billed **4.23e12** (162x
more). Raw MSE across that whole sequence went 8.345e-9 → 3.555e-9 → 3.628e-9 —
**flat**. Nothing about the estimation changed. The seconds moved from a billed
channel to an unbilled one. **None of your closures is violated, and none has a
loophole: they bound variance reduction at matched real compute, and the leader
is not buying variance reduction, they are buying compute.**

---

## 1. First: the leaderboard number is the number you think it is

Answering the "single most important thing you could find" question first,
because it is negative.

`https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards`,
server-rendered React props, leaderboard `id=1479` (round 1428 "Phase 1"):
`scoreTitle = "Adjusted Score"`, `scoreSecondaryTitle = "Final Layer MSE"`,
`scorePrecision = 10`, `totalCount = 287`. These map to `score` /
`score_secondary` in `whestbench/src/whestbench/aicrowd_client.py:26`.

Reproduced from the raw submission JSON of dpskv5's #1 (sub 324969),
`results.aggregates.public` against `results.public_scores` +
`results.per_mlp[].telemetry` (`recon2/subs/324969.json`):

```
mean over the 50 PUBLIC MLPs of  mse_m * max(0.1, C_m/B)   = 3.6297402518637044e-10
reported aggregates.public.score                            = 3.6297402518637034e-10
mean over the 50 public MLPs of mse_m                       = 3.6283799276226603e-09
reported aggregates.public.score_secondary                  = 3.6283799276226603e-09
```

Agreement to the last bit of a float64 mean. Also verified per MLP:
`adjusted_final_layer_score == final_layer_mse * max(0.1, effective_compute/B)`
to <1e-18 on all 50, and `C = F + 1e11·R` to **exactly 0 relative error** on all
100 MLPs of four different submissions.

**So: same metric, same formula, `B = 2.72e11`, floor 0.1, mean over the 50
public MLPs only (indices 0-49 of 100).** There is no differently-normalised
column. `per_layer_mse` is the plain unnormalised mean of squared error — an
all-zeros prediction at layer 1 scores 0.3185, and `mean_i(σ_i/√(2π))² = 2/2π =
0.3183`. Everything else in this document is measured on that same scale.

One useful corollary: **the per-MLP values are float32.** All 1,700 numbers in
`public_scores` round-trip exactly through `np.float32`. The aggregate is a
float64 mean of float32 terms.

## 2. Top 10 as of 2026-08-07 06:00 UTC, and what each has said

Board has moved since `recon.md` (#2 and #3 both improved ~4x overnight).

| # | who | adjusted | raw | mult | subs | best sub | forum posts | statement of method |
|---|---|---|---|---|---|---|---|---|
| 1 | **dpskv5** | 3.63e-10 | 3.63e-9 | 0.100 | 285 | 324969 | **0** | **none anywhere** |
| 2 | **joe_wanza** | 9.65e-10 | 3.95e-9 | 0.244 | 994 | 325329 | **0** | **none anywhere** |
| 3 | **huang_chung_yi** | 4.37e-9 | 2.99e-8 | 0.146 | 469 | 325426 | **0** | **none anywhere** |
| 4 | **ednacob** | 4.62e-8 | 9.11e-8 | 0.507 | 115 | 325177 | **0** | **none anywhere** |
| 5 | dstepanov | 5.81e-8 | 1.05e-7 | 0.553 | 461 | 325331 | 1 (queue backlog, 18128) | none |
| 6 | ely2sh | 6.26e-8 | 5.57e-8 | 1.124 | 250 | 323861 | 1 (submission cap, 18117) | none |
| 7 | thylinao | 7.88e-8 | 1.72e-7 | 0.457 | 100 | 325411 | 5 (18122, 18125#4-6, 18132) | accounting only, no method |
| 8 | SKIBIDI_TOILET | 8.37e-8 | 1.99e-7 | 0.422 | 375 | 324895 | 3 (18099, 18118) | none |
| 9 | abhinav_gorrepati | 8.59e-8 | 1.93e-7 | 0.446 | 252 | 324107 | 1 (18122) | none |
| 10 | jtel | 9.02e-8 | 2.01e-7 | 0.449 | 223 | 325223 | 2 (18076, 18099) | none |

Verified by `GET /user_actions.json?username=<u>&filter=4,5` and `/u/<u>.json`
for each (`recon2/discourse/ua_*.json`, `prof_*.json`):

> `dpskv5`: `created_at 2026-07-27T13:30:03Z`, **`last_posted_at: null`,
> `last_seen_at: null`**, 0 user actions.
> `joe_wanza`, `huang_chung_yi`, `ednacob`: same, `last_posted_at: null`, 0 posts.

**The entire top 4 has never written a word on the forum.** GitHub: no repository
matches `whestbench`, `arc-white-box-estimation`, `whest`, `flopscope`, or any
leader username other than AIcrowd's own three repos and four unrelated
participant repos (`ascender1729/whestbench-cumulant-propagation` k=3 at 1.2e-6,
`galfaroi/…`, `aarushitandon0/…`, `trainingnair/…`). The AIcrowd **Notebooks**
tab is empty, **Resources**/**Baselines** are 404, **Insights** is demographics
only. **No open-sourced submission exists.** The Phase 1 open-source obligation
only bites *after* the prize determination (Rules §6), so there is nothing to
find yet.

**Best-documented method at any level:** natasha_stewart/SOX, topic 18106, PDF
attached, adjusted 1.551e-7 — rank ~41. Nothing above 1.55e-7 has a write-up.
Nothing at 1e-8 or better has any public description at all.

## 3. The mechanism, from dpskv5's own submission history

285 submissions pulled from the challenge's own GraphQL feed
(`POST /graphql`, `challenge.submissionsPage(first:100, after:base64(offset))`,
7,000 rows back to 2026-07-24, `recon2/aicrowd/feed_index.json`), then the full
per-MLP telemetry for 14 individual submissions. Everything below is measured.

### 3.1 The invariant that makes the table readable

For **any** plain sampler, `mse = v/N` and `C = N·(cost per sample)`, so

```
mse × C  =  v × forward_pass_flops  =  0.05510 × 4,194,304  =  2.311e5
```

is a constant, independent of `N`. Measured, not assumed: a straightforward
flopscope MLP chain (`fnp.matmul` + `fnp.maximum`, 32 layers, width 256) bills
**exactly** `N × 4,194,304` — `charged/real = 1.000` at N = 256, 2,048, 8,192
(`flopscope 0.10.0+np2.3.5`, local run). And the grader's own reference constant
lands where it should: `sampling_mse = 6.4695e-7` adjusted ⇒ `mse × C = 1.76e5`
⇒ **1.3x** the invariant. So `xMC := 2.311e5 / (mse × C)` is "how many times
better than plain Monte Carlo at matched *charged* compute", and it is
calibrated against the grader.

### 3.2 dpskv5's whole run, in one table

`C` is the actual `mean_effective_compute` (not the floored multiplier);
`bk`/`oh`/`res` are mean per-MLP `flopscope_backend_time_s` /
`flopscope_overhead_time_s` / `residual_wall_time_s`; `instr = F/C`.

| submission | when | raw mse | adjusted | C | mse×C | **xMC** | instr | bk s | oh s | res s |
|---|---|---|---|---|---|---|---|---|---|---|
| 320303 | 28 Jul | 7.92e-8 | 2.48e-7 | 8.50e11 | 6.73e4 | 3.4 | 0.000 | 0.00 | 0.09 | 8.51 |
| 320375 | 28 Jul | 2.88e-8 | 1.84e-7 | 1.75e12 | 5.04e4 | 4.6 | 0.000 | 0.07 | 0.11 | 17.38 |
| 322066 | 01 Aug | 6.08e-8 | 7.56e-8 | 3.39e11 | 2.06e4 | 11.2 | 0.000 | 0.00 | 0.02 | 3.39 |
| 323473 | 04 Aug | 4.97e-8 | 6.36e-8 | 3.50e11 | 1.74e4 | 13.3 | 0.000 | 0.00 | 0.02 | 3.50 |
| **324419** | **05 Aug 17:44** | **8.35e-9** | 1.29e-7 | **4.23e12** | 3.53e4 | **6.5** | 0.031 | 0.62 | 0.08 | **39.98** |
| **324846** | **06 Aug 09:05** | **3.56e-9** | 5.83e-9 | 4.81e11 | 1.71e3 | **135** | 0.367 | 17.21 | **29.92** | 2.95 |
| **324969** | **06 Aug 12:33** | **3.63e-9** | **3.63e-10** | **2.61e10** | **9.46e1** | **2,443** | 0.933 | **45.08** | 0.23 | **0.015** |
| 325084 | 06 Aug 16:18 | 7.48e-8 | 5.70e-8 | 2.08e11 | 1.56e4 | 14.8 | 0.000 | 0.00 | 0.02 | 2.08 |

Read the three bold rows.

* **324419 → 324969: raw MSE improves 2.3x. Charged compute falls 162x.
  Wall-clock time is unchanged: 40.68 s → 45.33 s.** The seconds went from
  `residual_wall_time_s = 39.98` (billed at λ = 1e11 FLOP/s ⇒ 4.00e12 of the
  4.23e12) to `flopscope_backend_time_s = 45.08` (billed at zero).
* **324846 → 324969 is the tightest pair in the whole dataset: raw MSE
  3.555e-9 → 3.628e-9 (2% apart, i.e. the same estimator), charged compute
  4.81e11 → 2.61e10 (18.4x less), 3.5 hours apart.** The visible difference is
  `oh 29.92 → 0.23`, `bk 17.21 → 45.08`, `F 1.765e11 → 2.432e10`.
* Every dpskv5 submission before 05 Aug is **pure residual arbitrage**:
  `flops_used` mean **2.1e6** — half of one forward pass — with 2-17 s of
  residual. `instr = 0.000`. This is exactly the class Kerensa/jtel/mliston
  documented in topics 18099/18108.

**The estimator, priced against wall-clock at λ, is worth 3.4x-14.8x over plain
Monte Carlo.** That is the honest reading of every row where dpskv5's seconds
were actually charged. The 2,443x appears only in the row where they are not.

### 3.3 It is not one person's trick

| entry | raw mse | adjusted | C | **xMC** | instr | bk s | oh s | res s |
|---|---|---|---|---|---|---|---|---|
| dpskv5 324969 **#1** | 3.63e-9 | 3.63e-10 | 2.61e10 | **2,443** | 0.933 | 45.08 | 0.23 | 0.015 |
| joe_wanza 325329 **#2** | 3.95e-9 | 9.65e-10 | 6.66e10 | **878** | 0.704 | 48.28 | 3.88 | 0.192 |
| huang_chung_yi 325426 **#3** | 2.99e-8 | 4.37e-9 | 3.99e10 | **194** | 0.238 | 3.19 | 16.49 | 0.302 |
| ednacob 325177 #4 | 9.11e-8 | 4.62e-8 | 1.38e11 | **18.4** | 0.928 | 39.41 | 3.27 | 0.095 |
| dstepanov 325331 #5 | 1.05e-7 | 5.81e-8 | 1.52e11 | 14.4 | 0.000 | 0.00 | 0.03 | 1.56 |
| thylinao 325411 #7 | 1.72e-7 | 7.88e-8 | 1.25e11 | **10.7** | 0.948 | 19.24 | 2.37 | 0.061 |
| **this repo, shipped** | 3.716e-6 | 3.95e-7 | 2.89e10 | **2.1** | — | — | — | — |
| grader `sampling_mse` | 6.469e-6 | 6.47e-7 | 2.72e10 | 1.3 | — | — | — | — |

joe_wanza's own history shows the identical break: on **2026-07-27** they hit
raw **1.147e-8 at multiplier 15.878** (C = 4.32e12, `mse×C = 4.96e4`, **xMC
4.7**) — a completely ordinary sampler run 16x over budget and billed for it.
Today the same accuracy class sits at C = 6.66e10, **xMC 878**. **Their raw MSE
improved 2.9x over eleven days; their cost-efficiency improved 187x.**

huang_chung_yi went from raw 1.23e-7 (29 Jul) to **2.99e-8 today at 03:42**, in
one step, with `setup_wall_s = 20.6 s` (vs 0.01-1.3 s for everyone else) and
`flopscope_overhead_time_s = 16.5 s` against only 3.19 s of backend.

### 3.4 The structural hole, verbatim

`whestbench/src/whestbench/runner.py:104`:

```python
# wall_time_s = flopscope_backend_time_s + flopscope_overhead_time_s + residual_wall_time_s
```

`whestbench/src/whestbench/budget.py:17-24`:

```python
def effective_compute(flops_used, residual_wall_time_s, lambda_flops_per_second=1e11):
    """C_m = F_m + lambda * R_m."""
    return float(flops_used) + float(lambda_flops_per_second) * float(residual_wall_time_s)
```

Verified against the deployed evaluator on 400 per-MLP rows: `C = F + 1e11·R`
holds to **exactly zero relative error**, and `wall = bk + oh + res` likewise.
So **two of the three time channels are free**, and the leaders are in them.

The organisers' safeguard (mohanty, 18132, 2026-08-05) is written entirely about
the third:

> "The principle we plan to apply is that participant-side code should not use
> **residual time** as a second compute lane that effectively performs more than
> λ FLOP/s of numerical work while being charged only at λ (= 1e11 FLOP/s)."
>
> "A submission is not automatically ineligible merely because most of its
> charged cost comes from residual time."

dpskv5's #1 has `residual_wall_time_s = 0.0158 s` and an instrumented share of
**0.933**. It passes every stated test. And mohanty, 18122 #6:

> "on the eval servers you have flopscope-client and not the flopscope core
> package, so you wont be able to do operations directly on the array that
> bypass the accounting boundary."

Which is true, and beside the point: the top 3 are not bypassing the boundary,
they are **inside** it, using ops that bill fewer FLOPs than they compute.
flopscope's own `docs/reference/cost-model.md` §Non-exploitability names this as
the residual risk:

> "The place the cost model is genuinely sensitive is the narrower one where
> work happens *inside* the meter for a mispriced amount — which is what every
> invariant in the table above is defending."

The first public statement of it is one day old — inverted_donkey, topic
**18135**, 2026-08-07T05:25Z, unanswered:

> "I'm fairly certain that top 10 submissions on the leaderboard right now are
> noncompliant. Just looking at the corresponding timing breakdowns one can see
> that significant amounts of unregistered compute are either happening on the
> server, hidden in client's 'flopscope overhead', or billed against the
> residual."

### 3.5 Precedent: this exact family has already worked once and been patched

Topic **18082** (williawa, 2026-07-20), organiser-confirmed and fixed in
flopscope v0.9.0 (PR #150, deployed 2026-07-25):

> "When the coefficient matrix is 2-D and the RHS has a leading batch dimension,
> solve broadcasts the coefficient across the batch and performs one independent
> solve per batch — but flopscope charges only for a single batch… This lets a
> submission run a large deterministic-sample estimator (**rewriting each layer
> `A @ W` as `solve(inv(W.T), A.T)` over a batched RHS**) while being billed for
> a small fraction of the real FLOPs."
>
> williawa #2: "Also, **the current top submission exploits this hack.**"
> mohanty #3: "We have replicated this issue, and will be folding in a fix…
> once the fix is in, the affected submissions will automatically be evaluated
> using the patched flopscope."
> keenanpepper #5 (25 Jul): "this fix has clearly been deployed; big leaderboard
> upset."

`linalg.solve` is now billed `2n³/3 + 2n²·nrhs` with the batch included
(`cost-model.md:1049`). **The August leaders' rise (05-07 Aug) is after that
fix, so it is a different underpriced route — but the family, the payoff shape,
and the organiser response are all precedented.**

### 3.6 How much real arithmetic is hiding in those 45 seconds

Two independent estimates, both crude but concordant.

*From accuracy.* `mse = 3.628e-9` needs `N_eff = v/mse = 1.52e7` effective
samples. At dpskv5's own demonstrated honest efficiency (xMC 6.5-14.8), that is
`C_real = 2.31e5/(xMC · mse)` = **4.3e12 – 9.8e12** real FLOP-equivalents,
against **2.43e10 billed**. **Under-billed by 175x-400x.**

*From the timeline.* On 05 Aug the same accuracy class cost 4.23e12 charged, 95%
of it residual — i.e. 40 s of real single-core work at λ. On 06 Aug it costs
2.61e10 with a *longer* wall clock. **162x, measured, with no throughput
assumption at all.**

*What it is not.* It is not merely "having backend time": **ednacob also has
39.4 s of backend time and only reaches xMC 18.4.** The unbilled wall clock is
open to everyone; the top 3 additionally have operations that bill far below
their arithmetic. **I could not identify the specific operation.** flopscope
0.10.0's hardening is thorough (weight tiers ∈ {0,1,4,16}, no algorithm constant
in a weight, alias parity, free tier limited to views/metadata, gather at tier
4, complex/width packing priced to be losing, `svd(k=)` capped). The remaining
candidates I could not exclude, in order of plausibility: **symmetry-tagged
operands** (`unique_elements_for_shape` bills unique elements; PR #171
"fix(symmetry): degrade to dense billing when a group defeats enumeration" is
1 Aug and there are 28 open issues I could not read — `api.github.com` is
gated in this session and `github.com` HTML returns 403); **`identity_pattern`
repeated-operand savings** ("the joint symmetry of `A @ A`"); the documented
`svd(k=)` under-bill (`4mnk` billed, full economy SVD computed —
`_flops.py:svd_cost`); and `fnp.random.*` (a Gaussian draw bills 1/element and
costs ~20-40 real). This is the one part of the mechanism I am reporting as
*inferred from accounting identities*, not *identified*.

### 3.7 Two smaller facts that fall out

**dpskv5's #1 emits exact zeros for layers 2-31.** `all_layers_mse = 0.7437`,
and its `per_layer_mse[1..30]` is bit-identical to ednacob's and thylinao's,
which are the zero-prediction values `mean_i(μ_{l,i}²)`. Layer 1 is the exact
closed form (`per_layer_mse[0] = 6.773e-10 = v₁/1e9`, the GT floor). Its 08-06
09:05 predecessor 324846 filled *every* layer to ~1e-6. **They deleted 30 layers'
worth of reductions to save charged FLOPs** — which only makes sense if the
sample array is large enough that a `mean` over it is a material cost, i.e.
N ≳ 10⁶. That is a second, independent pointer at a very large real N.

**Every top entry's final layer is 100-1,500x better than its own layer-31
estimate**, which is impossible for a single sampler (`mse_l = v_l/N` is flat in
`l` near the end). Implied `N = v_l/mse_l`:

| | l=1 | 2 | 4 | 8 | 16 | **32** |
|---|---|---|---|---|---|---|
| huang 325426 | 7.9e3 | 4.8e3 | 2.4e3 | 1.4e3 | 1.2e3 | **1.8e6** |
| dstepanov | 2.1e3 | 1.7e3 | 1.5e3 | 1.4e3 | 1.4e3 | **5.2e5** |
| joe_wanza | 2.9e5 | 1.7e5 | 8.8e4 | 4.6e4 | 3.7e4 | **1.4e7** |
| dpskv5 324846 | 1.0e9 | 6.3e5 | 2.8e5 | 1.6e5 | 1.2e5 | **1.6e7** |

The intermediate columns are a cheap by-product (an analytic chain, or a small
pilot); the whole budget goes into the one scored row. Do not read the
intermediate layers as evidence about anyone's sample count.

## 4. Reconciliation with your closures

**Nothing here contradicts anything you have proved. Every closure survives with
its scope intact.** Taking them in turn.

**(a) ANOVA-order cap `1/(1 − Σ_{d≤k} f_d)`, `f₁ = 0.276`, `f_{≤2} = 0.43`
(`floor_theorem.md`).** Bounds a control variate's variance reduction at matched
sample count. The leaders' *estimators* land at xMC 4.7-18.4 — but xMC is a
compute ratio, not a variance-reduction ratio, and most of it is cost per sample
(dead-neuron pruning, float32, sparsity, dropping 30 layers), not correlation.
No leader's estimator needs a correlation above what the k≤2 family allows.
**No contradiction, and no measurement anywhere in this recon is even in tension
with 1.75x.**

**(b) Stein CVs with `φ = ψ·c` cap at `R² ≤ 1.8%` (`stein_cv.md` §4).** Nobody is
doing Stein CVs. Untouched.

**(c) The whole layer-1 Hermite family caps at `R² = 43.2%` (1.76x) at any
degree and basis (`hermite_rank_ceiling.md` §5.4).** Untouched, and it is the
tightest and most transferable thing in the repo. It is also **not the binding
constraint on your score** — see §5.

**(d) MLMC over rank-truncated networks, `Var(f − f_r)/V = 393 d²`
(`mlmc.md`).** Untouched. Note one adjacent door that this does *not* close and
that the leaders may be behind: jamesrahenry's own conclusion in 18097 was
*"A low-rank method must either beat λᵢ/N in the leading directions **or convert
rank into FLOP reduction**."* Your MLMC page prices the second half
(`c(r) = 2r/n`, so r=8 is a 16x cost cut) and rejects it *as an MLMC coupling*,
which is a different question from using it as a straight cost reduction. That
is worth at most ~16x, not 1,000x, so it is not the leader's mechanism — but it
is the largest legitimate lever the repo has priced and then not taken.

**(e) Degree spectrum `f = (26.7, 19.0, 11.2, 7.0, 4.7, 3.3, …)%`, mean degree
10.5.** Untouched; it is a property of the target, and nobody is disputing it.

**And the floor theorem itself is safe.** I tested the two ways it could have
been circumvented and both are closed:

* *Is the reference published?* No. The 100 graded MLPs are named and fixed
  (`patricia-hawkins`, `angela-walker`, … — byte-identical `mlp_name` lists
  across every submission from 2026-07-27 to 2026-08-07). I fingerprinted them
  by the zero-prediction MSE profile `mean_i(r_{l,i}²)` for l = 1..31, which
  ednacob and thylinao publish for free by emitting zeros, and matched against
  **both** published splits — `mini` (100 rows, already cached) and **`full`
  (1,000 rows, range-fetched from all 28 parquet shards, 52 MB of GETs,
  `recon2/full_alm.npz`)**. Best relative distance over 1,100 candidates:
  **1.33e-2**, median 2.19e-2, zero matches below 1e-3, and the best-match map is
  not injective. (Control: `mini`'s own 100 rows are likewise absent from `full`,
  disjoint seed sets, so the fingerprint discriminates.) **The graded ground
  truth is private and no lookup is possible.**
* *Is the reference re-baked per submission (so the floor is per-run)?* No, it is
  fixed. dpskv5 (06 Aug) and natasha_stewart (late Jul) both compute layer 1 in
  closed form; their **per-MLP** layer-1 MSEs agree to a max relative difference
  of **8.9e-4** across all 50 public MLPs. Two independent 1e9 bakes would differ
  by ~9% per MLP — which is exactly what the *published* `all_layer_means`
  does against them (correlation 0.11, ratios 0.65-1.35). So the grader's
  reference is one fixed array, and `v/N` is a fixed constant per MLP.

**One correction for the repo, and it matters:**
`scripts/19_fetch_official_suite.py` builds `official_mini.npz` from the
published `mini` split and its docstring says *"Against this suite the plain
`final_layer_mse` **is** the leaderboard number."* It is not the leaderboard's
suite — the graded 100 are a disjoint, private set. The suite remains a valid
i.i.d. sample from the same generator (same protocol, same width/depth), so
every *distributional* conclusion in the repo stands, and `raw_mse` measured on
it is an unbiased estimate of what the grader would report. But it is a sample
of 100, not the graded 100, and per-MLP quantities (`worst MLP`, the ±7% floor
CV) carry a second layer of sampling error that the docstring's wording hides.

### 4.1 So where did your 1,090x actually go

```
this repo   xMC   2.1
                   |  x 8.8   legitimate estimator/cost engineering you have not done
best honest xMC  18.4   (ednacob, instrumented share 0.93, residual share 0.07)
                   |  x 133   compute inside the meter, billed at ~1/133 of its arithmetic
dpskv5      xMC 2,443
```

`8.8 × 133 = 1,170`, against the 1,088x you measured. **The gap is ~9x of
technique and ~130x of accounting, and it is the accounting that dominates.**

## 5. What this changes for you

1. **Stop treating 3.63e-10 as an estimation target.** It is not reachable by
   estimation. `docs/state_of_play.md`'s closing line — *"the gap is not a
   modelling gap"* — is right, and this recon says exactly what it is instead.
   The reachable, defensible target is the **honest instrumented frontier at
   xMC ≈ 18** (ednacob), which at the 0.1 multiplier floor is
   `raw = 2.311e5/(18.4 × 2.72e10) = 4.6e-7`, **adjusted ≈ 4.6e-8** — around
   rank 4-5 on today's board, and 8.6x better than the current ship.
2. **You have ~9x of legitimate headroom and it is all in cost per sample, not
   in variance reduction.** Your `mse × C = 1.08e5` vs ednacob's 1.26e4. Your
   own ceilings say the variance-reduction half is nearly exhausted (1.65x
   shipped against a 1.76x family ceiling). So the 9x has to come from the other
   factor: fewer charged FLOPs per sample. The priced-but-untaken items in the
   repo are the rank-8 factored layer (`c(r) = 2r/n` = 16x, `mlmc.md`), the
   float32 audit, and dropping the 31 unscored layers' reductions (which dpskv5
   demonstrably did between 324846 and 324969).
3. **Weight the private re-evaluation heavily in the nomination decision.**
   mohanty, 18132: *"a submission deliberately built around unmetered numerical
   computation that exceeds what the residual-time charge is intended to
   represent may be adjusted or re-evaluated. In cases of deliberate
   circumvention, it may be disqualified."* And 18125: *"During final
   evaluation, we may also make targeted changes to the cost model if we
   identify an exploit."* The precedent (18082 → v0.9.0 → "big leaderboard
   upset", 25 Jul) is that ARC patches and **re-grades**. Phase 1 prize rankings
   are decided *exclusively* by the private re-run. A 2,443x entry whose xMC
   collapses to ~6 under a corrected cost model lands around rank 60.
   **The realistic competitive question is not "how do I reach 3.6e-10", it is
   "where do I rank once the accounting is corrected", and there the answer is
   the honest frontier at xMC 18.**
4. **A hole nobody appears to be using, flagged for completeness and not
   recommended.** The 100 MLPs are fixed and named, the reference is fixed, and
   every submission publishes its per-MLP `final_layer_mse` in float32 to anyone
   without authentication. 256 coordinate-probe submissions (one per output
   neuron, all 50 public MLPs probed simultaneously since each returns a
   separate MSE) would recover the public reference vectors to ~1e-17. Nobody is
   doing it: dpskv5's raw MSE plateaus at **bit-identical float64 values across
   57 consecutive graded submissions** (`6.121702437411614e-08`, 02 Aug 17:03 to
   03 Aug 21:45; also 33x `6.075530631477477e-08` and 32x `7.856660833027718e-08`), which means bit-identical predictions and therefore no probing;
   and probing would give ~1e-17, not 3.6e-9. It would also be worthless against
   the private re-run and is plainly barred by Rules §12. Recording it because
   it is the one channel that could in principle break the floor theorem, and it
   is open.

## 6. What I did not find

Stated plainly, per the brief.

* **No statement of the leader's method exists anywhere.** Not on Discourse
  (0 posts from the top 4), not on AIcrowd (no notebooks, no resources, no
  insights), not on GitHub, not in any linked PDF. The mechanism in §3 is
  reconstructed entirely from submission telemetry and the whestbench/flopscope
  source, not from testimony.
* **I did not identify the specific mispriced flopscope operation.** §3.6 lists
  the candidates I could not exclude and says why (GitHub issue access is gated
  in this session; 28 open flopscope issues unread).
* **No method at 1e-8 or better is documented by anyone.** The best-documented
  method at any score is natasha_stewart/SOX at 1.551e-7 (topic 18106, PDF),
  rank ~41; then radiant-allomancer at ~3e-7 (18085) and evaaaz at 4.10e-7
  (18053). `docs/recon.md` §7 already catalogues all three and nothing has been
  added to the forum since.

## 7. Corrections to `docs/recon.md`

* **§1.2 "#1 dpskv5 is genuine"** — the reasoning ("94% instrumented, at the
  multiplier floor, 45 s of flopscope backend time") is exactly inverted. Those
  three facts *are* the mechanism: 94% instrumented means they left the residual
  lane, and 45 s of backend time is the unbilled channel they moved into.
  dpskv5's own history (§3.2) settles it: identical estimator, 162x less charged
  compute, same wall clock.
* **§1.2 "#2 joe_wanza is half arbitrage"** — now 0.704 instrumented, 48.3 s
  backend, 0.19 s residual. Same migration.
* **§7 tier-A ordering.** The board's structure says the dominant lever is cost
  per charged sample, not any of A1-A8. A1 (final-layer Rao-Blackwell) cannot be
  worth the 100-1,500x that §3.7's table shows, because that table's
  intermediate columns are not sample counts.
* **§10 `official_mini` provenance** — see the correction at the end of §4.

## 8. Leaderboard drift since `recon.md` (24 h)

| | 06 Aug adj | 07 Aug adj | | 06 Aug raw | 07 Aug raw |
|---|---|---|---|---|---|
| dpskv5 | 3.63e-10 | 3.63e-10 | | 3.63e-9 | 3.63e-9 |
| joe_wanza | 3.31e-9 | **9.65e-10** | | 4.37e-9 | 3.95e-9 |
| huang_chung_yi | 2.14e-8 | **4.37e-9** | | 2.14e-7 | 2.99e-8 |
| ednacob | 4.72e-8 | 4.62e-8 | | 9.31e-8 | 9.11e-8 |
| thylinao | 8.30e-8 | 7.88e-8 | | 1.72e-7 | 1.72e-7 |

Total submissions 15,512 → 15,874 in 12 hours. `totalCount` 286 → 287.
#2 and #3 improved 3.4x and 4.9x *adjusted* with raw improving 1.1x and 7.2x —
the same signature.

## 9. Artefacts

Under `$WHEST_ARTIFACTS/recon2/` (outside the repo):

```
aicrowd/leaderboards.html, leaderboard_props.json   full board, all 3 rounds, 200 nodes + cursor
aicrowd/feed/p*.json, feed_index.json               7,000 submissions back to 2026-07-24 via POST /graphql
aicrowd/tab_{insights,discussion,notebooks}.html    empty of method content
aicrowd/p_dpskv5.html                               participant page (no id, no links)
subs/{320303,320375,322066,323473,324419,324846,
      324900,324969,325084,325177,325329,325331,
      325411,325426}.{html,json}                    full per-MLP telemetry
discourse/ua_*.json, prof_*.json                    per-user post history for the top 10
discourse/cat_p*.json, t18135.json, t18122.json,
          t18132.json, t18099.json, t18108.json     forum, incl. the one new topic
discourse/s_*.json                                  13 keyword searches
full_alm.npz, fetch_full.py                         all 1,000 `full`-split all_layer_means (52 MB of ranged GETs)
```

Key URLs:

- Leaderboard: `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards`
- dpskv5 #1: `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324969`
- dpskv5 the hour before: `…/submissions/324846`, `…/submissions/324419`
- The `solve` precedent: `https://discourse.aicrowd.com/t/there-is-a-bug-in-flopscope-numpy-linalg-solve-that-undercounts-batched-right-hand-sides/18082`
- Residual-lane rulings: `…/18122`, `…/18132`, `…/18125`
- The one-day-old accusation: `https://discourse.aicrowd.com/t/how-exactly-is-phase-1-submission-validation-going-to-work/18135`
- Cost model: `AIcrowd/flopscope` `docs/reference/cost-model.md` §Non-exploitability, §The meter boundary
- Billing identity: `AIcrowd/whestbench` `src/whestbench/budget.py:17-24`, `src/whestbench/runner.py:104`
